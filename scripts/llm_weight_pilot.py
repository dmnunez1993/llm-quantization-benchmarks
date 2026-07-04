from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DIM = 24
TARGET_BITS_PER_DIM = 2.0
LEECH_MAX_SHELL = 13
QTIP_CONFIG = dict(L=8, K=int(TARGET_BITS_PER_DIM), V=1, tlut_bits=8, decode_mode="quantlut")
QTIP_GIT_URL = "https://github.com/Cornell-RelaxML/qtip.git"


@dataclass
class QuantizerReport:
    method: str
    rate_bits_per_dim: float
    mse: float
    rmse: float
    sqnr_bits: float
    quantize_seconds: float
    dequantize_seconds: float


def synchronize_for_timing() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def per_vector_mse(x: np.ndarray, recon: np.ndarray) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    recon64 = recon.astype(np.float64, copy=False)
    return np.mean((x64 - recon64) ** 2, axis=1)


def per_vector_sqnr_bits(x: np.ndarray, recon: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    x64 = x.astype(np.float64, copy=False)
    recon64 = recon.astype(np.float64, copy=False)
    signal_power = np.mean(x64**2, axis=1)
    noise_power = np.mean((x64 - recon64) ** 2, axis=1)
    return 0.5 * np.log2(np.maximum(signal_power, eps) / np.maximum(noise_power, eps))


def timed_call(fn, *args):
    synchronize_for_timing()
    start = time.perf_counter()
    result = fn(*args)
    synchronize_for_timing()
    return result, time.perf_counter() - start


class LeechQuantizer:
    def __init__(self, max_shell: int = LEECH_MAX_SHELL):
        self.backend = "cpu"
        self.device = "cpu"
        if torch.cuda.is_available():
            try:
                from leech_lattice_vector_quantizer.leech_lattice_vector_quantizer_gpu import (
                    LeechLatticeVectorQuantizerGpu,
                )

                self.quantizer = LeechLatticeVectorQuantizerGpu(max_shell=max_shell, verbose=False, device="cuda")
                warm_x = torch.zeros((1, DIM), dtype=torch.float32, device=self.quantizer.device)
                warm_idx = self.quantizer.quantize(warm_x)
                _ = self.quantizer.dequantize(warm_idx, check_bounds=False)
                torch.cuda.synchronize()
                self.backend = "gpu"
                self.device = str(self.quantizer.device)
            except Exception as exc:
                print(f"Leech GPU unavailable, falling back to CPU: {type(exc).__name__}: {exc}")
                from leech_lattice_vector_quantizer.leech_lattice_vector_quantizer import (
                    LeechLatticeVectorQuantizer,
                )

                self.quantizer = LeechLatticeVectorQuantizer(max_shell=max_shell, verbose=False)
        else:
            from leech_lattice_vector_quantizer.leech_lattice_vector_quantizer import LeechLatticeVectorQuantizer

            self.quantizer = LeechLatticeVectorQuantizer(max_shell=max_shell, verbose=False)
        self.rate = self.quantizer.shape_bits / DIM

    def encode_normalized(self, samples: np.ndarray):
        if self.backend == "gpu":
            x = torch.as_tensor(samples, dtype=torch.float32, device=self.quantizer.device)
            with torch.no_grad():
                return self.quantizer.quantize(x)
        indices = np.empty(samples.shape[:-1], dtype=np.int64)
        for i, row in enumerate(samples):
            with contextlib.redirect_stdout(io.StringIO()):
                indices[i] = self.quantizer.quantize(row)
        return indices

    def decode_normalized(self, indices) -> np.ndarray:
        if self.backend == "gpu":
            with torch.no_grad():
                recon = self.quantizer.dequantize(indices, check_bounds=False)
            return recon.detach().cpu().numpy().astype(np.float32)
        out = np.empty((*np.asarray(indices).shape, DIM), dtype=np.float32)
        for i, idx in enumerate(np.asarray(indices).reshape(-1)):
            out.reshape(-1, DIM)[i] = np.asarray(self.quantizer.dequantize(int(idx)), dtype=np.float32)
        return out


class UniformScalarQuantizer:
    def __init__(self, bits: int = 2):
        self.bits = int(bits)
        self.levels = 1 << self.bits
        self.qmin = -(self.levels // 2)
        self.qmax = self.levels // 2 - 1
        self.rate = float(bits)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = "gpu" if self.device.type == "cuda" else "cpu"

    def encode_normalized(self, samples: np.ndarray):
        if self.device.type == "cuda":
            x = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                return torch.clamp(torch.round(x), self.qmin, self.qmax).to(torch.int16)
        return np.clip(np.round(samples), self.qmin, self.qmax).astype(np.int16)

    def decode_normalized(self, codes) -> np.ndarray:
        if torch.is_tensor(codes):
            return codes.to(torch.float32).detach().cpu().numpy()
        return np.asarray(codes, dtype=np.float32)


def ensure_qtip_repo(repo_path: Path, auto_clone: bool = True) -> Path:
    if repo_path.exists():
        return repo_path
    if not auto_clone:
        raise FileNotFoundError(repo_path)
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", QTIP_GIT_URL, str(repo_path)], check=True)
    return repo_path


def load_qtip_bitshift_module(qtip_repo: Path):
    sys.modules.setdefault("lib", types.ModuleType("lib"))
    codebook_stub = types.ModuleType("lib.codebook")
    codebook_stub.kdict = {}
    sys.modules["lib.codebook"] = codebook_stub
    sys.modules.setdefault("lib.utils", types.ModuleType("lib.utils"))

    kernel_check = types.ModuleType("lib.utils.kernel_check")
    kernel_check.has_kernel = lambda *args, **kwargs: False
    sys.modules["lib.utils.kernel_check"] = kernel_check

    kernel_decompress = types.ModuleType("lib.utils.kernel_decompress")
    kernel_decompress.decode_compressed = lambda *args, **kwargs: None
    sys.modules["lib.utils.kernel_decompress"] = kernel_decompress

    matmul_had = types.ModuleType("lib.utils.matmul_had")
    matmul_had.matmul_hadU_cuda = lambda *args, **kwargs: None
    matmul_had.matmul_hadUt_cuda = lambda *args, **kwargs: None
    sys.modules["lib.utils.matmul_had"] = matmul_had

    module_path = qtip_repo / "lib" / "codebook" / "bitshift.py"
    spec = importlib.util.spec_from_file_location("qtip_bitshift_pilot", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(module_path)

    original_compile = torch.compile
    torch.compile = lambda fn=None, *args, **kwargs: (lambda f: f) if fn is None else fn
    try:
        spec.loader.exec_module(module)
    finally:
        torch.compile = original_compile
    return module


class QTIPQuantizer:
    def __init__(self, repo_path: Path, config: dict):
        module = load_qtip_bitshift_module(repo_path)
        self.config = dict(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = "gpu" if self.device.type == "cuda" else "cpu"
        self.cb = module.bitshift_codebook(**self.config).to(self.device)
        self.cb.fakeinf = self.cb.fakeinf.to(self.device)
        self.rate = float(self.config["K"])

    def encode_normalized(self, samples: np.ndarray):
        x = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x_t = x.T.contiguous().to(torch.float16)
            t = x_t.shape[0]
            roll_x = torch.roll(x_t, t // (2 * self.cb.V) * self.cb.V, 0)
            states = self.cb.quantize_seq(roll_x, overlap=None)
            overlap = states[t // (2 * self.cb.V)] >> self.cb.K * self.cb.V
            states = self.cb.quantize_seq(x_t, overlap=overlap)
            return states.T.contiguous().to(self.device)

    def decode_normalized(self, states) -> np.ndarray:
        states = states.to(self.device) if torch.is_tensor(states) else torch.as_tensor(states, device=self.device)
        with torch.no_grad():
            states_t = states.T.contiguous()
            batch = states.shape[0]
            dim = states.shape[1] * self.cb.V
            recon = self.cb.recons(states_t).transpose(0, 1).reshape(dim, batch).T.contiguous()
        return recon.detach().cpu().numpy().astype(np.float32)


def load_weight_blocks(
    repo_id: str,
    filename: str,
    tensor_name: str,
    n_blocks: int,
    normalize: str,
) -> tuple[np.ndarray, dict[str, object]]:
    path = hf_hub_download(repo_id, filename=filename)
    needed = n_blocks * DIM
    with safe_open(path, framework="pt", device="cpu") as f:
        if tensor_name not in f.keys():
            raise KeyError(f"{tensor_name!r} not found. Available first keys: {list(f.keys())[:12]}")
        tensor = f.get_tensor(tensor_name).flatten()[:needed].to(torch.float32)

    blocks = tensor.cpu().numpy().astype(np.float32).reshape(n_blocks, DIM)
    raw_rms = np.sqrt(np.mean(blocks.astype(np.float64) ** 2, axis=1))
    if normalize == "block_rms":
        blocks = blocks / np.maximum(raw_rms[:, None].astype(np.float32), 1e-12)
    elif normalize == "global_rms":
        blocks = blocks / max(float(np.sqrt(np.mean(blocks.astype(np.float64) ** 2))), 1e-12)
    elif normalize != "none":
        raise ValueError(f"unknown normalization: {normalize}")

    metadata = {
        "repo_id": repo_id,
        "filename": filename,
        "tensor_name": tensor_name,
        "n_blocks": n_blocks,
        "dim": DIM,
        "normalization": normalize,
        "raw_weight_mean": float(np.mean(tensor.cpu().numpy().astype(np.float64))),
        "raw_weight_std": float(np.std(tensor.cpu().numpy().astype(np.float64))),
        "raw_block_rms_mean": float(np.mean(raw_rms)),
        "raw_block_rms_min": float(np.min(raw_rms)),
        "raw_block_rms_max": float(np.max(raw_rms)),
    }
    return blocks, metadata


def make_quantizers() -> list[dict[str, object]]:
    qtip_repo = ensure_qtip_repo(ROOT / "third_party" / "qtip")
    leech = LeechQuantizer()
    uniform = UniformScalarQuantizer(bits=2)
    qtip = QTIPQuantizer(qtip_repo, QTIP_CONFIG)
    return [
        {
            "method": f"Leech Lattice Vector Quantization ({leech.backend})",
            "rate": leech.rate,
            "encode": leech.encode_normalized,
            "decode": leech.decode_normalized,
        },
        {
            "method": f"Uniform scalar 2-bit ({uniform.backend})",
            "rate": uniform.rate,
            "encode": uniform.encode_normalized,
            "decode": uniform.decode_normalized,
        },
        {
            "method": f"QTIP bitshift ({qtip.backend})",
            "rate": qtip.rate,
            "encode": qtip.encode_normalized,
            "decode": qtip.decode_normalized,
        },
    ]


def run_pilot(items: list[dict[str, object]], x: np.ndarray, warmup: int, order_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(order_seed)
    rows = []
    warm_x = x[:1]
    for item in items:
        for _ in range(warmup):
            codes = item["encode"](warm_x)
            _ = item["decode"](codes)
    synchronize_for_timing()

    for block_id in range(x.shape[0]):
        x_run = x[block_id : block_id + 1]
        order = list(range(len(items)))
        rng.shuffle(order)
        for idx in order:
            item = items[idx]
            codes, quantize_seconds = timed_call(item["encode"], x_run)
            recon, dequantize_seconds = timed_call(item["decode"], codes)
            rows.append(
                {
                    "block_id": int(block_id),
                    "method": item["method"],
                    "rate_bpd": float(item["rate"]),
                    "mse": float(per_vector_mse(x_run, recon)[0]),
                    "sqnr_bits": float(per_vector_sqnr_bits(x_run, recon)[0]),
                    "quantize_seconds": float(quantize_seconds),
                    "dequantize_seconds": float(dequantize_seconds),
                    "original_vector": json.dumps(x_run.reshape(-1).astype(float).tolist(), separators=(",", ":")),
                    "dequantized_vector": json.dumps(np.asarray(recon).reshape(-1).astype(float).tolist(), separators=(",", ":")),
                }
            )

    data = pd.DataFrame(rows)
    summary = (
        data.groupby("method", as_index=False)
        .agg(
            rate_bpd=("rate_bpd", "mean"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            sqnr_bits_mean=("sqnr_bits", "mean"),
            sqnr_bits_std=("sqnr_bits", "std"),
            quantize_seconds_mean=("quantize_seconds", "mean"),
            dequantize_seconds_mean=("dequantize_seconds", "mean"),
        )
        .sort_values("sqnr_bits_mean", ascending=False)
        .reset_index(drop=True)
    )
    return data, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small quantizer pilot on real LLM weight blocks.")
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--filename", default="model.safetensors")
    parser.add_argument("--tensor-name", default="model.layers.0.mlp.down_proj.weight")
    parser.add_argument("--n-blocks", type=int, default=30)
    parser.add_argument("--normalize", choices=("block_rms", "global_rms", "none"), default="block_rms")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--order-seed", type=int, default=123)
    args = parser.parse_args()

    x, metadata = load_weight_blocks(
        args.repo_id,
        args.filename,
        args.tensor_name,
        args.n_blocks,
        args.normalize,
    )
    items = make_quantizers()
    data, summary = run_pilot(items, x, warmup=args.warmup, order_seed=args.order_seed)

    safe_repo = args.repo_id.replace("/", "__")
    out_dir = ROOT / "results" / "llm_weight_pilot" / f"{safe_repo}_{args.normalize}_{args.n_blocks}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "pilot_experiment.csv"
    summary_path = out_dir / "summary.csv"
    metadata_path = out_dir / "metadata.json"
    data.to_csv(data_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Input: {args.repo_id}:{args.tensor_name}")
    print(f"Normalization: {args.normalize}")
    print(f"Blocks: {args.n_blocks} of dimension {DIM}")
    print(f"Raw block RMS mean: {metadata['raw_block_rms_mean']:.8f}")
    print(f"Pilot CSV: {data_path}")
    print(f"Summary CSV: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
