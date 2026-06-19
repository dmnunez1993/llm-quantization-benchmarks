from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


DIM_E8 = 8


@dataclass(frozen=True)
class GaussianQuantizerResult:
    method: str
    rate: float
    mse: float
    sqnr_bits: float
    scale: float
    metadata: dict[str, float | int | str]


def mse_to_sqnr_bits(mse: float) -> float:
    return -0.5 * math.log2(mse)


def optimize_scale(
    samples: np.ndarray,
    quantize_normalized,
    scales: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    best_scale = float(scales[0])
    best_mse = float("inf")
    for scale in scales:
        recon = float(scale) * quantize_normalized(samples / float(scale))
        mse = float(np.mean((samples - recon) ** 2))
        if mse < best_mse:
            best_scale = float(scale)
            best_mse = mse
    return best_scale, best_mse


class UniformScalarQuantizer:
    """Clipped scalar uniform quantizer with an optimized global scale."""

    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError("bits must be positive")
        self.bits = bits
        self.levels = 1 << bits
        self.qmin = -(self.levels // 2)
        self.qmax = self.levels // 2 - 1
        self.rate = float(bits)

    def quantize_normalized(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x), self.qmin, self.qmax)

    def evaluate(
        self,
        samples: np.ndarray,
        scales: Sequence[float] | np.ndarray | None = None,
    ) -> GaussianQuantizerResult:
        if scales is None:
            scales = np.linspace(0.05, 3.0, 160)
        scale, mse = optimize_scale(samples, self.quantize_normalized, scales)
        return GaussianQuantizerResult(
            method="Uniform",
            rate=self.rate,
            mse=mse,
            sqnr_bits=mse_to_sqnr_bits(mse),
            scale=scale,
            metadata={"bits": self.bits, "levels": self.levels},
        )


class CubicE8Quantizer:
    """
    Cubically bounded E8 lattice quantizer.

    The finite codebook is E8 intersected with the coordinate cube
    [-radius, radius]^8. A global scale is optimized for Gaussian MSE.
    """

    def __init__(self, radius: float):
        if radius < 0.5:
            raise ValueError("radius must be at least 0.5")
        self.radius = float(radius)
        self.codebook = enumerate_cubic_e8(self.radius)
        if self.codebook.size == 0:
            raise ValueError("empty E8 codebook")
        self.rate = math.log2(self.codebook.shape[0]) / DIM_E8

    def quantize_blocks_normalized(self, blocks: np.ndarray) -> np.ndarray:
        return quantize_blocks_to_codebook(blocks, self.codebook)

    def quantize_normalized(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[-1] % DIM_E8 != 0:
            raise ValueError("last dimension must be divisible by 8")
        blocks = samples.reshape(-1, DIM_E8)
        return self.quantize_blocks_normalized(blocks).reshape(samples.shape)

    def evaluate(
        self,
        samples: np.ndarray,
        scales: Sequence[float] | np.ndarray | None = None,
    ) -> GaussianQuantizerResult:
        if scales is None:
            scales = np.linspace(0.05, 3.0, 96)
        scale, mse = optimize_scale(samples, self.quantize_normalized, scales)
        return GaussianQuantizerResult(
            method="E8 (cubic)",
            rate=self.rate,
            mse=mse,
            sqnr_bits=mse_to_sqnr_bits(mse),
            scale=scale,
            metadata={"radius": self.radius, "codebook_size": int(self.codebook.shape[0])},
        )


class QuipSharpE8Quantizer:
    """Official QuIP# E8P/RVQ codebook wrapper for Gaussian codebook tests."""

    RATE_BY_NAME = {
        "E8P12": 2.0,
        "E8P12RVQ3B": 3.0,
        "E8P12RVQ4B": 4.0,
    }

    def __init__(self, codebook_name: str, repo_root: str | Path | None = None):
        if codebook_name not in self.RATE_BY_NAME:
            raise ValueError(f"unknown QuIP# codebook {codebook_name!r}")
        self.codebook_name = codebook_name
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        Codebook = load_quip_codebook_class(self.repo_root, codebook_name)
        self.codebook = Codebook(inference=False)
        self.rate = self.RATE_BY_NAME[codebook_name]

    def quantize_normalized(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[-1] % DIM_E8 != 0:
            raise ValueError("last dimension must be divisible by 8")
        blocks = torch.tensor(samples.reshape(-1, DIM_E8), dtype=torch.float32)
        with torch.no_grad():
            recon = self.codebook.quantize(blocks, return_idx=False)
        return recon.cpu().numpy().reshape(samples.shape)

    def evaluate(
        self,
        samples: np.ndarray,
        scales: Sequence[float] | np.ndarray | None = None,
    ) -> GaussianQuantizerResult:
        if scales is None:
            scales = np.linspace(0.2, 2.0, 96)
        scale, mse = optimize_scale(samples, self.quantize_normalized, scales)
        return GaussianQuantizerResult(
            method="QuIP# official E8P/RVQ",
            rate=self.rate,
            mse=mse,
            sqnr_bits=mse_to_sqnr_bits(mse),
            scale=scale,
            metadata={"codebook": self.codebook_name},
        )


def enumerate_cubic_e8(radius: float) -> np.ndarray:
    integer_values = range(-int(math.floor(radius)), int(math.floor(radius)) + 1)
    points: list[tuple[float, ...]] = []

    def rec_integer(prefix: list[int], parity_sum: int) -> None:
        if len(prefix) == DIM_E8:
            if parity_sum % 2 == 0:
                points.append(tuple(float(v) for v in prefix))
            return
        for value in integer_values:
            rec_integer(prefix + [value], parity_sum + value)

    rec_integer([], 0)

    half_min = math.ceil(-radius - 0.5)
    half_max = math.floor(radius - 0.5)
    half_shifts = range(half_min, half_max + 1)

    def rec_half(prefix: list[float], parity_sum: int) -> None:
        if len(prefix) == DIM_E8:
            if parity_sum % 2 == 0:
                points.append(tuple(prefix))
            return
        for shift in half_shifts:
            value = shift + 0.5
            if abs(value) <= radius + 1e-12:
                rec_half(prefix + [value], parity_sum + shift)

    rec_half([], 0)
    return np.array(points, dtype=np.float32)


def quantize_blocks_to_codebook(blocks: np.ndarray, codebook: np.ndarray, chunk: int = 4096) -> np.ndarray:
    out = np.empty_like(blocks, dtype=np.float32)
    codebook = codebook.astype(np.float32, copy=False)
    code_norm = np.sum(codebook * codebook, axis=1)
    for start in range(0, blocks.shape[0], chunk):
        stop = min(start + chunk, blocks.shape[0])
        b = blocks[start:stop].astype(np.float32, copy=False)
        distances = np.sum(b * b, axis=1, keepdims=True) + code_norm[None, :] - 2.0 * b @ codebook.T
        out[start:stop] = codebook[np.argmin(distances, axis=1)]
    return out


def load_quip_codebook_class(repo_root: Path, name: str):
    quip_root = repo_root / "third_party" / "quip-sharp"
    if not quip_root.exists():
        raise FileNotFoundError(
            "Missing third_party/quip-sharp. Clone https://github.com/Cornell-RelaxML/quip-sharp there."
        )

    module_by_name = {
        "E8P12": ("latticee8_padded12.py", "E8P12_codebook"),
        "E8P12RVQ3B": ("latticee8_padded12_rvq3bit.py", "E8P12RVQ3B_codebook"),
        "E8P12RVQ4B": ("latticee8_padded12_rvq4bit.py", "E8P12RVQ4B_codebook"),
    }
    filename, class_name = module_by_name[name]
    install_quip_import_stubs()
    module_path = quip_root / "lib" / "codebook" / filename
    spec = importlib.util.spec_from_file_location(f"quip_sharp_{name.lower()}", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(module_path)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def install_quip_import_stubs() -> None:
    sys.modules.setdefault("quiptools_cuda", types.SimpleNamespace())
    sys.modules.setdefault("lib", types.ModuleType("lib"))
    sys.modules.setdefault("lib.utils", types.ModuleType("lib.utils"))
    matmul_had = types.ModuleType("lib.utils.matmul_had")

    def _not_needed(*args, **kwargs):
        raise RuntimeError("Hadamard CUDA kernels are not needed for Gaussian codebook evaluation")

    matmul_had.matmul_hadU_cuda = _not_needed
    matmul_had.matmul_hadUt_cuda = _not_needed
    sys.modules["lib.utils.matmul_had"] = matmul_had
