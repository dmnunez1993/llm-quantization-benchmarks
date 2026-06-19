from __future__ import annotations

import ctypes
from functools import lru_cache

import torch
from cuda.bindings import driver, nvrtc


CUDA_SOURCE = r"""
extern "C" __device__ long long comb_count(long long counts[16], const long long* comb, int max_vals) {
    long long remaining = 0;
    for (int i = 0; i < max_vals; ++i) remaining += counts[i];
    long long out = 1;
    for (int i = 0; i < max_vals; ++i) {
        long long chosen = counts[i];
        out *= comb[remaining * 25 + chosen];
        remaining -= chosen;
    }
    return out;
}

extern "C" __device__ void unrank_multiset(
    long long rank,
    const long long* counts_ptr,
    const long long* values_ptr,
    int class_id,
    const long long* comb,
    int max_vals,
    long long seq[24]
) {
    long long counts[16];
    long long values[16];
    int width = 0;
    for (int i = 0; i < max_vals; ++i) {
        counts[i] = counts_ptr[class_id * max_vals + i];
        values[i] = values_ptr[class_id * max_vals + i];
        width += (int)counts[i];
    }
    for (int pos = 0; pos < 24; ++pos) {
        seq[pos] = 0;
        if (pos >= width) continue;
        for (int vi = 0; vi < max_vals; ++vi) {
            if (counts[vi] <= 0) continue;
            counts[vi] -= 1;
            long long branch = comb_count(counts, comb, max_vals);
            if (rank < branch) {
                seq[pos] = values[vi];
                break;
            }
            rank -= branch;
            counts[vi] += 1;
        }
    }
}

extern "C" __device__ long long even_lower_sign_count(
    const long long compact[24],
    int prefix_len,
    long long target
) {
    long long fixed = 0;
    int f0 = 0;
    int f1 = 0;
    for (int pos = 0; pos < prefix_len; ++pos) {
        long long mag = compact[pos];
        if (mag == 0) continue;
        fixed += mag & 7LL;
        if ((mag & 3LL) == 2LL) ++f1;
        else ++f0;
    }
    long long delta = (target - fixed) & 7LL;
    if (delta != 0 && delta != 4) return 0;
    if (f1 == 0) return delta == 0 ? (1LL << f0) : 0;
    return 1LL << (f0 + f1 - 1);
}

extern "C" __global__ void llvq_dequantize_kernel(
    const long long* indices,
    long long* out,
    const long long* class_starts,
    const int* parity,
    const long long* perm_count,
    const long long* sign_count,
    const long long* f1_perm,
    const long long* golay_start,
    const long long* golay_indices,
    const int* golay_bits_table,
    const long long* f0_counts,
    const long long* f1_counts,
    const long long* odd_counts,
    const long long* f0_values,
    const long long* f1_values,
    const long long* odd_values,
    const long long* comb,
    long long n,
    int n_classes,
    int max_vals
) {
    long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    long long index = indices[row];
    int lo = 0;
    int hi = n_classes;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (class_starts[mid] <= index) lo = mid;
        else hi = mid;
    }
    int class_id = lo;
    long long local = index - class_starts[class_id];
    long long pcount = perm_count[class_id];
    long long permutation_choice = local % pcount;
    long long cursor = local / pcount;
    long long gchoice;
    int golay_bits;

    for (int d = 0; d < 24; ++d) out[row * 24 + d] = 0;

    if (parity[class_id] == 0) {
        long long scount = sign_count[class_id];
        long long sign_choice = cursor % scount;
        gchoice = cursor / scount;
        int golay_index = (int)golay_indices[golay_start[class_id] + gchoice];
        golay_bits = golay_bits_table[golay_index];

        long long f1p = f1_perm[class_id];
        long long f1_rank = permutation_choice % f1p;
        long long f0_rank = permutation_choice / f1p;
        long long f0_seq[24];
        long long f1_seq[24];
        unrank_multiset(f0_rank, f0_counts, f0_values, class_id, comb, max_vals, f0_seq);
        unrank_multiset(f1_rank, f1_counts, f1_values, class_id, comb, max_vals, f1_seq);

        int seen0 = 0;
        int seen1 = 0;
        int nz = 0;
        long long compact[24];
        int compact_pos[24];
        for (int d = 0; d < 24; ++d) {
            int bit = (golay_bits >> d) & 1;
            long long mag = bit == 0 ? f0_seq[seen0++] : f1_seq[seen1++];
            if (mag != 0) {
                compact[nz] = mag;
                compact_pos[nz] = d;
                ++nz;
            }
        }

        long long signed_compact[24];
        long long partial_high = 0;
        long long rank = sign_choice;
        for (int pos = nz - 1; pos >= 0; --pos) {
            long long mag = compact[pos];
            long long needed = (-(partial_high + mag)) & 7LL;
            long long zero_count = even_lower_sign_count(compact, pos, needed);
            bool choose_neg = rank >= zero_count;
            if (choose_neg) rank -= zero_count;
            long long signed_mag = choose_neg ? -mag : mag;
            signed_compact[pos] = signed_mag;
            partial_high += signed_mag;
        }
        for (int pos = 0; pos < nz; ++pos) {
            out[row * 24 + compact_pos[pos]] = signed_compact[pos];
        }
    } else {
        gchoice = cursor;
        int golay_index = (int)golay_indices[golay_start[class_id] + gchoice];
        golay_bits = golay_bits_table[golay_index];
        long long odd_seq[24];
        unrank_multiset(permutation_choice, odd_counts, odd_values, class_id, comb, max_vals, odd_seq);
        for (int d = 0; d < 24; ++d) {
            long long mag = odd_seq[d];
            int bit = (golay_bits >> d) & 1;
            long long target = bit == 1 ? 3 : 1;
            long long sign = ((mag & 3LL) == target) ? 1 : -1;
            out[row * 24 + d] = sign * mag;
        }
    }
}
"""


def _check_cuda(result: tuple) -> tuple:
    err = result[0]
    if int(err) != 0:
        raise RuntimeError(str(err))
    return result


@lru_cache(maxsize=8)
def _load_kernel(major: int, minor: int):
    source = CUDA_SOURCE.encode("utf-8")
    _check_cuda(driver.cuInit(0))
    err, program = nvrtc.nvrtcCreateProgram(source, b"llvq_dequantize.cu", 0, [], [])
    _check_cuda((err,))
    options = [f"--gpu-architecture=compute_{major}{minor}".encode("ascii"), b"--std=c++17"]
    err = nvrtc.nvrtcCompileProgram(program, len(options), options)[0]
    if int(err) != 0:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        log = bytearray(log_size)
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(bytes(log).decode("utf-8", errors="replace"))
    _, ptx_size = nvrtc.nvrtcGetPTXSize(program)
    ptx = bytearray(ptx_size)
    _check_cuda(nvrtc.nvrtcGetPTX(program, ptx))
    _, module = _check_cuda(driver.cuModuleLoadData(bytes(ptx)))
    _, function = _check_cuda(driver.cuModuleGetFunction(module, b"llvq_dequantize_kernel"))
    return module, function


def _ptr_arg(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _value_arg(value, ctype):
    return ctype(value)


def dequantize_lattice_cuda(
    indices: torch.Tensor,
    out: torch.Tensor,
    meta: dict[str, torch.Tensor | int],
    comb: torch.Tensor,
) -> torch.Tensor:
    if indices.device.type != "cuda":
        raise ValueError("indices must be a CUDA tensor")
    torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(indices.device)
    _, function = _load_kernel(major, minor)

    args = [
        _ptr_arg(indices),
        _ptr_arg(out),
        _ptr_arg(meta["class_starts"]),
        _ptr_arg(meta["parity"]),
        _ptr_arg(meta["perm_count"]),
        _ptr_arg(meta["sign_count"]),
        _ptr_arg(meta["f1_perm"]),
        _ptr_arg(meta["golay_start"]),
        _ptr_arg(meta["golay_indices"]),
        _ptr_arg(meta["golay_bits"]),
        _ptr_arg(meta["f0_counts"]),
        _ptr_arg(meta["f1_counts"]),
        _ptr_arg(meta["odd_counts"]),
        _ptr_arg(meta["f0_values"]),
        _ptr_arg(meta["f1_values"]),
        _ptr_arg(meta["odd_values"]),
        _ptr_arg(comb),
        _value_arg(indices.numel(), ctypes.c_longlong),
        _value_arg(int(meta["n_classes"]), ctypes.c_int),
        _value_arg(int(meta["max_vals"]), ctypes.c_int),
    ]
    kernel_params = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in args]
    )
    threads = 128
    blocks = (indices.numel() + threads - 1) // threads
    _check_cuda(driver.cuLaunchKernel(function, blocks, 1, 1, threads, 1, 1, 0, 0, kernel_params, 0))
    return out
