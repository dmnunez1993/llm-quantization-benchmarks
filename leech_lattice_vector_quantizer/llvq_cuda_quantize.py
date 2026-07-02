from __future__ import annotations

import ctypes
from functools import lru_cache

import torch
from cuda.bindings import driver, nvrtc


CUDA_SOURCE = r"""
#define LLVQ_INF_F 3.4028234663852886e38f

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

extern "C" __device__ long long rank_multiset_sequence(
    const long long seq[24],
    int seq_len,
    const long long* counts_ptr,
    const long long* values_ptr,
    int class_id,
    const long long* comb,
    int max_vals
) {
    long long counts[16];
    long long values[16];
    for (int i = 0; i < max_vals; ++i) {
        counts[i] = counts_ptr[class_id * max_vals + i];
        values[i] = values_ptr[class_id * max_vals + i];
    }

    long long rank = 0;
    for (int pos = 0; pos < seq_len; ++pos) {
        long long actual = seq[pos];
        int actual_vi = -1;
        for (int vi = 0; vi < max_vals; ++vi) {
            if (counts[vi] <= 0) continue;
            if (values[vi] == actual) {
                actual_vi = vi;
                break;
            }
            counts[vi] -= 1;
            rank += comb_count(counts, comb, max_vals);
            counts[vi] += 1;
        }
        if (actual_vi >= 0) counts[actual_vi] -= 1;
    }
    return rank;
}

extern "C" __device__ long long rank_even_signs(const long long candidate[24]) {
    long long compact[24];
    int negative[24];
    int nz = 0;
    for (int d = 0; d < 24; ++d) {
        long long value = candidate[d];
        if (value == 0) continue;
        compact[nz] = value < 0 ? -value : value;
        negative[nz] = value < 0 ? 1 : 0;
        ++nz;
    }
    if (nz == 0) return 0;

    long long lower_counts[25][8];
    for (int pos = 0; pos <= 24; ++pos) {
        for (int r = 0; r < 8; ++r) lower_counts[pos][r] = 0;
    }
    lower_counts[0][0] = 1;
    for (int pos = 0; pos < nz; ++pos) {
        long long mag = compact[pos] & 7LL;
        for (int r = 0; r < 8; ++r) {
            long long count = lower_counts[pos][r];
            lower_counts[pos + 1][(r + mag) & 7LL] += count;
            lower_counts[pos + 1][(r - mag) & 7LL] += count;
        }
    }

    long long rank = 0;
    long long partial_high = 0;
    for (int pos = nz - 1; pos >= 0; --pos) {
        long long mag = compact[pos];
        if (negative[pos]) {
            long long needed = (-(partial_high + mag)) & 7LL;
            rank += lower_counts[pos][needed];
            partial_high -= mag;
        } else {
            partial_high += mag;
        }
    }
    return rank;
}

extern "C" __device__ long long final_index_from_pair(
    const float* x,
    long long best_pair,
    const int* golay_bits_table,
    const int* class_parity,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    const long long* class_starts,
    const long long* perm_count,
    const long long* sign_count,
    const long long* f1_perm,
    const long long* golay_start,
    const long long* golay_count,
    const long long* golay_indices,
    const long long* f0_counts,
    const long long* f1_counts,
    const long long* odd_counts,
    const long long* f0_values,
    const long long* f1_values,
    const long long* odd_values,
    const long long* comb,
    int n_golay,
    int max_vals
) {
    int class_id = (int)(best_pair / n_golay);
    int golay_index = (int)(best_pair - (long long)class_id * n_golay);
    int golay_bits = golay_bits_table[golay_index];
    int parity = class_parity[class_id];
    long long candidate[24];
    for (int d = 0; d < 24; ++d) candidate[d] = 0;

    long long gchoice = 0;
    long long gstart = golay_start[class_id];
    long long gcount = golay_count[class_id];
    for (long long i = 0; i < gcount; ++i) {
        if ((int)golay_indices[gstart + i] == golay_index) {
            gchoice = i;
            break;
        }
    }

    if (parity == 0) {
        int selected0 = 0;
        int f0_mask = ((1 << 24) - 1) ^ golay_bits;
        int f0_len = class_f0_len[class_id];
        for (int pick = 0; pick < 24; ++pick) {
            if (pick >= f0_len) break;
            float best_abs = LLVQ_INF_F;
            int best_d = 0;
            int best_bit = 0;
            for (int d = 0; d < 24; ++d) {
                int bit_mask = 1 << d;
                bool allowed = ((f0_mask & bit_mask) != 0) && ((selected0 & bit_mask) == 0);
                float abs_val = fabsf(x[d]);
                if (allowed && abs_val < best_abs) {
                    best_abs = abs_val;
                    best_d = d;
                    best_bit = bit_mask;
                }
            }
            selected0 |= best_bit;
            long long mag = (long long)class_f0_mags[class_id * 24 + pick];
            candidate[best_d] = x[best_d] >= 0.0f ? mag : -mag;
        }

        int selected1 = 0;
        int f1_mask = golay_bits;
        int f1_len = class_f1_len[class_id];
        int negative_parity = 0;
        int first_f1_d = 0;
        for (int pick = 0; pick < 24; ++pick) {
            if (pick >= f1_len) break;
            float best_abs = LLVQ_INF_F;
            int best_d = 0;
            int best_bit = 0;
            int best_neg = 0;
            for (int d = 0; d < 24; ++d) {
                int bit_mask = 1 << d;
                bool allowed = ((f1_mask & bit_mask) != 0) && ((selected1 & bit_mask) == 0);
                float abs_val = fabsf(x[d]);
                if (allowed && abs_val < best_abs) {
                    best_abs = abs_val;
                    best_d = d;
                    best_bit = bit_mask;
                    best_neg = x[d] < 0.0f ? 1 : 0;
                }
            }
            selected1 |= best_bit;
            if (pick == 0) first_f1_d = best_d;
            long long mag = (long long)class_f1_mags[class_id * 24 + pick];
            candidate[best_d] = x[best_d] >= 0.0f ? mag : -mag;
            negative_parity = (negative_parity + best_neg) & 1;
        }
        if (f1_len > 0 && negative_parity != class_required[class_id]) {
            candidate[first_f1_d] = -candidate[first_f1_d];
        }

        long long f0_seq[24];
        long long f1_seq[24];
        int f0_pos = 0;
        int f1_pos = 0;
        for (int d = 0; d < 24; ++d) {
            long long abs_val = candidate[d] < 0 ? -candidate[d] : candidate[d];
            if (((golay_bits >> d) & 1) == 0) f0_seq[f0_pos++] = abs_val;
            else f1_seq[f1_pos++] = abs_val;
        }
        long long f0_rank = rank_multiset_sequence(
            f0_seq,
            f0_pos,
            f0_counts,
            f0_values,
            class_id,
            comb,
            max_vals
        );
        long long f1_rank = rank_multiset_sequence(
            f1_seq,
            f1_pos,
            f1_counts,
            f1_values,
            class_id,
            comb,
            max_vals
        );
        long long permutation_choice = f0_rank * f1_perm[class_id] + f1_rank;
        long long sign_choice = rank_even_signs(candidate);
        long long local = (gchoice * sign_count[class_id] + sign_choice) * perm_count[class_id] + permutation_choice;
        return class_starts[class_id] + local;
    }

    int selected = 0;
    for (int rank = 0; rank < 24; ++rank) {
        float best_val = LLVQ_INF_F;
        int best_d = 0;
        int best_bit = 0;
        for (int d = 0; d < 24; ++d) {
            int bit_mask = 1 << d;
            bool free = (selected & bit_mask) == 0;
            int golay_bit = (golay_bits >> d) & 1;
            float flip = golay_bit == 1 ? -1.0f : 1.0f;
            float x_prime = x[d] * flip;
            if (free && x_prime < best_val) {
                best_val = x_prime;
                best_d = d;
                best_bit = bit_mask;
            }
        }
        selected |= best_bit;
        long long leader_val = (long long)class_odd_leaders[class_id * 24 + rank];
        int golay_bit = (golay_bits >> best_d) & 1;
        long long flip = golay_bit == 1 ? -1LL : 1LL;
        candidate[best_d] = leader_val * flip;
    }

    long long odd_seq[24];
    for (int d = 0; d < 24; ++d) odd_seq[d] = candidate[d] < 0 ? -candidate[d] : candidate[d];
    long long permutation_choice = rank_multiset_sequence(
        odd_seq,
        24,
        odd_counts,
        odd_values,
        class_id,
        comb,
        max_vals
    );
    long long local = gchoice * perm_count[class_id] + permutation_choice;
    return class_starts[class_id] + local;
}

extern "C" __device__ float llvq_score_pair(
    const float* x,
    const int* golay_bits_table,
    const int* class_parity,
    const float* class_shell,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    int class_id,
    int golay_index
) {
    int golay_bits = golay_bits_table[golay_index];
    int parity = class_parity[class_id];
    float shell = class_shell[class_id];
    float projection = 0.0f;

    if (parity == 0) {
        int selected0 = 0;
        int f0_mask = ((1 << 24) - 1) ^ golay_bits;
        int f0_len = class_f0_len[class_id];
        for (int pick = 0; pick < 24; ++pick) {
            if (pick >= f0_len) break;
            float best_abs = LLVQ_INF_F;
            int best_bit = 0;
            for (int d = 0; d < 24; ++d) {
                int bit_mask = 1 << d;
                bool allowed = ((f0_mask & bit_mask) != 0) && ((selected0 & bit_mask) == 0);
                float x_val = x[d];
                float abs_val = fabsf(x_val);
                if (allowed && abs_val < best_abs) {
                    best_abs = abs_val;
                    best_bit = bit_mask;
                }
            }
            selected0 |= best_bit;
            float mag = (float)class_f0_mags[class_id * 24 + pick];
            projection += mag * best_abs;
        }

        int selected1 = 0;
        int f1_mask = golay_bits;
        int f1_len = class_f1_len[class_id];
        int negative_parity = 0;
        float first_loss = 0.0f;
        for (int pick = 0; pick < 24; ++pick) {
            if (pick >= f1_len) break;
            float best_abs = LLVQ_INF_F;
            int best_bit = 0;
            int best_neg = 0;
            for (int d = 0; d < 24; ++d) {
                int bit_mask = 1 << d;
                bool allowed = ((f1_mask & bit_mask) != 0) && ((selected1 & bit_mask) == 0);
                float x_val = x[d];
                float abs_val = fabsf(x_val);
                if (allowed && abs_val < best_abs) {
                    best_abs = abs_val;
                    best_bit = bit_mask;
                    best_neg = x_val < 0.0f ? 1 : 0;
                }
            }
            selected1 |= best_bit;
            float mag = (float)class_f1_mags[class_id * 24 + pick];
            projection += mag * best_abs;
            negative_parity = (negative_parity + best_neg) & 1;
            if (pick == 0) first_loss = 2.0f * mag * best_abs;
        }

        int required = class_required[class_id];
        if (f1_len > 0 && negative_parity != required) {
            projection -= first_loss;
        }
    } else {
        int selected = 0;
        for (int rank = 0; rank < 24; ++rank) {
            float best_val = LLVQ_INF_F;
            int best_bit = 0;
            for (int d = 0; d < 24; ++d) {
                int bit_mask = 1 << d;
                bool free = (selected & bit_mask) == 0;
                int golay_bit = (golay_bits >> d) & 1;
                float flip = golay_bit == 1 ? -1.0f : 1.0f;
                float x_prime = x[d] * flip;
                if (free && x_prime < best_val) {
                    best_val = x_prime;
                    best_bit = bit_mask;
                }
            }
            selected |= best_bit;
            float leader_val = (float)class_odd_leaders[class_id * 24 + rank];
            projection += best_val * leader_val;
        }
    }

    return 0.35355339059327373f * projection - shell;
}

extern "C" __global__ void llvq_quantize_kernel(
    const float* x,
    long long* best_pair,
    float* best_score,
    long long* out_indices,
    const int* golay_bits_table,
    const int* class_parity,
    const float* class_shell,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    const long long* class_starts,
    const long long* perm_count,
    const long long* sign_count,
    const long long* f1_perm,
    const long long* golay_start,
    const long long* golay_count,
    const long long* golay_indices,
    const long long* f0_counts,
    const long long* f1_counts,
    const long long* odd_counts,
    const long long* f0_values,
    const long long* f1_values,
    const long long* odd_values,
    const long long* comb,
    const int* valid_pair_class,
    const int* valid_pair_golay,
    const int* valid_pair_order,
    long long n,
    int n_classes,
    int n_golay,
    int total_valid_pairs,
    int max_vals
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    float thread_best_score = -LLVQ_INF_F;
    long long thread_best_pair = 0;
    int thread_best_order = 2147483647;
    const float* row_x = x + (long long)row * 24;

    for (int slot = tid; slot < total_valid_pairs; slot += blockDim.x) {
        int class_id = valid_pair_class[slot];
        int golay_index = valid_pair_golay[slot];
        float score = llvq_score_pair(
            row_x,
            golay_bits_table,
            class_parity,
            class_shell,
            class_f0_mags,
            class_f1_mags,
            class_f0_len,
            class_f1_len,
            class_required,
            class_odd_leaders,
            class_id,
            golay_index
        );
        int order = valid_pair_order[slot];
        if (score > thread_best_score || (score == thread_best_score && order < thread_best_order)) {
            thread_best_score = score;
            thread_best_pair = (long long)class_id * n_golay + golay_index;
            thread_best_order = order;
        }
    }

    __shared__ float scores[256];
    __shared__ long long pairs[256];
    __shared__ int orders[256];
    scores[tid] = thread_best_score;
    pairs[tid] = thread_best_pair;
    orders[tid] = thread_best_order;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_score = scores[tid + stride];
            int other_order = orders[tid + stride];
            if (other_score > scores[tid] || (other_score == scores[tid] && other_order < orders[tid])) {
                scores[tid] = other_score;
                pairs[tid] = pairs[tid + stride];
                orders[tid] = other_order;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        best_score[row] = scores[0];
        best_pair[row] = pairs[0];
        out_indices[row] = final_index_from_pair(
            row_x,
            pairs[0],
            golay_bits_table,
            class_parity,
            class_f0_mags,
            class_f1_mags,
            class_f0_len,
            class_f1_len,
            class_required,
            class_odd_leaders,
            class_starts,
            perm_count,
            sign_count,
            f1_perm,
            golay_start,
            golay_count,
            golay_indices,
            f0_counts,
            f1_counts,
            odd_counts,
            f0_values,
            f1_values,
            odd_values,
            comb,
            n_golay,
            max_vals
        );
    }
}

extern "C" __global__ void llvq_quantize_tiles_kernel(
    const float* x,
    long long* tile_best_pair,
    float* tile_best_score,
    int* tile_best_order,
    const int* golay_bits_table,
    const int* class_parity,
    const float* class_shell,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    const int* valid_pair_class,
    const int* valid_pair_golay,
    const int* valid_pair_order,
    const int* valid_pair_shell,
    const float* lower_bound_score,
    long long n,
    int n_golay,
    int total_valid_pairs,
    int tile_size,
    int n_tiles,
    int min_shell,
    int n_shells,
    int prune_by_shell
) {
    int row = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= n || tile >= n_tiles) return;

    int start = tile * tile_size;
    int stop = start + tile_size;
    if (stop > total_valid_pairs) stop = total_valid_pairs;

    float thread_best_score = -LLVQ_INF_F;
    long long thread_best_pair = 0;
    int thread_best_order = 2147483647;
    const float* row_x = x + (long long)row * 24;
    bool active_shells[64];
    for (int i = 0; i < 64; ++i) active_shells[i] = true;

    if (prune_by_shell != 0) {
        float norm2 = 0.0f;
        for (int d = 0; d < 24; ++d) {
            float value = row_x[d];
            norm2 += value * value;
        }
        float norm = sqrtf(norm2);
        float best = lower_bound_score[row];
        for (int si = 0; si < 64; ++si) {
            if (si >= n_shells) {
                active_shells[si] = false;
            } else {
                int shell = min_shell + si;
                float upper = norm * sqrtf(2.0f * (float)shell) - (float)shell;
                active_shells[si] = upper >= best;
            }
        }
    }

    for (int slot = start + tid; slot < stop; slot += blockDim.x) {
        if (prune_by_shell != 0) {
            int shell_idx = valid_pair_shell[slot] - min_shell;
            if (shell_idx < 0 || shell_idx >= 64 || !active_shells[shell_idx]) continue;
        }
        int class_id = valid_pair_class[slot];
        int golay_index = valid_pair_golay[slot];
        float score = llvq_score_pair(
            row_x,
            golay_bits_table,
            class_parity,
            class_shell,
            class_f0_mags,
            class_f1_mags,
            class_f0_len,
            class_f1_len,
            class_required,
            class_odd_leaders,
            class_id,
            golay_index
        );
        int order = valid_pair_order[slot];
        if (score > thread_best_score || (score == thread_best_score && order < thread_best_order)) {
            thread_best_score = score;
            thread_best_pair = (long long)class_id * n_golay + golay_index;
            thread_best_order = order;
        }
    }

    __shared__ float scores[256];
    __shared__ long long pairs[256];
    __shared__ int orders[256];
    scores[tid] = thread_best_score;
    pairs[tid] = thread_best_pair;
    orders[tid] = thread_best_order;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_score = scores[tid + stride];
            int other_order = orders[tid + stride];
            if (other_score > scores[tid] || (other_score == scores[tid] && other_order < orders[tid])) {
                scores[tid] = other_score;
                pairs[tid] = pairs[tid + stride];
                orders[tid] = other_order;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        long long out_idx = (long long)row * n_tiles + tile;
        tile_best_score[out_idx] = scores[0];
        tile_best_pair[out_idx] = pairs[0];
        tile_best_order[out_idx] = orders[0];
    }
}

extern "C" __global__ void llvq_quantize_initial_shell_tiles_kernel(
    const float* x,
    long long* tile_best_pair,
    float* tile_best_score,
    int* tile_best_order,
    const int* golay_bits_table,
    const int* class_parity,
    const float* class_shell,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    const int* valid_pair_class,
    const int* valid_pair_golay,
    const int* valid_pair_order,
    const int* shell_slot_start,
    const int* shell_slot_count,
    long long n,
    int n_golay,
    int tile_size,
    int max_shell_tiles,
    int min_shell,
    int n_shells
) {
    int row = blockIdx.x;
    int tile = blockIdx.y;
    int tid = threadIdx.x;
    if (row >= n || tile >= max_shell_tiles) return;

    const float* row_x = x + (long long)row * 24;
    float norm2 = 0.0f;
    for (int d = 0; d < 24; ++d) {
        float value = row_x[d];
        norm2 += value * value;
    }
    float norm = sqrtf(norm2);
    float best_upper = -LLVQ_INF_F;
    int target_shell_idx = 0;
    for (int si = 0; si < n_shells; ++si) {
        int shell = min_shell + si;
        float upper = norm * sqrtf(2.0f * (float)shell) - (float)shell;
        if (upper > best_upper) {
            best_upper = upper;
            target_shell_idx = si;
        }
    }

    int shell_start = shell_slot_start[target_shell_idx];
    int shell_count = shell_slot_count[target_shell_idx];
    int start = shell_start + tile * tile_size;
    int stop = start + tile_size;
    int shell_stop = shell_start + shell_count;
    if (stop > shell_stop) stop = shell_stop;

    float thread_best_score = -LLVQ_INF_F;
    long long thread_best_pair = 0;
    int thread_best_order = 2147483647;

    for (int slot = start + tid; slot < stop; slot += blockDim.x) {
        int class_id = valid_pair_class[slot];
        int golay_index = valid_pair_golay[slot];
        float score = llvq_score_pair(
            row_x,
            golay_bits_table,
            class_parity,
            class_shell,
            class_f0_mags,
            class_f1_mags,
            class_f0_len,
            class_f1_len,
            class_required,
            class_odd_leaders,
            class_id,
            golay_index
        );
        int order = valid_pair_order[slot];
        if (score > thread_best_score || (score == thread_best_score && order < thread_best_order)) {
            thread_best_score = score;
            thread_best_pair = (long long)class_id * n_golay + golay_index;
            thread_best_order = order;
        }
    }

    __shared__ float scores[256];
    __shared__ long long pairs[256];
    __shared__ int orders[256];
    scores[tid] = thread_best_score;
    pairs[tid] = thread_best_pair;
    orders[tid] = thread_best_order;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_score = scores[tid + stride];
            int other_order = orders[tid + stride];
            if (other_score > scores[tid] || (other_score == scores[tid] && other_order < orders[tid])) {
                scores[tid] = other_score;
                pairs[tid] = pairs[tid + stride];
                orders[tid] = other_order;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        long long out_idx = (long long)row * max_shell_tiles + tile;
        tile_best_score[out_idx] = scores[0];
        tile_best_pair[out_idx] = pairs[0];
        tile_best_order[out_idx] = orders[0];
    }
}

extern "C" __global__ void llvq_quantize_reduce_tiles_kernel(
    const float* x,
    const long long* tile_best_pair,
    const float* tile_best_score,
    const int* tile_best_order,
    long long* best_pair,
    float* best_score,
    long long* out_indices,
    const int* golay_bits_table,
    const int* class_parity,
    const int* class_f0_mags,
    const int* class_f1_mags,
    const int* class_f0_len,
    const int* class_f1_len,
    const int* class_required,
    const int* class_odd_leaders,
    const long long* class_starts,
    const long long* perm_count,
    const long long* sign_count,
    const long long* f1_perm,
    const long long* golay_start,
    const long long* golay_count,
    const long long* golay_indices,
    const long long* f0_counts,
    const long long* f1_counts,
    const long long* odd_counts,
    const long long* f0_values,
    const long long* f1_values,
    const long long* odd_values,
    const long long* comb,
    long long n,
    int n_golay,
    int n_tiles,
    int max_vals
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= n) return;

    float thread_best_score = -LLVQ_INF_F;
    long long thread_best_pair = 0;
    int thread_best_order = 2147483647;

    for (int tile = tid; tile < n_tiles; tile += blockDim.x) {
        long long idx = (long long)row * n_tiles + tile;
        float score = tile_best_score[idx];
        int order = tile_best_order[idx];
        if (score > thread_best_score || (score == thread_best_score && order < thread_best_order)) {
            thread_best_score = score;
            thread_best_pair = tile_best_pair[idx];
            thread_best_order = order;
        }
    }

    __shared__ float scores[256];
    __shared__ long long pairs[256];
    __shared__ int orders[256];
    scores[tid] = thread_best_score;
    pairs[tid] = thread_best_pair;
    orders[tid] = thread_best_order;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_score = scores[tid + stride];
            int other_order = orders[tid + stride];
            if (other_score > scores[tid] || (other_score == scores[tid] && other_order < orders[tid])) {
                scores[tid] = other_score;
                pairs[tid] = pairs[tid + stride];
                orders[tid] = other_order;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        const float* row_x = x + (long long)row * 24;
        best_score[row] = scores[0];
        best_pair[row] = pairs[0];
        out_indices[row] = final_index_from_pair(
            row_x,
            pairs[0],
            golay_bits_table,
            class_parity,
            class_f0_mags,
            class_f1_mags,
            class_f0_len,
            class_f1_len,
            class_required,
            class_odd_leaders,
            class_starts,
            perm_count,
            sign_count,
            f1_perm,
            golay_start,
            golay_count,
            golay_indices,
            f0_counts,
            f1_counts,
            odd_counts,
            f0_values,
            f1_values,
            odd_values,
            comb,
            n_golay,
            max_vals
        );
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
    err, program = nvrtc.nvrtcCreateProgram(source, b"llvq_quantize.cu", 0, [], [])
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
    _, function = _check_cuda(driver.cuModuleGetFunction(module, b"llvq_quantize_kernel"))
    _, tile_function = _check_cuda(driver.cuModuleGetFunction(module, b"llvq_quantize_tiles_kernel"))
    _, initial_shell_function = _check_cuda(
        driver.cuModuleGetFunction(module, b"llvq_quantize_initial_shell_tiles_kernel")
    )
    _, reduce_function = _check_cuda(
        driver.cuModuleGetFunction(module, b"llvq_quantize_reduce_tiles_kernel")
    )
    return module, function, tile_function, initial_shell_function, reduce_function


def _ptr_arg(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def _value_arg(value, ctype):
    return ctype(value)


def quantize_lattice_cuda(
    x: torch.Tensor,
    indices: torch.Tensor,
    best_pair: torch.Tensor,
    best_score: torch.Tensor,
    meta: dict[str, torch.Tensor | int],
    rank_meta: dict[str, torch.Tensor | int],
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.device.type != "cuda":
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("x must be float32")
    torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(x.device)
    _, function, tile_function, initial_shell_function, reduce_function = _load_kernel(major, minor)

    threads = 256
    tile_size = int(meta.get("tile_size", 4096))
    tile_batch_threshold = int(meta.get("tile_batch_threshold", 4096))
    prune_by_shell = bool(meta.get("prune_by_shell", True))
    total_valid_pairs = int(meta["total_valid_pairs"])
    n_tiles = (total_valid_pairs + tile_size - 1) // tile_size
    if n_tiles > 1 and x.shape[0] <= tile_batch_threshold:
        lower_bound_score = best_score
        if prune_by_shell:
            max_shell_tiles = int(meta["max_shell_tiles"])
            init_tile_count = x.shape[0] * max_shell_tiles
            init_tile_best_pair = torch.empty((init_tile_count,), dtype=torch.long, device=x.device)
            init_tile_best_score = torch.empty((init_tile_count,), dtype=torch.float32, device=x.device)
            init_tile_best_order = torch.empty((init_tile_count,), dtype=torch.int32, device=x.device)
            init_indices = torch.empty((x.shape[0],), dtype=torch.long, device=x.device)
            init_pair = torch.empty((x.shape[0],), dtype=torch.long, device=x.device)
            init_score = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)

            init_args = [
                _ptr_arg(x),
                _ptr_arg(init_tile_best_pair),
                _ptr_arg(init_tile_best_score),
                _ptr_arg(init_tile_best_order),
                _ptr_arg(meta["golay_bits"]),
                _ptr_arg(meta["class_parity"]),
                _ptr_arg(meta["class_shell"]),
                _ptr_arg(meta["class_f0_mags"]),
                _ptr_arg(meta["class_f1_mags"]),
                _ptr_arg(meta["class_f0_len"]),
                _ptr_arg(meta["class_f1_len"]),
                _ptr_arg(meta["class_required"]),
                _ptr_arg(meta["class_odd_leaders"]),
                _ptr_arg(meta["valid_pair_class"]),
                _ptr_arg(meta["valid_pair_golay"]),
                _ptr_arg(meta["valid_pair_order"]),
                _ptr_arg(meta["shell_slot_start"]),
                _ptr_arg(meta["shell_slot_count"]),
                _value_arg(x.shape[0], ctypes.c_longlong),
                _value_arg(int(meta["n_golay"]), ctypes.c_int),
                _value_arg(tile_size, ctypes.c_int),
                _value_arg(max_shell_tiles, ctypes.c_int),
                _value_arg(int(meta["min_shell"]), ctypes.c_int),
                _value_arg(int(meta["n_shells"]), ctypes.c_int),
            ]
            init_kernel_params = (ctypes.c_void_p * len(init_args))(
                *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in init_args]
            )
            _check_cuda(
                driver.cuLaunchKernel(
                    initial_shell_function,
                    x.shape[0],
                    max_shell_tiles,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    0,
                    init_kernel_params,
                    0,
                )
            )

            init_reduce_args = [
                _ptr_arg(x),
                _ptr_arg(init_tile_best_pair),
                _ptr_arg(init_tile_best_score),
                _ptr_arg(init_tile_best_order),
                _ptr_arg(init_pair),
                _ptr_arg(init_score),
                _ptr_arg(init_indices),
                _ptr_arg(meta["golay_bits"]),
                _ptr_arg(meta["class_parity"]),
                _ptr_arg(meta["class_f0_mags"]),
                _ptr_arg(meta["class_f1_mags"]),
                _ptr_arg(meta["class_f0_len"]),
                _ptr_arg(meta["class_f1_len"]),
                _ptr_arg(meta["class_required"]),
                _ptr_arg(meta["class_odd_leaders"]),
                _ptr_arg(rank_meta["class_starts"]),
                _ptr_arg(rank_meta["perm_count"]),
                _ptr_arg(rank_meta["sign_count"]),
                _ptr_arg(rank_meta["f1_perm"]),
                _ptr_arg(rank_meta["golay_start"]),
                _ptr_arg(rank_meta["golay_count"]),
                _ptr_arg(rank_meta["golay_indices"]),
                _ptr_arg(rank_meta["f0_counts"]),
                _ptr_arg(rank_meta["f1_counts"]),
                _ptr_arg(rank_meta["odd_counts"]),
                _ptr_arg(rank_meta["f0_values"]),
                _ptr_arg(rank_meta["f1_values"]),
                _ptr_arg(rank_meta["odd_values"]),
                _ptr_arg(comb),
                _value_arg(x.shape[0], ctypes.c_longlong),
                _value_arg(int(meta["n_golay"]), ctypes.c_int),
                _value_arg(max_shell_tiles, ctypes.c_int),
                _value_arg(int(rank_meta["max_vals"]), ctypes.c_int),
            ]
            init_reduce_kernel_params = (ctypes.c_void_p * len(init_reduce_args))(
                *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in init_reduce_args]
            )
            _check_cuda(
                driver.cuLaunchKernel(
                    reduce_function,
                    x.shape[0],
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    0,
                    init_reduce_kernel_params,
                    0,
                )
            )
            lower_bound_score = init_score

        tile_count = x.shape[0] * n_tiles
        tile_best_pair = torch.empty((tile_count,), dtype=torch.long, device=x.device)
        tile_best_score = torch.empty((tile_count,), dtype=torch.float32, device=x.device)
        tile_best_order = torch.empty((tile_count,), dtype=torch.int32, device=x.device)

        tile_args = [
            _ptr_arg(x),
            _ptr_arg(tile_best_pair),
            _ptr_arg(tile_best_score),
            _ptr_arg(tile_best_order),
            _ptr_arg(meta["golay_bits"]),
            _ptr_arg(meta["class_parity"]),
            _ptr_arg(meta["class_shell"]),
            _ptr_arg(meta["class_f0_mags"]),
            _ptr_arg(meta["class_f1_mags"]),
            _ptr_arg(meta["class_f0_len"]),
            _ptr_arg(meta["class_f1_len"]),
            _ptr_arg(meta["class_required"]),
            _ptr_arg(meta["class_odd_leaders"]),
            _ptr_arg(meta["valid_pair_class"]),
            _ptr_arg(meta["valid_pair_golay"]),
            _ptr_arg(meta["valid_pair_order"]),
            _ptr_arg(meta["valid_pair_shell"]),
            _ptr_arg(lower_bound_score),
            _value_arg(x.shape[0], ctypes.c_longlong),
            _value_arg(int(meta["n_golay"]), ctypes.c_int),
            _value_arg(total_valid_pairs, ctypes.c_int),
            _value_arg(tile_size, ctypes.c_int),
            _value_arg(n_tiles, ctypes.c_int),
            _value_arg(int(meta["min_shell"]), ctypes.c_int),
            _value_arg(int(meta["n_shells"]), ctypes.c_int),
            _value_arg(1 if prune_by_shell else 0, ctypes.c_int),
        ]
        tile_kernel_params = (ctypes.c_void_p * len(tile_args))(
            *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in tile_args]
        )
        _check_cuda(
            driver.cuLaunchKernel(
                tile_function,
                x.shape[0],
                n_tiles,
                1,
                threads,
                1,
                1,
                0,
                0,
                tile_kernel_params,
                0,
            )
        )

        reduce_args = [
            _ptr_arg(x),
            _ptr_arg(tile_best_pair),
            _ptr_arg(tile_best_score),
            _ptr_arg(tile_best_order),
            _ptr_arg(best_pair),
            _ptr_arg(best_score),
            _ptr_arg(indices),
            _ptr_arg(meta["golay_bits"]),
            _ptr_arg(meta["class_parity"]),
            _ptr_arg(meta["class_f0_mags"]),
            _ptr_arg(meta["class_f1_mags"]),
            _ptr_arg(meta["class_f0_len"]),
            _ptr_arg(meta["class_f1_len"]),
            _ptr_arg(meta["class_required"]),
            _ptr_arg(meta["class_odd_leaders"]),
            _ptr_arg(rank_meta["class_starts"]),
            _ptr_arg(rank_meta["perm_count"]),
            _ptr_arg(rank_meta["sign_count"]),
            _ptr_arg(rank_meta["f1_perm"]),
            _ptr_arg(rank_meta["golay_start"]),
            _ptr_arg(rank_meta["golay_count"]),
            _ptr_arg(rank_meta["golay_indices"]),
            _ptr_arg(rank_meta["f0_counts"]),
            _ptr_arg(rank_meta["f1_counts"]),
            _ptr_arg(rank_meta["odd_counts"]),
            _ptr_arg(rank_meta["f0_values"]),
            _ptr_arg(rank_meta["f1_values"]),
            _ptr_arg(rank_meta["odd_values"]),
            _ptr_arg(comb),
            _value_arg(x.shape[0], ctypes.c_longlong),
            _value_arg(int(meta["n_golay"]), ctypes.c_int),
            _value_arg(n_tiles, ctypes.c_int),
            _value_arg(int(rank_meta["max_vals"]), ctypes.c_int),
        ]
        reduce_kernel_params = (ctypes.c_void_p * len(reduce_args))(
            *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in reduce_args]
        )
        _check_cuda(
            driver.cuLaunchKernel(
                reduce_function,
                x.shape[0],
                1,
                1,
                threads,
                1,
                1,
                0,
                0,
                reduce_kernel_params,
                0,
            )
        )
        return indices, best_pair, best_score

    args = [
        _ptr_arg(x),
        _ptr_arg(best_pair),
        _ptr_arg(best_score),
        _ptr_arg(indices),
        _ptr_arg(meta["golay_bits"]),
        _ptr_arg(meta["class_parity"]),
        _ptr_arg(meta["class_shell"]),
        _ptr_arg(meta["class_f0_mags"]),
        _ptr_arg(meta["class_f1_mags"]),
        _ptr_arg(meta["class_f0_len"]),
        _ptr_arg(meta["class_f1_len"]),
        _ptr_arg(meta["class_required"]),
        _ptr_arg(meta["class_odd_leaders"]),
        _ptr_arg(rank_meta["class_starts"]),
        _ptr_arg(rank_meta["perm_count"]),
        _ptr_arg(rank_meta["sign_count"]),
        _ptr_arg(rank_meta["f1_perm"]),
        _ptr_arg(rank_meta["golay_start"]),
        _ptr_arg(rank_meta["golay_count"]),
        _ptr_arg(rank_meta["golay_indices"]),
        _ptr_arg(rank_meta["f0_counts"]),
        _ptr_arg(rank_meta["f1_counts"]),
        _ptr_arg(rank_meta["odd_counts"]),
        _ptr_arg(rank_meta["f0_values"]),
        _ptr_arg(rank_meta["f1_values"]),
        _ptr_arg(rank_meta["odd_values"]),
        _ptr_arg(comb),
        _ptr_arg(meta["valid_pair_class"]),
        _ptr_arg(meta["valid_pair_golay"]),
        _ptr_arg(meta["valid_pair_order"]),
        _value_arg(x.shape[0], ctypes.c_longlong),
        _value_arg(int(meta["n_classes"]), ctypes.c_int),
        _value_arg(int(meta["n_golay"]), ctypes.c_int),
        _value_arg(int(meta["total_valid_pairs"]), ctypes.c_int),
        _value_arg(int(rank_meta["max_vals"]), ctypes.c_int),
    ]
    kernel_params = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in args]
    )
    _check_cuda(driver.cuLaunchKernel(function, x.shape[0], 1, 1, threads, 1, 1, 0, 0, kernel_params, 0))
    return indices, best_pair, best_score
