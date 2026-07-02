from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, comb, factorial, log2, sqrt
from pathlib import Path
import pickle
from random import Random
import sys
from time import perf_counter
from typing import Iterable, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from .llvq_cuda_dequantize import dequantize_lattice_cuda
except Exception:  # pragma: no cover - optional CUDA driver/NVRTC path
    dequantize_lattice_cuda = None

try:
    from .llvq_cuda_quantize import quantize_lattice_cuda
except Exception:  # pragma: no cover - optional CUDA driver/NVRTC path
    quantize_lattice_cuda = None


DIM = 24
LATTICE_SCALE = 1.0 / sqrt(8.0)
CLASS_LEADER_CACHE_VERSION = 1




@dataclass(frozen=True)
class ClassLeader:
    """Absolute-value leader for one Leech spherical-code class."""

    shell: int
    parity: str
    count: int
    mult_abs8: int
    mult_abs6: int
    mult_abs5: int
    mult_abs4: int
    mult_abs3: int
    mult_abs2: int
    mult_abs1: int
    mult_abs0: int
    extra_multiplicities: tuple[tuple[int, int], ...] = ()

    @property
    def multiplicities(self) -> dict[int, int]:
        out = {
            8: self.mult_abs8,
            6: self.mult_abs6,
            5: self.mult_abs5,
            4: self.mult_abs4,
            3: self.mult_abs3,
            2: self.mult_abs2,
            1: self.mult_abs1,
            0: self.mult_abs0,
        }
        for magnitude, count in self.extra_multiplicities:
            out[magnitude] = count
        return {magnitude: count for magnitude, count in out.items() if count}

    @property
    def leader(self) -> tuple[int, ...]:
        values: list[int] = []
        for magnitude, count in sorted(self.multiplicities.items(), reverse=True):
            values.extend([magnitude] * count)
        if len(values) != DIM:
            raise ValueError(f"leader has {len(values)} coordinates, expected {DIM}")
        return tuple(values)

    @property
    def even_weight(self) -> int:
        """Number of even leader coordinates congruent to 2 mod 4."""
        return sum(count for magnitude, count in self.multiplicities.items() if magnitude % 4 == 2)

    @property
    def nonzero_count(self) -> int:
        return DIM - self.mult_abs0


@dataclass(frozen=True)
class RankedCodeword:
    global_index: int
    shell: int
    shell_local_index: int
    class_index: int
    class_local_index: int
    class_offset: int
    leader: ClassLeader


@dataclass(frozen=True)
class LocalDecomposition:
    golay_choice: int
    golay_codeword: tuple[int, ...]
    sign_choice: int
    permutation_choice: int


@dataclass(frozen=True)
class ClassLocalStructure:
    shell: int
    class_index: int
    leader: ClassLeader
    golay_codeword_indices: tuple[int, ...]
    sign_count: int
    permutation_count: int

    @property
    def count(self) -> int:
        return len(self.golay_codeword_indices) * self.sign_count * self.permutation_count


# Reference leaders/counts from the Adoul-Barth small-shell leader table for m=2..4.
# These are not used to build the database; generation is done from the shell
# equation and class-construction rules, then compared against this table.
# Multiplicity columns are |8|, |6|, |5|, |4|, |3|, |2|, |1|, |0|.
TABLE_CLASS_LEADERS_REFERENCE: tuple[ClassLeader, ...] = (
    ClassLeader(2, "even", 1104, 0, 0, 0, 2, 0, 0, 0, 22),
    ClassLeader(2, "even", 97152, 0, 0, 0, 0, 0, 8, 0, 16),
    ClassLeader(2, "odd", 98304, 0, 0, 0, 0, 1, 0, 23, 0),
    ClassLeader(3, "even", 3108864, 0, 0, 0, 1, 0, 8, 0, 15),
    ClassLeader(3, "even", 5275648, 0, 0, 0, 0, 0, 12, 0, 12),
    ClassLeader(3, "odd", 98304, 0, 0, 1, 0, 0, 0, 23, 0),
    ClassLeader(3, "odd", 8290304, 0, 0, 0, 0, 3, 0, 21, 0),
    ClassLeader(4, "even", 170016, 0, 0, 0, 4, 0, 0, 0, 20),
    ClassLeader(4, "even", 48, 1, 0, 0, 0, 0, 0, 0, 23),
    ClassLeader(4, "even", 46632960, 0, 0, 0, 2, 0, 8, 0, 14),
    ClassLeader(4, "even", 777216, 0, 1, 0, 0, 0, 7, 0, 16),
    ClassLeader(4, "even", 126615552, 0, 0, 0, 1, 0, 12, 0, 11),
    ClassLeader(4, "even", 24870912, 0, 0, 0, 0, 0, 16, 0, 8),
    ClassLeader(4, "odd", 24870912, 0, 0, 1, 0, 2, 0, 21, 0),
    ClassLeader(4, "odd", 174096384, 0, 0, 0, 0, 5, 0, 19, 0),
)


class LeechLatticeVectorQuantizerGpu:
    """
    Implicit shell -> class -> local-symmetry database for Leech codewords.

    The database stores shell offsets, class offsets, class leaders, compatible
    Golay words, sign counts, and permutation-coset counts. It does not expand
    all codewords into memory.
    """

    def __init__(
        self,
        max_shell: int = 4,
        leaders: Sequence[ClassLeader] | None = None,
        cache_dir: str | Path | None = ".llvq_cache",
        use_cache: bool = True,
        verbose: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if max_shell < 2:
            raise ValueError("max_shell must be at least 2")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self._binom = torch.tensor(
            [[comb(n, k) if k <= n else 0 for k in range(DIM + 1)] for n in range(DIM + 1)],
            dtype=torch.long,
            device=self.device,
        )
        if leaders is None:
            leaders = generate_class_leaders(
                max_shell,
                cache_dir=cache_dir,
                use_cache=use_cache,
                verbose=verbose,
            )
        available_shells = {row.shell for row in leaders}
        missing = [m for m in range(2, max_shell + 1) if m not in available_shells]
        if missing:
            raise ValueError(f"no class leaders are available for shells: {missing}")

        self.max_shell = max_shell
        self.golay_codewords = extended_golay_codewords()
        self.golay_index_by_codeword = {
            codeword: index for index, codeword in enumerate(self.golay_codewords)
        }
        self.classes_by_shell = {
            shell: tuple(
                sorted(
                    (row for row in leaders if row.shell == shell),
                    key=lambda row: (row.leader, 0 if row.parity == "even" else 1, row.count),
                )
            )
            for shell in range(2, max_shell + 1)
        }
        self.shell_offsets = self._build_shell_offsets()
        self.class_offsets = self._build_class_offsets()
        self.local_structures = self._build_local_structures()
        self.class_index_by_shell_and_key = self._build_class_index()
        self.shell_counts = {
            shell: sum(row.count for row in rows) for shell, rows in self.classes_by_shell.items()
        }
        self.cumulative_shell_counts = {
            shell: self.shell_offsets[shell] + self.shell_counts[shell]
            for shell in range(2, max_shell + 1)
        }
        self.total_count = self.cumulative_shell_counts[max_shell]
        self.shape_bits = ceil(log2(self.total_count)) if self.total_count > 1 else 1
        self._validate()

    def rank_class(self, shell: int, class_index: int, class_local_index: int = 0) -> int:
        classes = self.classes_by_shell[shell]
        if class_index < 0 or class_index >= len(classes):
            raise IndexError(class_index)
        leader = classes[class_index]
        if class_local_index < 0 or class_local_index >= leader.count:
            raise IndexError(class_local_index)
        return self.shell_offsets[shell] + self.class_offsets[shell][class_index] + class_local_index

    def unrank(self, global_index: int) -> RankedCodeword:
        if global_index < 0 or global_index >= self.total_count:
            raise IndexError(global_index)

        shell = self.shell_for_index(global_index)
        shell_local = global_index - self.shell_offsets[shell]
        offsets = self.class_offsets[shell]
        class_index = bisect_right(offsets, shell_local) - 1
        class_offset = offsets[class_index]
        return RankedCodeword(
            global_index=global_index,
            shell=shell,
            shell_local_index=shell_local,
            class_index=class_index,
            class_local_index=shell_local - class_offset,
            class_offset=class_offset,
            leader=self.classes_by_shell[shell][class_index],
        )

    def shell_for_index(self, global_index: int) -> int:
        ends = [self.cumulative_shell_counts[shell] for shell in range(2, self.max_shell + 1)]
        return 2 + bisect_right(ends, global_index)

    def decompose_local_index(self, ranked: RankedCodeword) -> LocalDecomposition:
        structure = self.local_structures[ranked.shell][ranked.class_index]
        cursor = ranked.class_local_index
        permutation_choice = cursor % structure.permutation_count
        cursor //= structure.permutation_count
        sign_choice = cursor % structure.sign_count
        cursor //= structure.sign_count
        golay_choice = cursor
        golay_index = structure.golay_codeword_indices[golay_choice]
        return LocalDecomposition(
            golay_choice=golay_choice,
            golay_codeword=self.golay_codewords[golay_index],
            sign_choice=sign_choice,
            permutation_choice=permutation_choice,
        )

    def codeword(self, global_index: int) -> tuple[int, ...]:
        ranked = self.unrank(global_index)
        local = self.decompose_local_index(ranked)
        return unrank_class_codeword(ranked.leader, local)

    def dequantize_lattice(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
        *,
        check_bounds: bool = True,
    ) -> tuple[int, ...] | torch.Tensor:
        """Return integer Leech representative(s) before the 1/sqrt(8) scale."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return self.codeword(global_index)

        indices = torch.as_tensor(global_index, dtype=torch.long, device=self.device)
        output_shape = indices.shape
        codewords = self._dequantize_lattice_cuda(indices.reshape(-1), check_bounds=check_bounds)
        return codewords.reshape(*output_shape, DIM)

    def dequantize_lattice_cuda(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
        *,
        check_bounds: bool = True,
    ) -> tuple[int, ...] | torch.Tensor:
        """Return integer Leech representative(s) with the NVRTC CUDA kernel."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return self.codeword(global_index)

        indices = torch.as_tensor(global_index, dtype=torch.long, device=self.device)
        output_shape = indices.shape
        codewords = self._dequantize_lattice_cuda(indices.reshape(-1), check_bounds=check_bounds)
        return codewords.reshape(*output_shape, DIM)

    def dequantize(
        self,
        global_index: int | Sequence[int] | torch.Tensor,
        *,
        check_bounds: bool = True,
    ) -> tuple[float, ...] | torch.Tensor:
        """Return scaled Leech lattice vector(s) in Lambda_24."""
        if not torch.is_tensor(global_index) and isinstance(global_index, int):
            return tuple(LATTICE_SCALE * value for value in self.dequantize_lattice(global_index))

        codewords = self.dequantize_lattice(global_index, check_bounds=check_bounds)
        if not torch.is_tensor(codewords):
            return tuple(LATTICE_SCALE * value for value in codewords)
        return LATTICE_SCALE * codewords.to(self.dtype)

    def quantize(self, vector: Sequence[float] | torch.Tensor) -> int | torch.Tensor:
        """
        Quantize real-valued 24-D vector(s) with the fused NVRTC CUDA kernel.

        Tensor inputs may have shape `(..., 24)` and are searched/ranked on
        `self.device`. Sequence inputs preserve the CPU API and return `int`.
        """
        return self.quantize_cuda(vector)

    def quantize_cuda(self, vector: Sequence[float] | torch.Tensor) -> int | torch.Tensor:
        """
        Quantize with the fused NVRTC CUDA kernel.

        CUDA computes the winning compact `(class, Golay)` pair, reconstructs
        the winning codeword, ranks it, and returns final database indices.
        """
        if self.device.type != "cuda":
            raise ValueError("quantize_cuda requires a CUDA device")
        if quantize_lattice_cuda is None:
            raise RuntimeError("fused CUDA quantizer is unavailable")

        tensor_input = torch.is_tensor(vector)
        x = torch.as_tensor(vector, dtype=self.dtype, device=self.device)
        if x.shape[-1] != DIM:
            raise ValueError(f"expected last dimension {DIM}, got {x.shape[-1]}")

        single = x.ndim == 1
        output_shape = x.shape[:-1]
        x = x.reshape(-1, DIM).contiguous()
        x_scores = x.to(torch.float32).contiguous()

        meta = self._quantize_cuda_metadata()
        rank_meta = self._codeword_cuda_metadata()
        indices = torch.empty((x.shape[0],), dtype=torch.long, device=self.device)
        best_pair = torch.empty((x.shape[0],), dtype=torch.long, device=self.device)
        best_score = torch.empty((x.shape[0],), dtype=torch.float32, device=self.device)
        quantize_lattice_cuda(x_scores, indices, best_pair, best_score, meta, rank_meta, self._binom)

        indices = indices.reshape(output_shape)
        if single:
            indices = indices.reshape(())
        if not tensor_input:
            return int(indices.item())
        return indices

    def _quantize_cuda_metadata(self) -> dict[str, torch.Tensor | int]:
        cached = getattr(self, "_quantize_cuda_meta", None)
        if cached is not None:
            return cached

        parity: list[int] = []
        shell: list[float] = []
        even_weight: list[int] = []
        f0_mags: list[tuple[int, ...]] = []
        f1_mags: list[tuple[int, ...]] = []
        f0_len: list[int] = []
        f1_len: list[int] = []
        required: list[int] = []
        odd_leaders: list[tuple[int, ...]] = []

        for shell_id in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell_id]:
                leader = structure.leader
                parity.append(0 if leader.parity == "even" else 1)
                shell.append(float(shell_id))
                even_weight.append(leader.even_weight if leader.parity == "even" else -1)
                if leader.parity == "even":
                    f0_values = _ascending_multiset(_even_f0_multiplicities(leader))
                    f1_values = _ascending_multiset(_even_f1_multiplicities(leader))
                    odd_values: tuple[int, ...] = (0,) * DIM
                else:
                    f0_values = ()
                    f1_values = ()
                    odd_values = tuple(sorted(_odd_signed_leader_values(leader)))
                f0_len.append(len(f0_values))
                f1_len.append(len(f1_values))
                required.append((sum(leader.leader) % 8) // 4 if leader.parity == "even" else 0)
                f0_mags.append(f0_values + (0,) * (DIM - len(f0_values)))
                f1_mags.append(f1_values + (0,) * (DIM - len(f1_values)))
                odd_leaders.append(odd_values)

        golay_bits: list[int] = []
        golay_weight: list[int] = []
        for codeword in self.golay_codewords:
            mask = 0
            weight = 0
            for bit, value in enumerate(codeword):
                if value:
                    mask |= 1 << bit
                    weight += 1
            golay_bits.append(mask)
            golay_weight.append(weight)

        n_classes = len(parity)
        n_golay = len(self.golay_codewords)
        valid_pair_class: list[int] = []
        valid_pair_golay: list[int] = []
        valid_pair_order: list[int] = []
        valid_pair_shell: list[int] = []
        for class_id, (class_parity, required_weight) in enumerate(zip(parity, even_weight)):
            for golay_index, weight in enumerate(golay_weight):
                if class_parity == 0 and weight != required_weight:
                    continue
                valid_pair_class.append(class_id)
                valid_pair_golay.append(golay_index)
                valid_pair_order.append(golay_index * n_classes + class_id)
                valid_pair_shell.append(int(shell[class_id]))

        min_shell = 2
        n_shells = self.max_shell - min_shell + 1
        shell_slot_start: list[int] = []
        shell_slot_count: list[int] = []
        for shell_id in range(min_shell, self.max_shell + 1):
            start = len(valid_pair_shell)
            try:
                start = valid_pair_shell.index(shell_id)
            except ValueError:
                pass
            count = sum(1 for item in valid_pair_shell if item == shell_id)
            shell_slot_start.append(start)
            shell_slot_count.append(count)
        max_shell_tiles = max((count + 4096 - 1) // 4096 for count in shell_slot_count)

        self._quantize_cuda_meta = {
            "n_classes": n_classes,
            "n_golay": n_golay,
            "total_pairs": n_classes * n_golay,
            "total_valid_pairs": len(valid_pair_class),
            "tile_size": 4096,
            "tile_batch_threshold": 4096,
            "prune_by_shell": True,
            "min_shell": min_shell,
            "n_shells": n_shells,
            "max_shell_tiles": max_shell_tiles,
            "golay_bits": torch.tensor(golay_bits, dtype=torch.int32, device=self.device),
            "golay_weight": torch.tensor(golay_weight, dtype=torch.int32, device=self.device),
            "class_parity": torch.tensor(parity, dtype=torch.int32, device=self.device),
            "class_shell": torch.tensor(shell, dtype=torch.float32, device=self.device),
            "class_even_weight": torch.tensor(even_weight, dtype=torch.int32, device=self.device),
            "class_f0_mags": torch.tensor(f0_mags, dtype=torch.int32, device=self.device),
            "class_f1_mags": torch.tensor(f1_mags, dtype=torch.int32, device=self.device),
            "class_f0_len": torch.tensor(f0_len, dtype=torch.int32, device=self.device),
            "class_f1_len": torch.tensor(f1_len, dtype=torch.int32, device=self.device),
            "class_required": torch.tensor(required, dtype=torch.int32, device=self.device),
            "class_odd_leaders": torch.tensor(odd_leaders, dtype=torch.int32, device=self.device),
            "valid_pair_class": torch.tensor(valid_pair_class, dtype=torch.int32, device=self.device),
            "valid_pair_golay": torch.tensor(valid_pair_golay, dtype=torch.int32, device=self.device),
            "valid_pair_order": torch.tensor(valid_pair_order, dtype=torch.int32, device=self.device),
            "valid_pair_shell": torch.tensor(valid_pair_shell, dtype=torch.int32, device=self.device),
            "shell_slot_start": torch.tensor(shell_slot_start, dtype=torch.int32, device=self.device),
            "shell_slot_count": torch.tensor(shell_slot_count, dtype=torch.int32, device=self.device),
        }
        return self._quantize_cuda_meta

    def _dequantize_lattice_cuda(self, indices: torch.Tensor, *, check_bounds: bool = True) -> torch.Tensor:
        if self.device.type != "cuda":
            raise ValueError("dequantize_lattice_cuda requires a CUDA device")
        if dequantize_lattice_cuda is None:
            raise RuntimeError("fused CUDA dequantizer is unavailable")
        if check_bounds and ((indices < 0) | (indices >= self.total_count)).any():
            raise IndexError("global index outside codebook range")
        meta = self._codeword_cuda_metadata()
        out = torch.empty((indices.shape[0], DIM), dtype=torch.long, device=self.device)
        return dequantize_lattice_cuda(indices.contiguous(), out, meta, self._binom)

    def _codeword_cuda_metadata(self) -> dict[str, torch.Tensor | int]:
        cached = getattr(self, "_codeword_cuda_meta", None)
        if cached is not None:
            return cached

        max_distinct = 1
        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                leader = structure.leader
                if leader.parity == "even":
                    max_distinct = max(
                        max_distinct,
                        len(_even_f0_multiplicities(leader)),
                        len(_even_f1_multiplicities(leader)),
                    )
                else:
                    max_distinct = max(max_distinct, len(_odd_multiplicities(leader)))
        max_vals = 1 << (max_distinct - 1).bit_length()
        class_starts = []
        parity = []
        perm_count = []
        sign_count = []
        f1_perm = []
        golay_start = []
        golay_count = []
        golay_indices = []
        f0_counts = []
        f1_counts = []
        odd_counts = []
        f0_values = []
        f1_values = []
        odd_values = []

        def counts_and_values(multiplicities: dict[int, int]) -> tuple[list[int], list[int]]:
            values = [value for value, count in sorted(multiplicities.items(), reverse=True) if count]
            counts = [multiplicities[value] for value in values]
            values = values + [0] * (max_vals - len(values))
            counts = counts + [0] * (max_vals - len(counts))
            return counts[:max_vals], values[:max_vals]

        for shell in range(2, self.max_shell + 1):
            for structure in self.local_structures[shell]:
                leader = structure.leader
                class_starts.append(self.shell_offsets[shell] + self.class_offsets[shell][structure.class_index])
                parity.append(0 if leader.parity == "even" else 1)
                perm_count.append(structure.permutation_count)
                sign_count.append(structure.sign_count)
                golay_start.append(len(golay_indices))
                golay_count.append(len(structure.golay_codeword_indices))
                golay_indices.extend(structure.golay_codeword_indices)
                if leader.parity == "even":
                    f0c, f0v = counts_and_values(_even_f0_multiplicities(leader))
                    f1c, f1v = counts_and_values(_even_f1_multiplicities(leader))
                    oddc, oddv = [0] * max_vals, [0] * max_vals
                    f1_perm.append(multiset_permutation_count(_even_f1_multiplicities(leader)))
                else:
                    f0c, f0v = [0] * max_vals, [0] * max_vals
                    f1c, f1v = [0] * max_vals, [0] * max_vals
                    oddc, oddv = counts_and_values(_odd_multiplicities(leader))
                    f1_perm.append(1)
                f0_counts.append(f0c)
                f1_counts.append(f1c)
                odd_counts.append(oddc)
                f0_values.append(f0v)
                f1_values.append(f1v)
                odd_values.append(oddv)

        golay_bits = []
        for codeword in self.golay_codewords:
            mask = 0
            for bit, value in enumerate(codeword):
                if value:
                    mask |= 1 << bit
            golay_bits.append(mask)

        self._codeword_cuda_meta = {
            "max_vals": max_vals,
            "n_classes": len(class_starts),
            "class_starts": torch.tensor(class_starts, dtype=torch.long, device=self.device),
            "parity": torch.tensor(parity, dtype=torch.int32, device=self.device),
            "perm_count": torch.tensor(perm_count, dtype=torch.long, device=self.device),
            "sign_count": torch.tensor(sign_count, dtype=torch.long, device=self.device),
            "f1_perm": torch.tensor(f1_perm, dtype=torch.long, device=self.device),
            "golay_start": torch.tensor(golay_start, dtype=torch.long, device=self.device),
            "golay_count": torch.tensor(golay_count, dtype=torch.long, device=self.device),
            "golay_indices": torch.tensor(golay_indices, dtype=torch.long, device=self.device),
            "golay_bits": torch.tensor(golay_bits, dtype=torch.int32, device=self.device),
            "f0_counts": torch.tensor(f0_counts, dtype=torch.long, device=self.device),
            "f1_counts": torch.tensor(f1_counts, dtype=torch.long, device=self.device),
            "odd_counts": torch.tensor(odd_counts, dtype=torch.long, device=self.device),
            "f0_values": torch.tensor(f0_values, dtype=torch.long, device=self.device),
            "f1_values": torch.tensor(f1_values, dtype=torch.long, device=self.device),
            "odd_values": torch.tensor(odd_values, dtype=torch.long, device=self.device),
        }
        return self._codeword_cuda_meta

    def _build_shell_offsets(self) -> dict[int, int]:
        offsets = {}
        cursor = 0
        for shell in range(2, self.max_shell + 1):
            offsets[shell] = cursor
            cursor += sum(row.count for row in self.classes_by_shell[shell])
        return offsets

    def _build_class_offsets(self) -> dict[int, tuple[int, ...]]:
        out = {}
        for shell, rows in self.classes_by_shell.items():
            cursor = 0
            offsets = []
            for row in rows:
                offsets.append(cursor)
                cursor += row.count
            out[shell] = tuple(offsets)
        return out

    def _build_local_structures(self) -> dict[int, tuple[ClassLocalStructure, ...]]:
        return {
            shell: tuple(
                build_local_structure(shell, i, row, self.golay_codewords)
                for i, row in enumerate(rows)
            )
            for shell, rows in self.classes_by_shell.items()
        }

    def _build_class_index(self) -> dict[int, dict[tuple[tuple[int, ...], str], int]]:
        out = {}
        for shell, rows in self.classes_by_shell.items():
            out[shell] = {(row.leader, row.parity): index for index, row in enumerate(rows)}
        return out

    def _validate(self) -> None:
        theta_counts = leech_shell_counts(self.max_shell)
        for shell, rows in self.classes_by_shell.items():
            table_count = sum(row.count for row in rows)
            if table_count != theta_counts[shell]:
                raise ValueError(
                    f"shell {shell} leaders sum to {table_count}, theta series gives {theta_counts[shell]}"
                )
            for structure in self.local_structures[shell]:
                if structure.count != structure.leader.count:
                    raise ValueError(
                        f"shell {shell}, class {structure.class_index} has local count "
                        f"{structure.count}, leader count {structure.leader.count}"
                    )


def build_local_structure(
    shell: int,
    class_index: int,
    leader: ClassLeader,
    golay_codewords: Sequence[tuple[int, ...]],
) -> ClassLocalStructure:
    if leader.parity == "even":
        weight = leader.even_weight
        golay_indices = tuple(i for i, c in enumerate(golay_codewords) if sum(c) == weight)
        f0_count = multiset_permutation_count(_even_f0_multiplicities(leader))
        f1_count = multiset_permutation_count(_even_f1_multiplicities(leader))
        if weight == 0 and sum(leader.leader) % 8 != 0:
            sign_count = 0
        else:
            sign_exponent = leader.nonzero_count if weight == 0 else leader.nonzero_count - 1
            sign_count = 1 << sign_exponent
        permutation_count = f0_count * f1_count
    elif leader.parity == "odd":
        golay_indices = tuple(range(len(golay_codewords)))
        sign_count = 1
        permutation_count = multiset_permutation_count(_odd_multiplicities(leader))
    else:
        raise ValueError(f"unknown parity {leader.parity!r}")

    return ClassLocalStructure(
        shell=shell,
        class_index=class_index,
        leader=leader,
        golay_codeword_indices=golay_indices,
        sign_count=sign_count,
        permutation_count=permutation_count,
    )


def generate_class_leaders(
    max_shell: int,
    cache_dir: str | Path | None = ".llvq_cache",
    use_cache: bool = True,
    verbose: bool = False,
) -> tuple[ClassLeader, ...]:
    """Generate class leaders from the integer shell equation."""
    leaders = []
    for shell in range(2, max_shell + 1):
        if use_cache:
            start = perf_counter()
            shell_leaders, cache_status = generate_shell_class_leaders_cached(shell, cache_dir=cache_dir)
            if verbose:
                print(
                    f"{cache_status} class leaders for shell m={shell}: "
                    f"{len(shell_leaders)} classes in {perf_counter() - start:.3f}s",
                    flush=True,
                )
        else:
            start = perf_counter()
            shell_leaders = generate_shell_class_leaders(shell)
            if verbose:
                print(
                    f"generated class leaders for shell m={shell}: "
                    f"{len(shell_leaders)} classes in {perf_counter() - start:.3f}s",
                    flush=True,
                )
        leaders.extend(shell_leaders)
    return tuple(leaders)


def generate_shell_class_leaders_cached(
    shell: int,
    cache_dir: str | Path | None = ".llvq_cache",
) -> tuple[tuple[ClassLeader, ...], str]:
    cache_key = None if cache_dir is None else str(Path(cache_dir))
    return _generate_shell_class_leaders_cached(shell, cache_key)


def _generate_shell_class_leaders_cached(
    shell: int,
    cache_dir: str | None,
) -> tuple[tuple[ClassLeader, ...], str]:
    cache_path = _class_leader_cache_path(cache_dir, shell)
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if (
                payload.get("version") == CLASS_LEADER_CACHE_VERSION
                and payload.get("shell") == shell
                and isinstance(payload.get("leaders"), tuple)
            ):
                return payload["leaders"], "loaded cached"
        except (OSError, pickle.PickleError, AttributeError, EOFError):
            pass

    leaders = generate_shell_class_leaders(shell)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(
                {
                    "version": CLASS_LEADER_CACHE_VERSION,
                    "shell": shell,
                    "leaders": leaders,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        tmp_path.replace(cache_path)
    return leaders, "generated"


def _class_leader_cache_path(cache_dir: str | None, shell: int) -> Path | None:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"class_leaders_v{CLASS_LEADER_CACHE_VERSION}_shell{shell}.pkl"


def generate_shell_class_leaders(shell: int) -> tuple[ClassLeader, ...]:
    target_square_sum = 16 * shell
    leaders = []
    leaders.extend(generate_even_shell_leaders(shell, target_square_sum))
    leaders.extend(generate_odd_shell_leaders(shell, target_square_sum))
    return tuple(sorted(leaders, key=lambda row: (row.leader, 0 if row.parity == "even" else 1, row.count)))


def generate_even_shell_leaders(shell: int, target_square_sum: int) -> tuple[ClassLeader, ...]:
    max_abs = int(target_square_sum**0.5)
    magnitudes = tuple(value for value in range(0, max_abs + 1, 2))
    leaders = []
    for counts in _multiplicity_solutions(magnitudes, DIM, target_square_sum):
        leader = _leader_from_counts(shell, "even", counts, count=0)
        count = class_cardinality(leader)
        if count:
            leaders.append(_replace_count(leader, count))
    return tuple(leaders)


def generate_odd_shell_leaders(shell: int, target_square_sum: int) -> tuple[ClassLeader, ...]:
    max_abs = int(target_square_sum**0.5)
    magnitudes = tuple(value for value in range(1, max_abs + 1, 2))
    leaders = []
    for counts in _multiplicity_solutions(magnitudes, DIM, target_square_sum):
        leader = _leader_from_counts(shell, "odd", counts, count=0)
        count = class_cardinality(leader)
        if count:
            leaders.append(_replace_count(leader, count))
    return tuple(leaders)


def _multiplicity_solutions(
    magnitudes: Sequence[int],
    remaining_slots: int,
    remaining_square_sum: int,
) -> Iterable[dict[int, int]]:
    if not magnitudes:
        if remaining_slots == 0 and remaining_square_sum == 0:
            yield {}
        return

    magnitude = magnitudes[-1]
    square = magnitude * magnitude
    rest = magnitudes[:-1]
    max_count = remaining_slots if square == 0 else min(remaining_slots, remaining_square_sum // square)
    for count in range(max_count, -1, -1):
        next_slots = remaining_slots - count
        next_sum = remaining_square_sum - count * square
        for solution in _multiplicity_solutions(rest, next_slots, next_sum):
            if count:
                solution = dict(solution)
                solution[magnitude] = count
            yield solution


def class_cardinality(leader: ClassLeader) -> int:
    structure = build_local_structure(
        shell=leader.shell,
        class_index=0,
        leader=leader,
        golay_codewords=extended_golay_codewords(),
    )
    return structure.count


def _replace_count(leader: ClassLeader, count: int) -> ClassLeader:
    return _leader_from_counts(leader.shell, leader.parity, leader.multiplicities, count=count)


def _leader_from_counts(shell: int, parity: str, counts: dict[int, int], count: int) -> ClassLeader:
    fixed = {magnitude: counts.get(magnitude, 0) for magnitude in (8, 6, 5, 4, 3, 2, 1, 0)}
    extras = tuple(sorted(
        ((magnitude, value) for magnitude, value in counts.items() if magnitude not in fixed and value),
        reverse=True,
    ))
    return ClassLeader(
        shell=shell,
        parity=parity,
        count=count,
        mult_abs8=fixed[8],
        mult_abs6=fixed[6],
        mult_abs5=fixed[5],
        mult_abs4=fixed[4],
        mult_abs3=fixed[3],
        mult_abs2=fixed[2],
        mult_abs1=fixed[1],
        mult_abs0=fixed[0],
        extra_multiplicities=extras,
    )


def _even_f0_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 4 == 0}


def _even_f1_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 4 == 2}


def _odd_multiplicities(leader: ClassLeader) -> dict[int, int]:
    return {magnitude: count for magnitude, count in leader.multiplicities.items() if magnitude % 2 == 1}


def compare_with_reference(max_shell: int = 4) -> bool:
    generated = tuple(row for row in generate_class_leaders(max_shell) if row.shell <= max_shell)
    reference = tuple(
        sorted(
            (row for row in TABLE_CLASS_LEADERS_REFERENCE if row.shell <= max_shell),
            key=lambda row: (row.shell, row.leader, 0 if row.parity == "even" else 1, row.count),
        )
    )
    generated = tuple(
        sorted(generated, key=lambda row: (row.shell, row.leader, 0 if row.parity == "even" else 1, row.count))
    )
    return generated == reference


def _ascending_multiset(multiplicities: dict[int, int]) -> tuple[int, ...]:
    values = []
    for magnitude, count in sorted(multiplicities.items()):
        values.extend([magnitude] * count)
    return tuple(values)


def _odd_signed_leader_values(leader: ClassLeader) -> tuple[int, ...]:
    values = []
    for magnitude, count in _odd_multiplicities(leader).items():
        signed = magnitude if magnitude % 4 == 1 else -magnitude
        values.extend([signed] * count)
    return tuple(values)


def unrank_class_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    if leader.parity == "even":
        return unrank_even_codeword(leader, local)
    if leader.parity == "odd":
        return unrank_odd_codeword(leader, local)
    raise ValueError(f"unknown parity {leader.parity!r}")


def rank_class_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> int:
    if structure.leader.parity == "even":
        local = rank_even_codeword(vector, structure, golay_index_by_codeword)
    elif structure.leader.parity == "odd":
        local = rank_odd_codeword(vector, structure, golay_index_by_codeword)
    else:
        raise ValueError(f"unknown parity {structure.leader.parity!r}")

    return (
        (local.golay_choice * structure.sign_count + local.sign_choice)
        * structure.permutation_count
        + local.permutation_choice
    )


def rank_even_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> LocalDecomposition:
    codeword = tuple(1 if abs(value) % 4 == 2 else 0 for value in vector)
    golay_choice = _compatible_golay_choice(codeword, structure, golay_index_by_codeword)

    f0_values = tuple(abs(value) for value, bit in zip(vector, codeword, strict=True) if bit == 0)
    f1_values = tuple(abs(value) for value, bit in zip(vector, codeword, strict=True) if bit == 1)
    f0_counts = _even_f0_multiplicities(structure.leader)
    f1_counts = _even_f1_multiplicities(structure.leader)
    f0_rank = rank_multiset_sequence(f0_values, f0_counts)
    f1_rank = rank_multiset_sequence(f1_values, f1_counts)
    f1_permutations = multiset_permutation_count(f1_counts)
    permutation_choice = f0_rank * f1_permutations + f1_rank

    sign_choice = rank_even_signs(vector)
    if sign_choice >= structure.sign_count:
        raise ValueError("even sign pattern is outside the class sign range")

    return LocalDecomposition(
        golay_choice=golay_choice,
        golay_codeword=codeword,
        sign_choice=sign_choice,
        permutation_choice=permutation_choice,
    )


def rank_odd_codeword(
    vector: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> LocalDecomposition:
    codeword = tuple(0 if value % 4 == 1 else 1 if value % 4 == 3 else -1 for value in vector)
    if any(bit < 0 for bit in codeword):
        raise ValueError("odd coordinates must be congruent to 1 or 3 mod 4")
    golay_choice = _compatible_golay_choice(codeword, structure, golay_index_by_codeword)
    magnitudes = tuple(abs(value) for value in vector)
    permutation_choice = rank_multiset_sequence(magnitudes, _odd_multiplicities(structure.leader))
    return LocalDecomposition(
        golay_choice=golay_choice,
        golay_codeword=codeword,
        sign_choice=0,
        permutation_choice=permutation_choice,
    )


def _compatible_golay_choice(
    codeword: tuple[int, ...],
    structure: ClassLocalStructure,
    golay_index_by_codeword: dict[tuple[int, ...], int],
) -> int:
    try:
        golay_index = golay_index_by_codeword[codeword]
    except KeyError as exc:
        raise ValueError("coordinate congruence pattern is not a Golay codeword") from exc
    try:
        return structure.golay_codeword_indices.index(golay_index)
    except ValueError as exc:
        raise ValueError("Golay codeword is not compatible with this class") from exc


def unrank_even_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    f0_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 0)
    f1_positions = tuple(i for i, bit in enumerate(local.golay_codeword) if bit == 1)
    f0_counts = _even_f0_multiplicities(leader)
    f1_counts = _even_f1_multiplicities(leader)
    f1_permutations = multiset_permutation_count(f1_counts)
    f1_rank = local.permutation_choice % f1_permutations
    f0_rank = local.permutation_choice // f1_permutations

    unsigned = [0] * DIM
    for position, magnitude in zip(f0_positions, unrank_multiset_sequence(f0_counts, f0_rank), strict=True):
        unsigned[position] = magnitude
    for position, magnitude in zip(f1_positions, unrank_multiset_sequence(f1_counts, f1_rank), strict=True):
        unsigned[position] = magnitude
    return apply_even_signs(tuple(unsigned), local.sign_choice)


def unrank_odd_codeword(leader: ClassLeader, local: LocalDecomposition) -> tuple[int, ...]:
    magnitudes = unrank_multiset_sequence(
        _odd_multiplicities(leader),
        local.permutation_choice,
    )
    values = []
    for magnitude, bit in zip(magnitudes, local.golay_codeword, strict=True):
        target_mod4 = 3 if bit else 1
        sign = 1 if magnitude % 4 == target_mod4 else -1
        values.append(sign * magnitude)
    return tuple(values)


def apply_even_signs(unsigned: tuple[int, ...], sign_rank: int) -> tuple[int, ...]:
    signable_positions = tuple(i for i, value in enumerate(unsigned) if value != 0)
    for mask in range(1 << len(signable_positions)):
        signed = list(unsigned)
        for bit, position in enumerate(signable_positions):
            if (mask >> bit) & 1:
                signed[position] = -signed[position]
        if sum(signed) % 8 != 0:
            continue
        if sign_rank == 0:
            return tuple(signed)
        sign_rank -= 1
    raise IndexError(sign_rank)


def rank_even_signs(vector: tuple[int, ...]) -> int:
    if sum(vector) % 8 != 0:
        raise ValueError("even vector coordinate sum is not 0 mod 8")

    unsigned = tuple(abs(value) for value in vector)
    signable_positions = tuple(i for i, value in enumerate(unsigned) if value != 0)
    target_mask = 0
    for bit, position in enumerate(signable_positions):
        if vector[position] < 0:
            target_mask |= 1 << bit

    rank = 0
    for mask in range(1 << len(signable_positions)):
        signed_sum = 0
        for bit, position in enumerate(signable_positions):
            sign = -1 if (mask >> bit) & 1 else 1
            signed_sum += sign * unsigned[position]
        if signed_sum % 8 != 0:
            continue
        if mask == target_mask:
            return rank
        rank += 1
    raise ValueError("even sign pattern is not valid for this class")


def unrank_multiset_sequence(multiplicities: dict[int, int], rank: int) -> tuple[int, ...]:
    counts = {value: count for value, count in multiplicities.items() if count}
    total = multiset_permutation_count(counts)
    if rank < 0 or rank >= total:
        raise IndexError(rank)

    values = tuple(sorted(counts, reverse=True))
    out = []
    remaining = sum(counts.values())
    while remaining:
        for value in values:
            if counts.get(value, 0) == 0:
                continue
            counts[value] -= 1
            branch_count = multiset_permutation_count(counts)
            if rank < branch_count:
                out.append(value)
                remaining -= 1
                break
            rank -= branch_count
            counts[value] += 1
        else:
            raise RuntimeError("failed to unrank multiset sequence")
    return tuple(out)


def rank_multiset_sequence(sequence: Sequence[int], multiplicities: dict[int, int]) -> int:
    counts = {value: count for value, count in multiplicities.items() if count}
    if len(sequence) != sum(counts.values()):
        raise ValueError("sequence length does not match multiplicities")

    rank = 0
    values = tuple(sorted(counts, reverse=True))
    for actual in sequence:
        if counts.get(actual, 0) == 0:
            raise ValueError(f"value {actual} is not available in this multiset")
        for candidate in values:
            if candidate == actual:
                break
            if counts.get(candidate, 0) == 0:
                continue
            counts[candidate] -= 1
            rank += multiset_permutation_count(counts)
            counts[candidate] += 1
        counts[actual] -= 1
    return rank


def multiset_permutation_count(multiplicities: dict[int, int]) -> int:
    total = sum(multiplicities.values())
    out = factorial(total)
    for count in multiplicities.values():
        out //= factorial(count)
    return out


def squared_distance(x: Sequence[float], y: Sequence[int]) -> float:
    return sum((xi - yi) * (xi - yi) for xi, yi in zip(x, y, strict=True))


def dot_product(x: Sequence[float], y: Sequence[int]) -> float:
    return sum(xi * yi for xi, yi in zip(x, y, strict=True))


def format_float_vector(vector: Sequence[float] | None, digits: int = 4) -> tuple[float, ...]:
    if vector is None:
        return ()
    return tuple(round(value, digits) for value in vector)


@lru_cache(maxsize=1)
def extended_golay_codewords() -> tuple[tuple[int, ...], ...]:
    """
    Extended binary Golay code G24 in lexicographic order.

    This uses the cyclic perfect Golay [23,12,7] generator
    g(x)=x^11+x^9+x^7+x^6+x^5+x+1, then appends the parity bit.
    """
    generator_exponents = (11, 9, 7, 6, 5, 1, 0)
    generator = sum(1 << exponent for exponent in generator_exponents)
    words = []
    for message in range(1 << 12):
        code23 = 0
        for bit in range(12):
            if (message >> bit) & 1:
                code23 ^= generator << bit
        word23 = tuple((code23 >> bit) & 1 for bit in range(23))
        parity = sum(word23) & 1
        words.append(word23 + (parity,))
    return tuple(sorted(words))


def leech_shell_counts(max_shell: int) -> dict[int, int]:
    e4 = [0] * (max_shell + 1)
    e4[0] = 1
    for n in range(1, max_shell + 1):
        e4[n] = 240 * sigma_power(n, 3)
    e4_cubed = poly_pow_trunc(e4, 3, max_shell)
    delta = delta_coefficients(max_shell)
    return {m: e4_cubed[m] - 720 * delta[m] for m in range(2, max_shell + 1)}


def sigma_power(n: int, power: int) -> int:
    total = 0
    divisor = 1
    while divisor * divisor <= n:
        if n % divisor == 0:
            total += divisor**power
            other = n // divisor
            if other != divisor:
                total += other**power
        divisor += 1
    return total


def poly_mul_trunc(a: Sequence[int], b: Sequence[int], degree: int) -> list[int]:
    out = [0] * (degree + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b[: degree + 1 - i]):
            if bj:
                out[i + j] += ai * bj
    return out


def poly_pow_trunc(base: Sequence[int], exponent: int, degree: int) -> list[int]:
    out = [0] * (degree + 1)
    out[0] = 1
    power = list(base)
    while exponent:
        if exponent & 1:
            out = poly_mul_trunc(out, power, degree)
        exponent >>= 1
        if exponent:
            power = poly_mul_trunc(power, power, degree)
    return out


def delta_coefficients(degree: int) -> list[int]:
    coeffs = [0] * (degree + 1)
    coeffs[0] = 1
    binom24 = [factorial(24) // (factorial(k) * factorial(24 - k)) for k in range(25)]
    for n in range(1, degree + 1):
        factor = [0] * (degree + 1)
        for k, choose in enumerate(binom24):
            exponent = n * k
            if exponent > degree:
                break
            factor[exponent] = (-1) ** k * choose
        coeffs = poly_mul_trunc(coeffs, factor, degree)

    shifted = [0] * (degree + 1)
    for i in range(degree):
        shifted[i + 1] = coeffs[i]
    return shifted


def demo(
    max_shell: int = 2,
    samples: int = 1,
    seed: int = 0,
) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = Random(seed)
    sample_vectors = [tuple(rng.gauss(0.0, 1.0) for _ in range(DIM)) for _ in range(samples)]
    print(f"samples: {samples}")
    print(f"input vector[0]: {format_float_vector(sample_vectors[0])}")

    db = LeechLatticeVectorQuantizerGpu(max_shell=max_shell)
    x = torch.tensor(sample_vectors, dtype=db.dtype, device=db.device)
    quantized_indices = db.quantize(x)
    quantized_vectors = db.dequantize(quantized_indices)
    total_squared_error = (x - quantized_vectors).square().sum().item()
    first_index = int(quantized_indices[0].item())

    mse = total_squared_error / (samples * DIM)
    ranked = db.unrank(first_index)
    local = db.decompose_local_index(ranked)

    print(f"shells: 2..{max_shell}")
    print(f"total codewords: {db.total_count}")
    print(f"shape bits: {db.shape_bits}")
    print(f"shape bits/dim: {db.shape_bits / DIM:.6f}")
    if max_shell <= 4:
        print(f"generated leaders match reference table: {compare_with_reference(max_shell)}")
    print(f"quantized index[0]:   {first_index}")
    print(
        "quantized address[0]: "
        f"shell={ranked.shell}, class={ranked.class_index}, "
        f"class_local={ranked.class_local_index}, golay={local.golay_choice}, "
        f"sign={local.sign_choice}, perm={local.permutation_choice}"
    )
    print(f"integer codeword[0]:  {db.dequantize_lattice(first_index)}")
    print(f"dequantized vector[0]:{format_float_vector(db.dequantize(first_index))}")
    print(f"mse: {mse:.6f}")


def benchmark(
    max_shell: int = 4,
    samples: int = 64,
    seed: int = 0,
    device: str | None = None,
) -> None:
    from .leech_lattice_vector_quantizer import LeechLatticeVectorQuantizer

    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = Random(seed)
    sample_vectors = [tuple(rng.gauss(0.0, 1.0) for _ in range(DIM)) for _ in range(samples)]

    print(f"max_shell: {max_shell}")
    print(f"samples: {samples}")

    cpu_q = LeechLatticeVectorQuantizer(max_shell=max_shell)
    start = perf_counter()
    cpu_indices = [cpu_q.quantize(vector) for vector in sample_vectors]
    cpu_seconds = perf_counter() - start
    print(
        f"cpu original structured: {cpu_seconds:.6f}s total, "
        f"{cpu_seconds / samples * 1000.0:.3f} ms/vector"
    )

    gpu_q = LeechLatticeVectorQuantizerGpu(max_shell=max_shell, device=device)
    if gpu_q.device.type != "cuda":
        raise ValueError("benchmark requires a CUDA device")
    print(f"gpu device: {gpu_q.device}")
    x = torch.tensor(sample_vectors, dtype=gpu_q.dtype, device=gpu_q.device)

    _ = gpu_q.quantize_cuda(x[:1])
    torch.cuda.synchronize(gpu_q.device)
    start = perf_counter()
    cuda_indices = gpu_q.quantize(x)
    torch.cuda.synchronize(gpu_q.device)
    cuda_seconds = perf_counter() - start
    cuda_list = cuda_indices.detach().cpu().tolist()
    print(
        f"gpu cuda fused:         {cuda_seconds:.6f}s total, "
        f"{cuda_seconds / samples * 1000.0:.3f} ms/vector"
    )
    print(f"cuda fused matches cpu: {cuda_list == cpu_indices}")
    if cuda_seconds > 0:
        print(f"speedup vs cpu:         {cpu_seconds / cuda_seconds:.2f}x")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Implicit Leech shell/class/local codeword database")
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--indices", type=str, default="0,1103,1104,196559,196560")
    args = parser.parse_args()
    if args.check:
        indices = [int(part) for part in args.indices.split(",") if part.strip()]
        check_demo(max_shell=args.max_shell, indices=indices)
    elif args.benchmark:
        benchmark(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
            device=args.device,
        )
    else:
        demo(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
