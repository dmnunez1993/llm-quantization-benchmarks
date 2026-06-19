from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, factorial, log2, sqrt
from pathlib import Path
import pickle
from random import Random
from time import perf_counter
from typing import Iterable, Sequence


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


class LeechLatticeVectorQuantizer:
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
    ):
        if max_shell < 2:
            raise ValueError("max_shell must be at least 2")
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

    def dequantize_lattice(self, global_index: int) -> tuple[int, ...]:
        return self.codeword(global_index)

    def dequantize(self, global_index: int) -> tuple[float, ...]:
        """Return the scaled Leech lattice vector in Lambda_24."""
        return tuple(LATTICE_SCALE * value for value in self.dequantize_lattice(global_index))

    def encode_codeword(self, vector: Sequence[int]) -> int:
        """
        Return the exact database index for a Leech codeword.

        This is the inverse of `dequantize` for vectors generated by this
        database. It validates shell, class leader, Golay compatibility,
        signs, and permutation rank. It does not perform nearest-neighbor
        search for arbitrary real-valued inputs.
        """
        values = tuple(int(value) for value in vector)
        if len(values) != DIM:
            raise ValueError(f"expected {DIM} coordinates, got {len(values)}")

        square_sum = sum(value * value for value in values)
        if square_sum % 16 != 0:
            raise ValueError(f"vector squared norm {square_sum} is not a Leech shell norm")
        shell = square_sum // 16
        if shell < 2 or shell > self.max_shell:
            raise ValueError(f"shell {shell} is outside database range 2..{self.max_shell}")

        parity = "even" if all(value % 2 == 0 for value in values) else "odd"
        if parity == "odd" and not all(value % 2 != 0 for value in values):
            raise ValueError("coordinates must be all even or all odd")

        leader_key = tuple(sorted((abs(value) for value in values), reverse=True))
        class_key = (leader_key, parity)
        try:
            class_index = self.class_index_by_shell_and_key[shell][class_key]
        except KeyError as exc:
            raise ValueError(f"no generated class matches shell={shell}, parity={parity}, leader={leader_key}") from exc

        structure = self.local_structures[shell][class_index]
        local = rank_class_codeword(values, structure, self.golay_index_by_codeword)
        return self.rank_class(shell, class_index, local)

    def quantize(self, vector: Sequence[float]) -> int:
        """
        Quantize a real-valued 24-D vector by Adoul-Barth subclass search.

        The search maximizes the projection x^T y over all generated shells,
        classes, and Golay subclasses. Each subclass is solved as a permutation
        code with the even/odd sign rules from the Leech construction.
        """
        values = tuple(float(value) for value in vector)
        if len(values) != DIM:
            raise ValueError(f"expected {DIM} coordinates, got {len(values)}")

        best_score = float("-inf")
        best_codeword: tuple[int, ...] | None = None
        x_norm = sqrt(sum(value * value for value in values))
        shell_score_upper_bounds = {
            shell: x_norm * sqrt(2.0 * shell) - shell
            for shell in range(2, self.max_shell + 1)
        }
        for codeword in self.golay_codewords:
            for shell in range(2, self.max_shell + 1):
                if shell_score_upper_bounds[shell] < best_score:
                    continue
                for structure in self.local_structures[shell]:
                    if structure.leader.parity == "even":
                        if sum(codeword) != structure.leader.even_weight:
                            continue
                        projection, candidate = solve_even_subclass(values, structure.leader, codeword)
                    else:
                        projection, candidate = solve_odd_subclass(values, structure.leader, codeword)
                    score = LATTICE_SCALE * projection - shell
                    if score > best_score:
                        best_score = score
                        best_codeword = candidate

        if best_codeword is None:
            raise RuntimeError("no Leech candidate found")

        print(f"best score {best_score:.6f} for codeword {best_codeword}")
        return self.encode_codeword(best_codeword)

    def quantize_exhaustive(self, vector: Sequence[float], max_candidates: int = 1_000_000) -> int:
        """Old reference path: scan every dequantized codeword."""
        values = tuple(float(value) for value in vector)
        if len(values) != DIM:
            raise ValueError(f"expected {DIM} coordinates, got {len(values)}")
        if self.total_count > max_candidates:
            raise ValueError(
                f"exhaustive quantize would scan {self.total_count} codewords; "
                "increase max_candidates to run this check"
            )

        best_index = 0
        best_distance = float("inf")
        for index in range(self.total_count):
            distance = squared_distance(values, self.dequantize_lattice(index))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

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


def solve_even_subclass(
    x: Sequence[float],
    leader: ClassLeader,
    golay_codeword: tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    candidate = [0] * DIM
    f0_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 0)
    f1_positions = tuple(i for i, bit in enumerate(golay_codeword) if bit == 1)
    f0_magnitudes = _ascending_multiset(_even_f0_multiplicities(leader))
    f1_magnitudes = _ascending_multiset(_even_f1_multiplicities(leader))

    ordered_f0 = sorted(f0_positions, key=lambda pos: (abs(x[pos]), pos))
    ordered_f1 = sorted(f1_positions, key=lambda pos: (abs(x[pos]), pos))

    for position, magnitude in zip(ordered_f0, f0_magnitudes, strict=True):
        candidate[position] = magnitude if x[position] >= 0 else -magnitude
    for position, magnitude in zip(ordered_f1, f1_magnitudes, strict=True):
        candidate[position] = magnitude if x[position] >= 0 else -magnitude

    if f1_magnitudes:
        required_negative_parity = (sum(leader.leader) % 8) // 4
        actual_negative_parity = sum(1 for position in f1_positions if candidate[position] < 0) % 2
        if actual_negative_parity != required_negative_parity:
            position_to_flip = ordered_f1[0]
            candidate[position_to_flip] = -candidate[position_to_flip]

    vector = tuple(candidate)
    return dot_product(x, vector), vector


def solve_odd_subclass(
    x: Sequence[float],
    leader: ClassLeader,
    golay_codeword: tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    x_prime = tuple(-value if bit else value for value, bit in zip(x, golay_codeword, strict=True))
    ordered_positions = sorted(range(DIM), key=lambda pos: (x_prime[pos], pos))
    signed_leader = sorted(_odd_signed_leader_values(leader))

    candidate_prime = [0] * DIM
    for position, value in zip(ordered_positions, signed_leader, strict=True):
        candidate_prime[position] = value

    candidate = tuple(
        -value if bit else value
        for value, bit in zip(candidate_prime, golay_codeword, strict=True)
    )
    return dot_product(x, candidate), candidate


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
    max_candidates: int = 1_000_000,
) -> None:
    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = Random(seed)
    sample_vectors = [tuple(rng.gauss(0.0, 1.0) for _ in range(DIM)) for _ in range(samples)]
    print(f"samples: {samples}")
    print(f"input vector[0]: {format_float_vector(sample_vectors[0])}")

    db = LeechLatticeVectorQuantizer(max_shell=max_shell)
    total_squared_error = 0.0
    first_index = 0
    first_exhaustive_index: int | None = None
    first_quantized: tuple[int, ...] | None = None
    exhaustive_matches = True
    exhaustive_checked = db.total_count <= max_candidates

    for sample_index, vector in enumerate(sample_vectors):
        quantized_index = db.quantize(vector)
        if exhaustive_checked:
            exhaustive_index = db.quantize_exhaustive(vector, max_candidates=max_candidates)
            exhaustive_matches = exhaustive_matches and (quantized_index == exhaustive_index)
        else:
            exhaustive_index = None
        quantized_vector = db.dequantize_lattice(quantized_index)
        total_squared_error += squared_distance(vector, quantized_vector)
        if sample_index == 0:
            first_index = quantized_index
            first_exhaustive_index = exhaustive_index
            first_quantized = quantized_vector

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
    if exhaustive_checked:
        print(f"exhaustive index[0]:  {first_exhaustive_index}")
        print(f"nn matches exhaustive:{exhaustive_matches}")
    else:
        print(f"exhaustive check:     skipped; {db.total_count} codewords exceeds max_candidates")
    print(
        "quantized address[0]: "
        f"shell={ranked.shell}, class={ranked.class_index}, "
        f"class_local={ranked.class_local_index}, golay={local.golay_choice}, "
        f"sign={local.sign_choice}, perm={local.permutation_choice}"
    )
    print(f"integer codeword[0]:  {db.dequantize(first_index)}")
    print(f"dequantized vector[0]:{format_float_vector(db.dequantize_lattice(first_index))}")
    print(f"mse: {mse:.6f}")


def check_demo(max_shell: int = 4, indices: Iterable[int] = (0, 1103, 1104, 196559, 196560)) -> None:
    db = LeechLatticeVectorQuantizer(max_shell=max_shell)
    print(f"shells: 2..{max_shell}")
    print(f"total codewords: {db.total_count}")
    print(f"shape bits: {db.shape_bits}")
    if max_shell <= 4:
        print(f"generated leaders match reference table: {compare_with_reference(max_shell)}")
    for shell in range(2, max_shell + 1):
        print(f"m={shell}: n(m)={db.shell_counts[shell]}, N(m)={db.cumulative_shell_counts[shell]}")
        for i, structure in enumerate(db.local_structures[shell]):
            leader = structure.leader
            print(
                f"  class={i} parity={leader.parity} leader={leader.leader} "
                f"count={leader.count} golay={len(structure.golay_codeword_indices)} "
                f"signs={structure.sign_count} perms={structure.permutation_count}"
            )

    for index in indices:
        ranked = db.unrank(index)
        local = db.decompose_local_index(ranked)
        word = db.dequantize(index)
        quantized_index = db.encode_codeword(word)
        print(
            f"index={index}: shell={ranked.shell} class={ranked.class_index} "
            f"local={ranked.class_local_index} golay={local.golay_choice} "
            f"sign={local.sign_choice} perm={local.permutation_choice} "
            f"dequantize={word} quantize={quantized_index}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Implicit Leech shell/class/local codeword database")
    parser.add_argument("--max-shell", type=int, default=2)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=1_000_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--indices", type=str, default="0,1103,1104,196559,196560")
    args = parser.parse_args()
    if args.check:
        indices = [int(part) for part in args.indices.split(",") if part.strip()]
        check_demo(max_shell=args.max_shell, indices=indices)
    else:
        demo(
            max_shell=args.max_shell,
            samples=args.samples,
            seed=args.seed,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
