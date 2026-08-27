from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import biotite.structure as struc
import numpy as np


@dataclass(frozen=True)
class ClashPair:
    atom_1: str
    atom_2: str
    distance: float
    cutoff: float
    overlap: float


@dataclass(frozen=True)
class ClashScore:
    total_overlap_score: float
    clashing_pair_count: int
    worst_overlap: float
    worst_pair: ClashPair | None

    @property
    def passes(self) -> bool:
        return self.clashing_pair_count == 0


DEFAULT_HEAVY_ATOM_CUTOFF = 2.2
DEFAULT_PD_CUTOFF = 1.8


def score_heavy_atom_clashes(
    atom_array: struc.AtomArray,
    other: struc.AtomArray | None = None,
    bonded_atom_pairs: Iterable[tuple[str, str]] = (),
    ignored_atom_name_pairs: Iterable[tuple[str, str]] = (),
    heavy_atom_cutoff: float = DEFAULT_HEAVY_ATOM_CUTOFF,
    pd_cutoff: float = DEFAULT_PD_CUTOFF,
) -> ClashScore:
    """Score deterministic heavy-atom overlaps within or between atom arrays."""
    pairs = _clashing_pairs(
        atom_array=atom_array,
        other=other,
        bonded_atom_pairs=bonded_atom_pairs,
        ignored_atom_name_pairs=ignored_atom_name_pairs,
        heavy_atom_cutoff=heavy_atom_cutoff,
        pd_cutoff=pd_cutoff,
    )
    if not pairs:
        return ClashScore(
            total_overlap_score=0.0,
            clashing_pair_count=0,
            worst_overlap=0.0,
            worst_pair=None,
        )
    worst_pair = max(pairs, key=lambda pair: pair.overlap)
    return ClashScore(
        total_overlap_score=float(sum(pair.overlap for pair in pairs)),
        clashing_pair_count=len(pairs),
        worst_overlap=worst_pair.overlap,
        worst_pair=worst_pair,
    )


def _clashing_pairs(
    atom_array: struc.AtomArray,
    other: struc.AtomArray | None,
    bonded_atom_pairs: Iterable[tuple[str, str]],
    ignored_atom_name_pairs: Iterable[tuple[str, str]],
    heavy_atom_cutoff: float,
    pd_cutoff: float,
) -> list[ClashPair]:
    bonded_pairs = {frozenset(pair) for pair in bonded_atom_pairs}
    ignored_pairs = {frozenset(pair) for pair in ignored_atom_name_pairs}
    pairs: list[ClashPair] = []

    if other is None:
        heavy_indices = _heavy_atom_indices(atom_array)
        for left_offset, index_1 in enumerate(heavy_indices[:-1]):
            for index_2 in heavy_indices[left_offset + 1 :]:
                atom_name_1 = str(atom_array.atom_name[index_1])
                atom_name_2 = str(atom_array.atom_name[index_2])
                if frozenset((atom_name_1, atom_name_2)) in bonded_pairs:
                    continue
                if frozenset((atom_name_1, atom_name_2)) in ignored_pairs:
                    continue
                pair = _score_pair(atom_array, index_1, atom_array, index_2, heavy_atom_cutoff, pd_cutoff)
                if pair is not None:
                    pairs.append(pair)
        return pairs

    heavy_indices_1 = _heavy_atom_indices(atom_array)
    heavy_indices_2 = _heavy_atom_indices(other)
    for index_1 in heavy_indices_1:
        for index_2 in heavy_indices_2:
            atom_name_1 = str(atom_array.atom_name[index_1])
            atom_name_2 = str(other.atom_name[index_2])
            if frozenset((atom_name_1, atom_name_2)) in ignored_pairs:
                continue
            pair = _score_pair(atom_array, index_1, other, index_2, heavy_atom_cutoff, pd_cutoff)
            if pair is not None:
                pairs.append(pair)
    return pairs


def _score_pair(
    atom_array: struc.AtomArray,
    index_1: int,
    other: struc.AtomArray,
    index_2: int,
    heavy_atom_cutoff: float,
    pd_cutoff: float,
) -> ClashPair | None:
    element_1 = str(atom_array.element[index_1]).upper()
    element_2 = str(other.element[index_2]).upper()
    cutoff = pd_cutoff if "PD" in {element_1, element_2} else heavy_atom_cutoff
    distance = float(np.linalg.norm(atom_array.coord[index_1] - other.coord[index_2]))
    overlap = cutoff - distance
    if overlap <= 0:
        return None
    return ClashPair(
        atom_1=_atom_label(atom_array, index_1),
        atom_2=_atom_label(other, index_2),
        distance=distance,
        cutoff=cutoff,
        overlap=float(overlap),
    )


def _heavy_atom_indices(atom_array: struc.AtomArray) -> np.ndarray:
    elements = np.char.upper(atom_array.element.astype("U2"))
    return np.flatnonzero((elements != "H") & (elements != "D"))


def _atom_label(atom_array: struc.AtomArray, index: int) -> str:
    return (
        f"{atom_array.chain_id[index]}:"
        f"{int(atom_array.res_id[index])}:"
        f"{atom_array.atom_name[index]}"
    )
