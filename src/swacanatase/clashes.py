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
HYDROGEN_ELEMENTS = ("H", "h", "D", "d")
PD_ELEMENTS = ("PD", "Pd", "pD", "pd")


def score_heavy_atom_clashes(
    atom_array: struc.AtomArray,
    other: struc.AtomArray | None = None,
    bonded_atom_pairs: Iterable[tuple[str, str]] = (),
    ignored_atom_name_pairs: Iterable[tuple[str, str]] = (),
    ignored_atom_index_pairs: Iterable[tuple[int, int]] = (),
    ignore_same_residue: bool = False,
    ignore_inter_residue_backbone_n_c: bool = False,
    heavy_atom_cutoff: float = DEFAULT_HEAVY_ATOM_CUTOFF,
    pd_cutoff: float = DEFAULT_PD_CUTOFF,
) -> ClashScore:
    """Score deterministic heavy-atom overlaps within or between atom arrays."""
    pairs = _clashing_pairs(
        atom_array=atom_array,
        other=other,
        bonded_atom_pairs=bonded_atom_pairs,
        ignored_atom_name_pairs=ignored_atom_name_pairs,
        ignored_atom_index_pairs=ignored_atom_index_pairs,
        ignore_same_residue=ignore_same_residue,
        ignore_inter_residue_backbone_n_c=ignore_inter_residue_backbone_n_c,
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
    ignored_atom_index_pairs: Iterable[tuple[int, int]],
    ignore_same_residue: bool,
    ignore_inter_residue_backbone_n_c: bool,
    heavy_atom_cutoff: float,
    pd_cutoff: float,
) -> list[ClashPair]:
    bonded_pairs = {frozenset(pair) for pair in bonded_atom_pairs}
    ignored_pairs = {frozenset(pair) for pair in ignored_atom_name_pairs}
    ignored_within_indices = {frozenset(pair) for pair in ignored_atom_index_pairs}
    ignored_between_indices = set(ignored_atom_index_pairs)

    if other is None:
        heavy_indices = _heavy_atom_indices(atom_array)
        if heavy_indices.size < 2:
            return []
        left_offsets, right_offsets = np.triu_indices(heavy_indices.size, k=1)
        indices_1 = heavy_indices[left_offsets]
        indices_2 = heavy_indices[right_offsets]
        keep = _within_pair_keep_mask(
            atom_array=atom_array,
            indices_1=indices_1,
            indices_2=indices_2,
            bonded_pairs=bonded_pairs,
            ignored_pairs=ignored_pairs,
            ignored_within_indices=ignored_within_indices,
            ignore_same_residue=ignore_same_residue,
            ignore_inter_residue_backbone_n_c=ignore_inter_residue_backbone_n_c,
        )
        return _clash_pairs_from_indices(
            atom_array=atom_array,
            indices_1=indices_1[keep],
            other=atom_array,
            indices_2=indices_2[keep],
            heavy_atom_cutoff=heavy_atom_cutoff,
            pd_cutoff=pd_cutoff,
        )

    heavy_indices_1 = _heavy_atom_indices(atom_array)
    heavy_indices_2 = _heavy_atom_indices(other)
    if heavy_indices_1.size == 0 or heavy_indices_2.size == 0:
        return []
    indices_1 = np.repeat(heavy_indices_1, heavy_indices_2.size)
    indices_2 = np.tile(heavy_indices_2, heavy_indices_1.size)
    keep = _between_pair_keep_mask(
        atom_array=atom_array,
        indices_1=indices_1,
        other=other,
        indices_2=indices_2,
        ignored_pairs=ignored_pairs,
        ignored_between_indices=ignored_between_indices,
        ignore_same_residue=ignore_same_residue,
        ignore_inter_residue_backbone_n_c=ignore_inter_residue_backbone_n_c,
    )
    return _clash_pairs_from_indices(
        atom_array=atom_array,
        indices_1=indices_1[keep],
        other=other,
        indices_2=indices_2[keep],
        heavy_atom_cutoff=heavy_atom_cutoff,
        pd_cutoff=pd_cutoff,
    )


def _within_pair_keep_mask(
    atom_array: struc.AtomArray,
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    bonded_pairs: set[frozenset[str]],
    ignored_pairs: set[frozenset[str]],
    ignored_within_indices: set[frozenset[int]],
    ignore_same_residue: bool,
    ignore_inter_residue_backbone_n_c: bool,
) -> np.ndarray:
    keep = np.ones(indices_1.size, dtype=bool)
    if ignored_within_indices:
        keep &= ~_within_index_pair_mask(indices_1, indices_2, ignored_within_indices)
    if ignore_same_residue:
        keep &= ~_same_residue_mask(atom_array, indices_1, atom_array, indices_2)
    if ignore_inter_residue_backbone_n_c:
        keep &= ~_inter_residue_backbone_n_c_mask(
            atom_array,
            indices_1,
            atom_array,
            indices_2,
        )
    atom_names_1 = atom_array.atom_name[indices_1].astype("U6")
    atom_names_2 = atom_array.atom_name[indices_2].astype("U6")
    if bonded_pairs:
        keep &= ~_name_pair_mask(atom_names_1, atom_names_2, bonded_pairs)
    if ignored_pairs:
        keep &= ~_name_pair_mask(atom_names_1, atom_names_2, ignored_pairs)
    return keep


def _between_pair_keep_mask(
    atom_array: struc.AtomArray,
    indices_1: np.ndarray,
    other: struc.AtomArray,
    indices_2: np.ndarray,
    ignored_pairs: set[frozenset[str]],
    ignored_between_indices: set[tuple[int, int]],
    ignore_same_residue: bool,
    ignore_inter_residue_backbone_n_c: bool,
) -> np.ndarray:
    keep = np.ones(indices_1.size, dtype=bool)
    if ignored_between_indices:
        keep &= ~_between_index_pair_mask(
            indices_1,
            indices_2,
            ignored_between_indices,
        )
    if ignore_same_residue:
        keep &= ~_same_residue_mask(atom_array, indices_1, other, indices_2)
    if ignore_inter_residue_backbone_n_c:
        keep &= ~_inter_residue_backbone_n_c_mask(
            atom_array,
            indices_1,
            other,
            indices_2,
        )
    if ignored_pairs:
        keep &= ~_name_pair_mask(
            atom_array.atom_name[indices_1].astype("U6"),
            other.atom_name[indices_2].astype("U6"),
            ignored_pairs,
        )
    return keep


def _clash_pairs_from_indices(
    atom_array: struc.AtomArray,
    indices_1: np.ndarray,
    other: struc.AtomArray,
    indices_2: np.ndarray,
    heavy_atom_cutoff: float,
    pd_cutoff: float,
) -> list[ClashPair]:
    if indices_1.size == 0:
        return []

    distances = np.linalg.norm(
        atom_array.coord[indices_1] - other.coord[indices_2],
        axis=1,
    )
    cutoffs = np.full(indices_1.size, heavy_atom_cutoff, dtype=float)
    cutoffs[
        np.isin(atom_array.element[indices_1], PD_ELEMENTS)
        | np.isin(other.element[indices_2], PD_ELEMENTS)
    ] = pd_cutoff
    overlaps = cutoffs - distances
    clash_offsets = np.flatnonzero(overlaps > 0.0)
    return [
        ClashPair(
            atom_1=_atom_label(atom_array, int(indices_1[offset])),
            atom_2=_atom_label(other, int(indices_2[offset])),
            distance=float(distances[offset]),
            cutoff=float(cutoffs[offset]),
            overlap=float(overlaps[offset]),
        )
        for offset in clash_offsets
    ]


def _within_index_pair_mask(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    ignored_within_indices: set[frozenset[int]],
) -> np.ndarray:
    mask = np.zeros(indices_1.size, dtype=bool)
    for pair in ignored_within_indices:
        if len(pair) != 2:
            continue
        index_1, index_2 = tuple(pair)
        mask |= ((indices_1 == index_1) & (indices_2 == index_2)) | (
            (indices_1 == index_2) & (indices_2 == index_1)
        )
    return mask


def _between_index_pair_mask(
    indices_1: np.ndarray,
    indices_2: np.ndarray,
    ignored_between_indices: set[tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(indices_1.size, dtype=bool)
    for index_1, index_2 in ignored_between_indices:
        mask |= (indices_1 == index_1) & (indices_2 == index_2)
    return mask


def _name_pair_mask(
    atom_names_1: np.ndarray,
    atom_names_2: np.ndarray,
    name_pairs: set[frozenset[str]],
) -> np.ndarray:
    mask = np.zeros(atom_names_1.size, dtype=bool)
    for pair in name_pairs:
        if len(pair) == 1:
            name = next(iter(pair))
            mask |= (atom_names_1 == name) & (atom_names_2 == name)
            continue
        if len(pair) != 2:
            continue
        name_1, name_2 = tuple(pair)
        mask |= ((atom_names_1 == name_1) & (atom_names_2 == name_2)) | (
            (atom_names_1 == name_2) & (atom_names_2 == name_1)
        )
    return mask


def _same_residue_mask(
    atom_array: struc.AtomArray,
    indices: np.ndarray,
    other: struc.AtomArray,
    other_indices: np.ndarray,
) -> np.ndarray:
    return (
        atom_array.chain_id[indices] == other.chain_id[other_indices]
    ) & (atom_array.res_id[indices] == other.res_id[other_indices])


def _inter_residue_backbone_n_c_mask(
    atom_array: struc.AtomArray,
    indices: np.ndarray,
    other: struc.AtomArray,
    other_indices: np.ndarray,
) -> np.ndarray:
    names_1 = atom_array.atom_name[indices].astype("U6")
    names_2 = other.atom_name[other_indices].astype("U6")
    backbone_n_c = ((names_1 == "N") & (names_2 == "C")) | (
        (names_1 == "C") & (names_2 == "N")
    )
    return backbone_n_c & ~_same_residue_mask(atom_array, indices, other, other_indices)


def _heavy_atom_indices(atom_array: struc.AtomArray) -> np.ndarray:
    return np.flatnonzero(~np.isin(atom_array.element, HYDROGEN_ELEMENTS))


def _atom_label(atom_array: struc.AtomArray, index: int) -> str:
    return (
        f"{atom_array.chain_id[index]}:"
        f"{int(atom_array.res_id[index])}:"
        f"{atom_array.atom_name[index]}"
    )
