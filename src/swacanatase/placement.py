from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np

from tuber.writers import write_structure

from .active_site import build_bp5_palladium_active_site
from .bp5_rotamers import BP5RotamerPlacement, enumerate_bp5_chi_rotamers
from .clashes import score_heavy_atom_clashes
from .ligands import DEFAULT_LIGAND_DIR, load_bp5_bond_pairs
from .nanoring import generate_armchair_nanoring
from .secondary_structure import (
    SecondaryStructureClashScore,
    SecondaryStructureOrientationMetrics,
    SecondaryStructureType,
    SecondaryStructureSegment,
    build_regular_secondary_structure_segment,
    measure_secondary_structure_orientation,
)

DEFAULT_M_VALUES = (18, 24, 30, 36)
DEFAULT_GENERATED_DATA_DIR = Path("data/generated")
DEFAULT_NANORING_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "nanoring"
DEFAULT_THEOZYME_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "theozyme"
DEFAULT_ROTAMER_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "rotamers"
DEFAULT_SECONDARY_STRUCTURE_OUTPUT_DIR = (
    DEFAULT_GENERATED_DATA_DIR / "secondary_structure"
)
RING_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
BP5_VIRTUAL_CARBON_ATOMS = frozenset({"CV1", "CV2"})
BP5_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


@dataclass(frozen=True)
class NanoringAnchorPair:
    """Two central-band scaffold carbons used as a BP5 virtual-carbon target."""

    atom_indices: tuple[int, int]
    coordinates: np.ndarray
    midpoint: np.ndarray
    ring_axis: np.ndarray
    radial_direction: np.ndarray
    tangential_direction: np.ndarray
    angular_midpoint_degrees: float
    anchor_distance: float


@dataclass(frozen=True)
class BP5NanoringPlacement:
    """A scaffold, selected anchor pairs, and the placed BP5 sidechain model."""

    m: int
    nanoring: struc.AtomArray
    sidechains: struc.AtomArray
    complex: struc.AtomArray
    anchor_pairs: tuple[NanoringAnchorPair, ...]


@dataclass(frozen=True)
class BP5SecondaryStructurePlacement:
    """A grown protein segment candidate with full-state clash components."""

    rotamer_candidate: BP5RotamerPlacement
    segment: SecondaryStructureSegment
    orientation_metrics: SecondaryStructureOrientationMetrics
    scaffold_clash_score: float
    bp5_clash_score: float
    neighboring_backbone_clash_score: float
    clash_score: float

    @property
    def secondary_structure_direction(self) -> np.ndarray:
        return self.orientation_metrics.secondary_structure_direction

    @property
    def radial_alignment(self) -> float:
        return self.orientation_metrics.radial_alignment

    @property
    def tangential_alignment(self) -> float:
        return self.orientation_metrics.tangential_alignment

    @property
    def axial_alignment(self) -> float:
        return self.orientation_metrics.axial_alignment

    @property
    def n_terminal_exit_vector(self) -> np.ndarray:
        return self.orientation_metrics.n_terminal_exit_vector

    @property
    def c_terminal_exit_vector(self) -> np.ndarray:
        return self.orientation_metrics.c_terminal_exit_vector


@dataclass(frozen=True)
class BP5SymmetricRotamerState:
    """One BP5 chi state represented at every selected nanoring anchor."""

    rotamer_name: str
    candidates: tuple[BP5RotamerPlacement, ...]
    clash_score: float

    @property
    def sidechains(self) -> struc.AtomArray:
        return struc.concatenate([candidate.atom_array for candidate in self.candidates])


@dataclass(frozen=True)
class BP5SymmetricSecondaryStructureState:
    """One symmetric secondary-structure state represented at every anchor."""

    rotamer_name: str
    candidates: tuple[BP5SecondaryStructurePlacement, ...]
    scaffold_clash_score: float
    bp5_clash_score: float
    neighboring_backbone_clash_score: float
    clash_score: float

    @property
    def segments(self) -> struc.AtomArray:
        return struc.concatenate(
            [candidate.segment.atom_array for candidate in self.candidates]
        )


@dataclass(frozen=True)
class BP5NanoringRotamerPlacement:
    """Rigid BP5/Pd placement plus full-state rotamer/segment candidates."""

    m: int
    nanoring: struc.AtomArray
    rigid_sidechains: struc.AtomArray
    anchor_pairs: tuple[NanoringAnchorPair, ...]
    rotamer_candidates: tuple[BP5RotamerPlacement, ...]
    accepted_rotamer_candidates: tuple[BP5RotamerPlacement, ...]
    secondary_structure_candidates: tuple[BP5SecondaryStructurePlacement, ...] = ()
    accepted_secondary_structure_candidates: tuple[
        BP5SecondaryStructurePlacement, ...
    ] = ()
    rotamer_states: tuple[BP5SymmetricRotamerState, ...] = ()
    accepted_rotamer_states: tuple[BP5SymmetricRotamerState, ...] = ()
    secondary_structure_states: tuple[BP5SymmetricSecondaryStructureState, ...] = ()
    accepted_secondary_structure_states: tuple[
        BP5SymmetricSecondaryStructureState, ...
    ] = ()


def generate_m_equals_n_nanoring(
    m: int,
    units: float | int = 1.5,
    hydrogen_terminate: bool = False,
    center_z: bool = True,
) -> struc.AtomArray:
    """Generate a first-pass M=N nanoring as an armchair ``(m,m)`` scaffold."""
    return generate_armchair_nanoring(
        n=m,
        units=units,
        hydrogen_terminate=hydrogen_terminate,
        center_z=center_z,
    )


def generate_m_equals_n_nanorings(
    m_values: Iterable[int] = DEFAULT_M_VALUES,
    units: float | int = 1.5,
    hydrogen_terminate: bool = False,
    center_z: bool = True,
) -> dict[int, struc.AtomArray]:
    """Generate multiple first-pass M=N nanoring scaffolds."""
    return {
        m: generate_m_equals_n_nanoring(
            m=m,
            units=units,
            hydrogen_terminate=hydrogen_terminate,
            center_z=center_z,
        )
        for m in m_values
    }


def central_para_linker_anchor_pairs(
    nanoring: struc.AtomArray,
    count: int | None = None,
    z_band: str = "lower",
    phase_offset: int = 1,
) -> tuple[NanoringAnchorPair, ...]:
    """Select central-band para-like carbon pairs for BP5 virtual-carbon anchors.

    In an armchair ``(m,m)`` unit, each axial band contains ``2*m`` carbons whose
    cyclic nearest-neighbor gaps alternate between short and long chords.  The
    longer same-band chords are the para-like pair class.  This is useful for
    comparing against same-pseudo-benzene placement, but the BP5 placement
    default uses :func:`central_inter_benzene_linker_anchor_pairs`.
    """
    return _central_band_anchor_pairs(
        nanoring=nanoring,
        count=count,
        z_band=z_band,
        phase_offset=phase_offset,
        pair_distance_class="long",
    )


def central_inter_benzene_linker_anchor_pairs(
    nanoring: struc.AtomArray,
    count: int | None = None,
    z_band: str = "lower",
    phase_offset: int = 1,
) -> tuple[NanoringAnchorPair, ...]:
    """Select central-band adjacent carbon pairs that link pseudo-benzenes."""
    return _central_band_anchor_pairs(
        nanoring=nanoring,
        count=count,
        z_band=z_band,
        phase_offset=phase_offset,
        pair_distance_class="short",
    )


def _central_band_anchor_pairs(
    nanoring: struc.AtomArray,
    count: int | None,
    z_band: str,
    phase_offset: int,
    pair_distance_class: str,
) -> tuple[NanoringAnchorPair, ...]:
    if pair_distance_class not in {"short", "long"}:
        raise ValueError("pair_distance_class must be 'short' or 'long'")

    carbon_indices = np.flatnonzero(np.char.upper(nanoring.element.astype("U2")) == "C")
    if carbon_indices.size == 0:
        raise ValueError("nanoring must contain carbon atoms")

    band_indices = _central_z_band_indices(nanoring, carbon_indices, z_band=z_band)
    if band_indices.size < 4:
        raise ValueError("central z band must contain at least 4 carbons")

    sorted_indices = _sort_indices_by_angle(nanoring.coord[band_indices], band_indices)
    adjacent_pairs = _cyclic_adjacent_pairs(sorted_indices)
    distances = np.array(
        [
            np.linalg.norm(nanoring.coord[index_1] - nanoring.coord[index_2])
            for index_1, index_2 in adjacent_pairs
        ],
        dtype=float,
    )
    distance_cutoff = float(np.median(distances))
    linker_pairs = [
        pair for pair, distance in zip(adjacent_pairs, distances, strict=True)
        if (
            distance > distance_cutoff
            if pair_distance_class == "long"
            else distance <= distance_cutoff
        )
    ]
    if not linker_pairs:
        raise ValueError(
            f"failed to identify {pair_distance_class} same-band carbon pairs"
        )

    anchors = tuple(
        sorted(
            (_build_anchor_pair(nanoring, pair) for pair in linker_pairs),
            key=lambda anchor: anchor.angular_midpoint_degrees,
        )
    )
    if count is None:
        return anchors

    if count < 1:
        raise ValueError("count must be at least 1")
    if len(anchors) % count != 0:
        raise ValueError(
            f"cannot select {count} evenly spaced pairs from {len(anchors)} anchors"
        )
    step = len(anchors) // count
    if phase_offset < 0:
        raise ValueError("phase_offset must be non-negative")
    normalized_phase_offset = phase_offset % step
    selected_anchors = [
        anchors[(normalized_phase_offset + step * index) % len(anchors)]
        for index in range(count)
    ]
    return tuple(
        sorted(
            selected_anchors,
            key=lambda anchor: anchor.angular_midpoint_degrees,
        )
    )


def place_bp5_sidechains_around_nanoring(
    m: int,
    units: float | int = 1.5,
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
    coordinate_set: str = "ideal",
    z_band: str = "lower",
    anchor_phase_offset: int = 1,
    sidechain_direction: str = "outward",
    snap_virtual_carbons: bool = False,
) -> BP5NanoringPlacement:
    """Generate an M=N nanoring and place ``m/2`` BP5 sidechains around it."""
    if m % 2 != 0:
        raise ValueError("m must be even to place m/2 evenly spaced BP5 sidechains")

    nanoring = generate_m_equals_n_nanoring(m=m, units=units)
    anchor_pairs = central_inter_benzene_linker_anchor_pairs(
        nanoring,
        count=m // 2,
        z_band=z_band,
        phase_offset=anchor_phase_offset,
    )
    bp5 = build_bp5_palladium_active_site(
        cif_path=cif_path,
        coordinate_set=coordinate_set,
    )
    sidechains = [
        _place_single_bp5_sidechain(
            bp5,
            anchor,
            residue_id=residue_id,
            starting_atom_id=nanoring.array_length()
            + 1
            + (residue_id - 1) * bp5.array_length(),
            sidechain_direction=sidechain_direction,
            snap_virtual_carbons=snap_virtual_carbons,
        )
        for residue_id, anchor in enumerate(anchor_pairs, start=1)
    ]
    nanoring = _with_chain_id(nanoring, chain_id="B")
    sidechain_array = struc.concatenate(sidechains)
    return BP5NanoringPlacement(
        m=m,
        nanoring=nanoring,
        sidechains=sidechain_array,
        complex=struc.concatenate([nanoring, sidechain_array]),
        anchor_pairs=anchor_pairs,
    )


def place_bp5_rotamer_ensembles_around_nanoring(
    m: int,
    units: float | int = 1.5,
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
    coordinate_set: str = "ideal",
    z_band: str = "lower",
    anchor_phase_offset: int = 1,
    sidechain_direction: str = "outward",
    snap_virtual_carbons: bool = False,
    max_rotamers_per_site: int | None = None,
    rotamer_clash_cutoff: float | None = None,
    secondary_structure: SecondaryStructureType | None = None,
    residues_before: int = 0,
    residues_after: int = 0,
    secondary_structure_clash_cutoff: float | None = None,
) -> BP5NanoringRotamerPlacement:
    """Place BP5/Pd sidechains and enumerate fixed-active-site chi rotamers."""
    if max_rotamers_per_site is not None and max_rotamers_per_site < 1:
        raise ValueError("max_rotamers_per_site must be at least 1")
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if secondary_structure not in {None, "alpha_helix", "beta_strand"}:
        raise ValueError(
            "secondary_structure must be None, 'alpha_helix', or 'beta_strand'"
        )

    rigid_placement = place_bp5_sidechains_around_nanoring(
        m=m,
        units=units,
        cif_path=cif_path,
        coordinate_set=coordinate_set,
        z_band=z_band,
        anchor_phase_offset=anchor_phase_offset,
        sidechain_direction=sidechain_direction,
        snap_virtual_carbons=snap_virtual_carbons,
    )
    bond_pairs = load_bp5_bond_pairs(cif_path=cif_path)

    candidates: list[BP5RotamerPlacement] = []
    for residue_id in range(1, len(rigid_placement.anchor_pairs) + 1):
        residue = rigid_placement.sidechains[
            rigid_placement.sidechains.res_id == residue_id
        ]
        residue_candidates = enumerate_bp5_chi_rotamers(
            atom_array=residue,
            bond_pairs=bond_pairs,
            residue_id=residue_id,
        )
        candidates.extend(residue_candidates)

    rotamer_states = _build_symmetric_rotamer_states(
        candidates=tuple(candidates),
        nanoring=rigid_placement.nanoring,
        bond_pairs=bond_pairs,
        residue_count=len(rigid_placement.anchor_pairs),
    )
    accepted_rotamer_states = _select_symmetric_rotamer_states(
        states=rotamer_states,
        rotamer_clash_cutoff=rotamer_clash_cutoff,
        max_rotamer_states=max_rotamers_per_site,
    )
    scored_candidates = _flatten_rotamer_states(rotamer_states)
    accepted = _flatten_rotamer_states(accepted_rotamer_states)

    secondary_candidates: tuple[BP5SecondaryStructurePlacement, ...] = ()
    accepted_secondary_candidates: tuple[BP5SecondaryStructurePlacement, ...] = ()
    secondary_states: tuple[BP5SymmetricSecondaryStructureState, ...] = ()
    accepted_secondary_states: tuple[BP5SymmetricSecondaryStructureState, ...] = ()
    if secondary_structure is not None:
        secondary_states = _build_symmetric_secondary_structure_states(
            rotamer_states=accepted_rotamer_states,
            nanoring=rigid_placement.nanoring,
            bond_pairs=bond_pairs,
            anchor_pairs=rigid_placement.anchor_pairs,
            secondary_structure=secondary_structure,
            residues_before=residues_before,
            residues_after=residues_after,
            starting_atom_id=rigid_placement.nanoring.array_length() + 1,
        )
        accepted_secondary_states = _select_symmetric_secondary_states(
            states=secondary_states,
            clash_cutoff=secondary_structure_clash_cutoff,
        )
        secondary_candidates = _flatten_secondary_structure_states(secondary_states)
        accepted_secondary_candidates = _flatten_secondary_structure_states(
            accepted_secondary_states
        )

    return BP5NanoringRotamerPlacement(
        m=m,
        nanoring=rigid_placement.nanoring,
        rigid_sidechains=rigid_placement.sidechains,
        anchor_pairs=rigid_placement.anchor_pairs,
        rotamer_candidates=scored_candidates,
        accepted_rotamer_candidates=accepted,
        secondary_structure_candidates=secondary_candidates,
        accepted_secondary_structure_candidates=accepted_secondary_candidates,
        rotamer_states=rotamer_states,
        accepted_rotamer_states=accepted_rotamer_states,
        secondary_structure_states=secondary_states,
        accepted_secondary_structure_states=accepted_secondary_states,
    )


def write_bp5_nanoring_series(
    output_dir: str | Path = DEFAULT_GENERATED_DATA_DIR,
    m_values: Iterable[int] = DEFAULT_M_VALUES,
    units: float | int = 1.5,
    anchor_phase_offset: int = 1,
    snap_virtual_carbons: bool = False,
    file_format: str = "cif",
    overwrite: bool = False,
    enumerate_bp5_rotamers: bool = False,
    max_rotamers_per_site: int | None = None,
    rotamer_clash_cutoff: float | None = None,
    secondary_structure: SecondaryStructureType | None = None,
    residues_before: int = 0,
    residues_after: int = 0,
    secondary_structure_clash_cutoff: float | None = None,
) -> list[Path]:
    """Write nanoring-only and BP5-placed structures for each requested M value."""
    if max_rotamers_per_site is not None and max_rotamers_per_site < 1:
        raise ValueError("max_rotamers_per_site must be at least 1")
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if secondary_structure not in {None, "alpha_helix", "beta_strand"}:
        raise ValueError(
            "secondary_structure must be None, 'alpha_helix', or 'beta_strand'"
        )

    output_dir = Path(output_dir)
    nanoring_output_dir = output_dir / "nanoring"
    theozyme_output_dir = output_dir / "theozyme"
    rotamer_output_dir = output_dir / "rotamers"
    secondary_structure_output_dir = output_dir / "secondary_structure"
    nanoring_output_dir.mkdir(parents=True, exist_ok=True)
    theozyme_output_dir.mkdir(parents=True, exist_ok=True)
    if enumerate_bp5_rotamers:
        rotamer_output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_structure is not None:
        secondary_structure_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for m in m_values:
        placement = place_bp5_sidechains_around_nanoring(
            m=m,
            units=units,
            anchor_phase_offset=anchor_phase_offset,
            snap_virtual_carbons=snap_virtual_carbons,
        )
        ring_path = nanoring_output_dir / f"nanoring_M{m}.{file_format}"
        complex_path = theozyme_output_dir / f"nanoring_M{m}_bp5.{file_format}"
        written_paths.append(
            write_structure(
                atom_array=placement.nanoring,
                output_path=ring_path,
                file_format=file_format,
                overwrite=overwrite,
            )
        )
        written_paths.append(
            write_structure(
                atom_array=placement.complex,
                output_path=complex_path,
                file_format=file_format,
                overwrite=overwrite,
            )
        )

        if not enumerate_bp5_rotamers and secondary_structure is None:
            continue

        rotamer_placement = place_bp5_rotamer_ensembles_around_nanoring(
            m=m,
            units=units,
            anchor_phase_offset=anchor_phase_offset,
            snap_virtual_carbons=snap_virtual_carbons,
            max_rotamers_per_site=max_rotamers_per_site,
            rotamer_clash_cutoff=rotamer_clash_cutoff,
            secondary_structure=secondary_structure,
            residues_before=residues_before,
            residues_after=residues_after,
            secondary_structure_clash_cutoff=secondary_structure_clash_cutoff,
        )
        if enumerate_bp5_rotamers:
            for state in rotamer_placement.accepted_rotamer_states:
                rotamer_path = (
                    rotamer_output_dir
                    / f"nanoring_M{m}_{state.rotamer_name}.{file_format}"
                )
                written_paths.append(
                    write_structure(
                        atom_array=struc.concatenate(
                            [rotamer_placement.nanoring, state.sidechains]
                        ),
                        output_path=rotamer_path,
                        file_format=file_format,
                        overwrite=overwrite,
                    )
                )

        for state in rotamer_placement.accepted_secondary_structure_states:
            segment_path = (
                secondary_structure_output_dir
                / (
                    f"nanoring_M{m}_{state.rotamer_name}_{secondary_structure}_"
                    f"pre{residues_before}_post{residues_after}.{file_format}"
                )
            )
            written_paths.append(
                write_structure(
                    atom_array=struc.concatenate(
                        [rotamer_placement.nanoring, state.segments]
                    ),
                    output_path=segment_path,
                    file_format=file_format,
                    overwrite=overwrite,
                )
            )
    return written_paths


@dataclass(frozen=True)
class _BP5StateClashScore:
    scaffold_score: float
    bp5_score: float

    @property
    def total_overlap_score(self) -> float:
        return float(self.scaffold_score + self.bp5_score)


def _build_symmetric_rotamer_states(
    candidates: tuple[BP5RotamerPlacement, ...],
    nanoring: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
    residue_count: int,
) -> tuple[BP5SymmetricRotamerState, ...]:
    groups: defaultdict[str, list[BP5RotamerPlacement]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.rotamer.name].append(candidate)

    states: list[BP5SymmetricRotamerState] = []
    for rotamer_name, group in groups.items():
        ordered_group = _complete_symmetric_rotamer_group(group, residue_count)
        score = _score_bp5_atom_group(
            atom_arrays=tuple(candidate.atom_array for candidate in ordered_group),
            nanoring=nanoring,
            bond_pairs=bond_pairs,
        )
        scored_group = tuple(
            replace(candidate, clash_score=score.total_overlap_score)
            for candidate in ordered_group
        )
        states.append(
            BP5SymmetricRotamerState(
                rotamer_name=rotamer_name,
                candidates=scored_group,
                clash_score=score.total_overlap_score,
            )
        )
    return tuple(states)


def _select_symmetric_rotamer_states(
    states: tuple[BP5SymmetricRotamerState, ...],
    rotamer_clash_cutoff: float | None,
    max_rotamer_states: int | None,
) -> tuple[BP5SymmetricRotamerState, ...]:
    accepted_groups = [
        state
        for state in states
        if rotamer_clash_cutoff is None or state.clash_score <= rotamer_clash_cutoff
    ]
    accepted_groups = sorted(accepted_groups, key=_rotamer_state_score_key)
    if max_rotamer_states is not None:
        accepted_groups = accepted_groups[:max_rotamer_states]
    return tuple(accepted_groups)


def _flatten_rotamer_states(
    states: tuple[BP5SymmetricRotamerState, ...],
) -> tuple[BP5RotamerPlacement, ...]:
    state_rank = {state.rotamer_name: rank for rank, state in enumerate(states)}
    return tuple(
        sorted(
            (
                candidate
                for state in states
                for candidate in state.candidates
            ),
            key=lambda candidate: (
                candidate.residue_id,
                state_rank[candidate.rotamer.name],
            ),
        )
    )


def _complete_symmetric_rotamer_group(
    group: Iterable[BP5RotamerPlacement],
    residue_count: int,
) -> tuple[BP5RotamerPlacement, ...]:
    ordered_group = tuple(sorted(group, key=lambda candidate: candidate.residue_id))
    residue_ids = tuple(candidate.residue_id for candidate in ordered_group)
    expected_residue_ids = tuple(range(1, residue_count + 1))
    if residue_ids != expected_residue_ids:
        rotamer_name = ordered_group[0].rotamer.name if ordered_group else "<empty>"
        raise ValueError(
            f"rotamer state {rotamer_name!r} must be represented at every BP5 site"
        )
    return ordered_group


def _rotamer_state_score_key(
    state: BP5SymmetricRotamerState,
) -> tuple[float, str]:
    return (state.clash_score, state.rotamer_name)


def _select_symmetric_secondary_states(
    states: tuple[BP5SymmetricSecondaryStructureState, ...],
    clash_cutoff: float | None,
) -> tuple[BP5SymmetricSecondaryStructureState, ...]:
    accepted_groups = [
        state
        for state in states
        if clash_cutoff is None or state.clash_score <= clash_cutoff
    ]
    return tuple(sorted(accepted_groups, key=_secondary_state_score_key))


def _flatten_secondary_structure_states(
    states: tuple[BP5SymmetricSecondaryStructureState, ...],
) -> tuple[BP5SecondaryStructurePlacement, ...]:
    state_rank = {state.rotamer_name: rank for rank, state in enumerate(states)}
    return tuple(
        sorted(
            (
                candidate
                for state in states
                for candidate in state.candidates
            ),
            key=lambda candidate: (
                candidate.rotamer_candidate.residue_id,
                state_rank[candidate.rotamer_candidate.rotamer.name],
            ),
        )
    )


def _secondary_state_score_key(
    state: BP5SymmetricSecondaryStructureState,
) -> tuple[float, str]:
    return (state.clash_score, state.rotamer_name)


def _build_symmetric_secondary_structure_states(
    rotamer_states: tuple[BP5SymmetricRotamerState, ...],
    nanoring: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
    anchor_pairs: tuple[NanoringAnchorPair, ...],
    secondary_structure: SecondaryStructureType,
    residues_before: int,
    residues_after: int,
    starting_atom_id: int,
) -> tuple[BP5SymmetricSecondaryStructureState, ...]:
    segment_span = residues_before + 1 + residues_after
    states: list[BP5SymmetricSecondaryStructureState] = []
    for rotamer_state in rotamer_states:
        segment_pairs: list[tuple[BP5RotamerPlacement, SecondaryStructureSegment]] = []
        next_atom_id = starting_atom_id
        for candidate in rotamer_state.candidates:
            segment = build_regular_secondary_structure_segment(
                bp5_rotamer=candidate,
                secondary_structure_type=secondary_structure,
                residues_before=residues_before,
                residues_after=residues_after,
                starting_residue_id=1 + (candidate.residue_id - 1) * segment_span,
                starting_atom_id=next_atom_id,
            )
            next_atom_id += segment.atom_array.array_length()
            segment_pairs.append((candidate, segment))

        symmetric_segment_pairs = tuple(
            sorted(segment_pairs, key=lambda pair: pair[0].residue_id)
        )
        score = _score_symmetric_secondary_structure_state(
            segment_pairs=symmetric_segment_pairs,
            nanoring=nanoring,
            bond_pairs=bond_pairs,
        )
        scored_candidates: list[BP5SecondaryStructurePlacement] = []
        for candidate, segment in symmetric_segment_pairs:
            anchor_pair = anchor_pairs[candidate.residue_id - 1]
            orientation_metrics = measure_secondary_structure_orientation(
                segment=segment,
                radial_direction=anchor_pair.radial_direction,
                tangential_direction=anchor_pair.tangential_direction,
                ring_axis=anchor_pair.ring_axis,
            )
            scored_candidates.append(
                BP5SecondaryStructurePlacement(
                    rotamer_candidate=candidate,
                    segment=segment,
                    orientation_metrics=orientation_metrics,
                    scaffold_clash_score=score.scaffold_score,
                    bp5_clash_score=score.bp5_score,
                    neighboring_backbone_clash_score=score.neighboring_backbone_score,
                    clash_score=score.total_overlap_score,
                )
            )
        states.append(
            BP5SymmetricSecondaryStructureState(
                rotamer_name=rotamer_state.rotamer_name,
                candidates=tuple(scored_candidates),
                scaffold_clash_score=score.scaffold_score,
                bp5_clash_score=score.bp5_score,
                neighboring_backbone_clash_score=score.neighboring_backbone_score,
                clash_score=score.total_overlap_score,
            ),
        )
    return tuple(states)


def _score_bp5_atom_group(
    atom_arrays: tuple[struc.AtomArray, ...],
    nanoring: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
) -> _BP5StateClashScore:
    bp5_arrays = tuple(
        _without_atom_names(atom_array, BP5_VIRTUAL_CARBON_ATOMS)
        for atom_array in atom_arrays
    )
    scaffold_score = _score_atom_arrays_against_other(
        atom_arrays=bp5_arrays,
        other=nanoring,
    )
    intra_bp5_score = float(
        sum(
            score_heavy_atom_clashes(
                atom_array=atom_array,
                bonded_atom_pairs=bond_pairs,
            ).total_overlap_score
            for atom_array in bp5_arrays
        )
    )
    inter_bp5_score = _score_atom_array_pairs(bp5_arrays)
    return _BP5StateClashScore(
        scaffold_score=scaffold_score,
        bp5_score=float(intra_bp5_score + inter_bp5_score),
    )


def _score_symmetric_secondary_structure_state(
    segment_pairs: tuple[tuple[BP5RotamerPlacement, SecondaryStructureSegment], ...],
    nanoring: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
) -> SecondaryStructureClashScore:
    segments = tuple(segment for _, segment in segment_pairs)
    bp5_arrays = tuple(_bp5_segment_atoms(segment) for segment in segments)
    generated_arrays = tuple(_generated_segment_atoms(segment) for segment in segments)
    bp5_score = _score_bp5_atom_group(
        atom_arrays=bp5_arrays,
        nanoring=nanoring,
        bond_pairs=bond_pairs,
    )

    scaffold_score = float(
        bp5_score.scaffold_score
        + _score_atom_arrays_against_other(
            atom_arrays=generated_arrays,
            other=nanoring,
        )
    )
    own_generated_bp5_score = float(
        sum(
            score_heavy_atom_clashes(
                atom_array=generated_atoms,
                other=_bp5_segment_non_backbone_atoms(segment),
            ).total_overlap_score
            for generated_atoms, segment in zip(
                generated_arrays,
                segments,
                strict=True,
            )
        )
    )
    neighboring_generated_bp5_score = 0.0
    for generated_index, generated_atoms in enumerate(generated_arrays):
        for bp5_index, bp5_atoms in enumerate(bp5_arrays):
            if generated_index == bp5_index:
                continue
            neighboring_generated_bp5_score += score_heavy_atom_clashes(
                atom_array=generated_atoms,
                other=bp5_atoms,
            ).total_overlap_score

    return SecondaryStructureClashScore(
        scaffold_score=scaffold_score,
        bp5_score=float(
            bp5_score.bp5_score
            + own_generated_bp5_score
            + neighboring_generated_bp5_score
        ),
        neighboring_backbone_score=_score_atom_array_pairs(generated_arrays),
    )


def _score_atom_arrays_against_other(
    atom_arrays: tuple[struc.AtomArray, ...],
    other: struc.AtomArray,
) -> float:
    return float(
        sum(
            score_heavy_atom_clashes(
                atom_array=atom_array,
                other=other,
            ).total_overlap_score
            for atom_array in atom_arrays
        )
    )


def _score_atom_array_pairs(atom_arrays: tuple[struc.AtomArray, ...]) -> float:
    score = 0.0
    for left_index, left in enumerate(atom_arrays[:-1]):
        for right in atom_arrays[left_index + 1 :]:
            score += score_heavy_atom_clashes(
                atom_array=left,
                other=right,
            ).total_overlap_score
    return float(score)


def _bp5_segment_atoms(segment: SecondaryStructureSegment) -> struc.AtomArray:
    return _without_atom_names(
        segment.atom_array[segment.atom_array.res_id == segment.bp5_residue_id],
        BP5_VIRTUAL_CARBON_ATOMS,
    )


def _bp5_segment_non_backbone_atoms(
    segment: SecondaryStructureSegment,
) -> struc.AtomArray:
    return _without_atom_names(
        _bp5_segment_atoms(segment),
        BP5_BACKBONE_ATOMS,
    )


def _generated_segment_atoms(segment: SecondaryStructureSegment) -> struc.AtomArray:
    return segment.atom_array[segment.atom_array.res_id != segment.bp5_residue_id]


def _without_atom_names(
    atom_array: struc.AtomArray,
    atom_names: frozenset[str],
) -> struc.AtomArray:
    return atom_array[~np.isin(atom_array.atom_name, list(atom_names))]


def _place_single_bp5_sidechain(
    bp5: struc.AtomArray,
    anchor: NanoringAnchorPair,
    residue_id: int,
    starting_atom_id: int,
    sidechain_direction: str,
    snap_virtual_carbons: bool,
) -> struc.AtomArray:
    if sidechain_direction not in {"outward", "inward"}:
        raise ValueError("sidechain_direction must be 'outward' or 'inward'")

    placed = bp5.copy()
    name_to_index = _atom_indices(placed)
    cv1 = placed.coord[name_to_index["CV1"]]
    cv2 = placed.coord[name_to_index["CV2"]]
    pd = placed.coord[name_to_index["PD"]]
    source_midpoint = (cv1 + cv2) / 2.0
    source_frame = _orthonormal_frame(
        primary=cv2 - cv1,
        secondary=pd - source_midpoint,
    )

    radial_direction = anchor.radial_direction
    if sidechain_direction == "inward":
        radial_direction = -radial_direction
    target_frame = _orthonormal_frame(
        primary=anchor.coordinates[1] - anchor.coordinates[0],
        secondary=radial_direction,
    )
    rotation = target_frame @ source_frame.T
    placed.coord = (placed.coord - source_midpoint) @ rotation.T + anchor.midpoint

    if snap_virtual_carbons:
        placed.coord[name_to_index["CV1"]] = anchor.coordinates[0]
        placed.coord[name_to_index["CV2"]] = anchor.coordinates[1]

    placed.chain_id = np.full(placed.array_length(), "A", dtype="U4")
    placed.res_id = np.full(placed.array_length(), residue_id, dtype=int)
    placed.set_annotation(
        "atom_id",
        np.arange(
            starting_atom_id,
            starting_atom_id + placed.array_length(),
            dtype=int,
        ),
    )
    return placed


def _with_chain_id(atom_array: struc.AtomArray, chain_id: str) -> struc.AtomArray:
    copied = atom_array.copy()
    copied.chain_id = np.full(copied.array_length(), chain_id, dtype="U4")
    return copied


def _central_z_band_indices(
    atom_array: struc.AtomArray,
    atom_indices: np.ndarray,
    z_band: str,
) -> np.ndarray:
    if z_band not in {"lower", "upper"}:
        raise ValueError("z_band must be 'lower' or 'upper'")

    z_values = atom_array.coord[atom_indices, 2]
    levels = np.array(sorted({round(float(z), 6) for z in z_values}), dtype=float)
    z_center = (float(z_values.min()) + float(z_values.max())) / 2.0
    central_distance = np.abs(levels - z_center)
    min_distance = float(central_distance.min())
    central_levels = levels[np.isclose(central_distance, min_distance, atol=1e-6)]
    selected_level = float(
        central_levels.min() if z_band == "lower" else central_levels.max()
    )
    return atom_indices[np.isclose(z_values, selected_level, atol=1e-5)]


def _sort_indices_by_angle(coordinates: np.ndarray, indices: np.ndarray) -> np.ndarray:
    angles = np.mod(np.arctan2(coordinates[:, 1], coordinates[:, 0]), 2.0 * np.pi)
    return indices[np.argsort(angles)]


def _cyclic_adjacent_pairs(indices: np.ndarray) -> list[tuple[int, int]]:
    return [
        (int(indices[index]), int(indices[(index + 1) % len(indices)]))
        for index in range(len(indices))
    ]


def _build_anchor_pair(
    atom_array: struc.AtomArray,
    atom_indices: tuple[int, int],
) -> NanoringAnchorPair:
    ordered_indices = _order_anchor_indices(atom_array.coord, atom_indices)
    coordinates = np.array(
        [atom_array.coord[index] for index in ordered_indices],
        dtype=float,
    )
    midpoint = coordinates.mean(axis=0)
    radial_direction = _radial_direction(midpoint)
    tangential_direction = _unit(coordinates[1] - coordinates[0])
    return NanoringAnchorPair(
        atom_indices=ordered_indices,
        coordinates=coordinates,
        midpoint=midpoint,
        ring_axis=RING_AXIS.copy(),
        radial_direction=radial_direction,
        tangential_direction=tangential_direction,
        angular_midpoint_degrees=_angle_degrees(midpoint),
        anchor_distance=float(np.linalg.norm(coordinates[1] - coordinates[0])),
    )


def _order_anchor_indices(
    coordinates: np.ndarray,
    atom_indices: tuple[int, int],
) -> tuple[int, int]:
    index_1, index_2 = atom_indices
    midpoint = (coordinates[index_1] + coordinates[index_2]) / 2.0
    tangent = np.cross(RING_AXIS, _radial_direction(midpoint))
    if np.dot(coordinates[index_2] - coordinates[index_1], tangent) < 0:
        return (index_2, index_1)
    return atom_indices


def _radial_direction(point: np.ndarray) -> np.ndarray:
    radial = np.array([point[0], point[1], 0.0], dtype=float)
    return _unit(radial)


def _angle_degrees(point: np.ndarray) -> float:
    return float(np.degrees(np.mod(np.arctan2(point[1], point[0]), 2.0 * np.pi)))


def _orthonormal_frame(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    x_axis = _unit(primary)
    y_axis = secondary - np.dot(secondary, x_axis) * x_axis
    y_axis = _unit(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _atom_indices(atom_array: struc.AtomArray) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, atom_name in enumerate(atom_array.atom_name.tolist()):
        if atom_name in indices:
            raise ValueError(f"Duplicate atom name {atom_name!r}")
        indices[atom_name] = index
    return indices


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate M=N armchair nanorings and place M/2 BP5 sidechains "
            "around each scaffold."
        )
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=list(DEFAULT_M_VALUES),
        help="M values to generate. Defaults to 18 24 30 36.",
    )
    parser.add_argument(
        "--units",
        type=float,
        default=1.5,
        help="Armchair nanotube units to use for each M=N scaffold.",
    )
    parser.add_argument(
        "--anchor-phase-offset",
        type=int,
        default=1,
        help="Phase offset into the evenly spaced central-band anchor pairs.",
    )
    parser.add_argument(
        "--snap-virtual-carbons",
        action="store_true",
        help="Overwrite CV1/CV2 coordinates onto the matched nanoring carbons.",
    )
    parser.add_argument(
        "--enumerate-bp5-rotamers",
        action="store_true",
        help="Also write accepted BP5 chi-rotamer complexes under rotamers/.",
    )
    parser.add_argument(
        "--max-rotamers-per-site",
        type=int,
        default=None,
        help=(
            "Keep only the top K symmetric rotamer states, with each state "
            "represented at every BP5 site."
        ),
    )
    parser.add_argument(
        "--rotamer-clash-cutoff",
        type=float,
        default=None,
        help="Reject rotamer candidates with clash scores above this value.",
    )
    parser.add_argument(
        "--secondary-structure-clash-cutoff",
        type=float,
        default=None,
        help=(
            "Reject grown secondary-structure candidates with clash scores above "
            "this value."
        ),
    )
    parser.add_argument(
        "--secondary-structure",
        choices=["none", "alpha_helix", "beta_strand"],
        default="none",
        help=(
            "Also write regular secondary-structure segment complexes under "
            "secondary_structure/."
        ),
    )
    parser.add_argument(
        "--residues-before",
        type=int,
        default=0,
        help="Number of residues to grow before BP5 in secondary-structure mode.",
    )
    parser.add_argument(
        "--residues-after",
        type=int,
        default=0,
        help="Number of residues to grow after BP5 in secondary-structure mode.",
    )
    parser.add_argument("--format", choices=["pdb", "cif"], default="cif")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_GENERATED_DATA_DIR,
        help=(
            "Generated-data root. Nanoring-only files are written under "
            "nanoring/ and BP5 complexes under theozyme/."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    written_paths = write_bp5_nanoring_series(
        output_dir=args.output_dir,
        m_values=args.m,
        units=args.units,
        anchor_phase_offset=args.anchor_phase_offset,
        snap_virtual_carbons=args.snap_virtual_carbons,
        file_format=args.format,
        overwrite=args.overwrite,
        enumerate_bp5_rotamers=args.enumerate_bp5_rotamers,
        max_rotamers_per_site=args.max_rotamers_per_site,
        rotamer_clash_cutoff=args.rotamer_clash_cutoff,
        secondary_structure=(
            None if args.secondary_structure == "none" else args.secondary_structure
        ),
        residues_before=args.residues_before,
        residues_after=args.residues_after,
        secondary_structure_clash_cutoff=args.secondary_structure_clash_cutoff,
    )
    for path in written_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
