from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import biotite.structure as struc
import numpy as np

from tuber.writers import write_structure

from .active_site import build_bp5_palladium_active_site
from .bp5_rotamers import (
    BP5RotamerPlacement,
    DEFAULT_BP5_CHI_ROTAMERS,
    enumerate_bp5_chi_rotamers,
)
from .clashes import score_heavy_atom_clashes
from .ligands import DEFAULT_LIGAND_DIR, load_bp5_bond_pairs
from .nanoring import generate_armchair_nanoring
from .secondary_structure import (
    NanoringCylinderIntrusionScore,
    SecondaryStructureClashScore,
    SecondaryStructureOrientationMetrics,
    SecondaryStructureType,
    SecondaryStructureSegment,
    build_regular_secondary_structure_segment,
    measure_secondary_structure_orientation,
    score_nanoring_cylinder_intrusions,
)

DEFAULT_M_VALUES = (18, 24, 30, 36)
DEFAULT_GENERATED_DATA_DIR = Path("data/generated")
DEFAULT_NANORING_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "nanoring"
DEFAULT_THEOZYME_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "theozyme"
DEFAULT_ROTAMER_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "rotamers"
DEFAULT_SECONDARY_STRUCTURE_OUTPUT_DIR = (
    DEFAULT_GENERATED_DATA_DIR / "secondary_structure"
)
DEFAULT_REPORT_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "reports"
DEFAULT_SECONDARY_STRUCTURE_RESIDUES_BEFORE = 3
DEFAULT_SECONDARY_STRUCTURE_RESIDUES_AFTER = 3
DEFAULT_ROTAMER_CLASH_CUTOFF_PER_SITE = 6.0
DEFAULT_SECONDARY_STRUCTURE_CLASH_CUTOFF_PER_SITE = 6.0
RING_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)
BP5_VIRTUAL_CARBON_ATOMS = frozenset({"CV1", "CV2"})
BP5_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
_T = TypeVar("_T")

ROTAMER_REPORT_FIELDS = (
    "m",
    "units",
    "anchor_phase_offset",
    "snap_virtual_carbons",
    "residue_id",
    "anchor_angle_degrees",
    "rotamer_name",
    "chi1_target_degrees",
    "chi2_target_degrees",
    "chi1_degrees",
    "chi2_degrees",
    "chi2_validation_degrees",
    "state_scan_index",
    "state_score_rank",
    "accepted",
    "rotamer_clash_score",
    "rotamer_clash_score_per_site",
    "output_path",
)
SECONDARY_STRUCTURE_REPORT_FIELDS = (
    "m",
    "units",
    "anchor_phase_offset",
    "snap_virtual_carbons",
    "secondary_structure",
    "residues_before",
    "residues_after",
    "residue_id",
    "anchor_angle_degrees",
    "rotamer_name",
    "state_scan_index",
    "state_score_rank",
    "accepted",
    "cylinder_intrusion_count",
    "cylinder_intrusion_count_per_site",
    "cylinder_total_intrusion_depth",
    "cylinder_total_intrusion_depth_per_site",
    "cylinder_max_intrusion_depth",
    "cylinder_radius",
    "cylinder_z_min",
    "cylinder_z_max",
    "secondary_clash_score",
    "secondary_clash_score_per_site",
    "scaffold_clash_score",
    "scaffold_clash_score_per_site",
    "bp5_clash_score",
    "bp5_clash_score_per_site",
    "neighboring_backbone_clash_score",
    "neighboring_backbone_clash_score_per_site",
    "radial_alignment",
    "tangential_alignment",
    "axial_alignment",
    "secondary_structure_direction_x",
    "secondary_structure_direction_y",
    "secondary_structure_direction_z",
    "n_terminal_exit_vector_x",
    "n_terminal_exit_vector_y",
    "n_terminal_exit_vector_z",
    "c_terminal_exit_vector_x",
    "c_terminal_exit_vector_y",
    "c_terminal_exit_vector_z",
    "output_path",
)


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
    cylinder_intrusion_score: NanoringCylinderIntrusionScore

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

    @property
    def cylinder_intrusion_count(self) -> int:
        return self.cylinder_intrusion_score.intruding_atom_count

    @property
    def cylinder_total_intrusion_depth(self) -> float:
        return self.cylinder_intrusion_score.total_intrusion_depth

    @property
    def cylinder_max_intrusion_depth(self) -> float:
        return self.cylinder_intrusion_score.max_intrusion_depth


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
    cylinder_intrusion_score: NanoringCylinderIntrusionScore

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
    rotamer_scan_limit: int | None = None,
    max_rotamers_per_site: int | None = None,
    rotamer_clash_cutoff: float | None = DEFAULT_ROTAMER_CLASH_CUTOFF_PER_SITE,
    secondary_structure: SecondaryStructureType | None = None,
    secondary_structure_scan_limit: int | None = None,
    residues_before: int = DEFAULT_SECONDARY_STRUCTURE_RESIDUES_BEFORE,
    residues_after: int = DEFAULT_SECONDARY_STRUCTURE_RESIDUES_AFTER,
    secondary_structure_clash_cutoff: float | None = (
        DEFAULT_SECONDARY_STRUCTURE_CLASH_CUTOFF_PER_SITE
    ),
    secondary_structure_cylinder_filter: bool = True,
    secondary_structure_cylinder_radius: float | None = None,
) -> BP5NanoringRotamerPlacement:
    """Place BP5/Pd sidechains and enumerate fixed-active-site chi rotamers."""
    _validate_positive_limit(rotamer_scan_limit, "rotamer_scan_limit")
    if max_rotamers_per_site is not None and max_rotamers_per_site < 1:
        raise ValueError("max_rotamers_per_site must be at least 1")
    _validate_positive_limit(
        secondary_structure_scan_limit,
        "secondary_structure_scan_limit",
    )
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if (
        secondary_structure_cylinder_radius is not None
        and secondary_structure_cylinder_radius <= 0.0
    ):
        raise ValueError("secondary_structure_cylinder_radius must be positive")
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
    rotamers_to_scan = _limit_tuple(
        DEFAULT_BP5_CHI_ROTAMERS,
        rotamer_scan_limit,
    )

    candidates: list[BP5RotamerPlacement] = []
    for residue_id in range(1, len(rigid_placement.anchor_pairs) + 1):
        residue = rigid_placement.sidechains[
            rigid_placement.sidechains.res_id == residue_id
        ]
        residue_candidates = enumerate_bp5_chi_rotamers(
            atom_array=residue,
            bond_pairs=bond_pairs,
            rotamers=rotamers_to_scan,
            residue_id=residue_id,
        )
        candidates.extend(residue_candidates)

    rotamer_states = _build_symmetric_rotamer_states(
        candidates=tuple(candidates),
        nanoring=rigid_placement.nanoring,
        anchor_pairs=rigid_placement.anchor_pairs,
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
        rotamer_states_for_secondary_structure = _limit_tuple(
            accepted_rotamer_states,
            secondary_structure_scan_limit,
        )
        secondary_states = _build_symmetric_secondary_structure_states(
            rotamer_states=rotamer_states_for_secondary_structure,
            nanoring=rigid_placement.nanoring,
            anchor_pairs=rigid_placement.anchor_pairs,
            secondary_structure=secondary_structure,
            residues_before=residues_before,
            residues_after=residues_after,
            starting_atom_id=rigid_placement.nanoring.array_length() + 1,
            cylinder_radius=secondary_structure_cylinder_radius,
        )
        accepted_secondary_states = _select_symmetric_secondary_states(
            states=secondary_states,
            clash_cutoff=secondary_structure_clash_cutoff,
            cylinder_filter=secondary_structure_cylinder_filter,
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
    write_reports: bool = False,
    rotamer_scan_limit: int | None = None,
    max_rotamers_per_site: int | None = None,
    rotamer_clash_cutoff: float | None = DEFAULT_ROTAMER_CLASH_CUTOFF_PER_SITE,
    secondary_structure: SecondaryStructureType | None = None,
    secondary_structure_scan_limit: int | None = None,
    residues_before: int = DEFAULT_SECONDARY_STRUCTURE_RESIDUES_BEFORE,
    residues_after: int = DEFAULT_SECONDARY_STRUCTURE_RESIDUES_AFTER,
    secondary_structure_clash_cutoff: float | None = (
        DEFAULT_SECONDARY_STRUCTURE_CLASH_CUTOFF_PER_SITE
    ),
    secondary_structure_cylinder_filter: bool = True,
    secondary_structure_cylinder_radius: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """Write nanoring-only and BP5-placed structures for each requested M value."""
    _validate_positive_limit(rotamer_scan_limit, "rotamer_scan_limit")
    if max_rotamers_per_site is not None and max_rotamers_per_site < 1:
        raise ValueError("max_rotamers_per_site must be at least 1")
    _validate_positive_limit(
        secondary_structure_scan_limit,
        "secondary_structure_scan_limit",
    )
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if (
        secondary_structure_cylinder_radius is not None
        and secondary_structure_cylinder_radius <= 0.0
    ):
        raise ValueError("secondary_structure_cylinder_radius must be positive")
    if secondary_structure not in {None, "alpha_helix", "beta_strand"}:
        raise ValueError(
            "secondary_structure must be None, 'alpha_helix', or 'beta_strand'"
        )

    m_values = tuple(m_values)
    output_dir = Path(output_dir)
    nanoring_output_dir = output_dir / "nanoring"
    theozyme_output_dir = output_dir / "theozyme"
    rotamer_output_dir = output_dir / "rotamers"
    secondary_structure_output_dir = output_dir / "secondary_structure"
    report_output_dir = output_dir / "reports"
    nanoring_output_dir.mkdir(parents=True, exist_ok=True)
    theozyme_output_dir.mkdir(parents=True, exist_ok=True)
    if enumerate_bp5_rotamers:
        rotamer_output_dir.mkdir(parents=True, exist_ok=True)
    if secondary_structure is not None:
        secondary_structure_output_dir.mkdir(parents=True, exist_ok=True)
    if write_reports:
        report_output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    rotamer_report_rows: list[dict[str, object]] = []
    secondary_structure_report_rows: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for m_index, m in enumerate(m_values, start=1):
        _emit_progress(
            progress,
            f"[{m_index}/{len(m_values)}] M={m}: generating scaffold and rigid BP5 placement",
        )
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
        _emit_progress(
            progress,
            f"[{m_index}/{len(m_values)}] M={m}: wrote scaffold and rigid complex",
        )

        if not enumerate_bp5_rotamers and secondary_structure is None:
            run_summaries.append(
                {
                    "m": m,
                    "anchor_count": len(placement.anchor_pairs),
                    "rotamer_states_scanned": 0,
                    "rotamer_states_accepted": 0,
                    "secondary_structure_states_scanned": 0,
                    "secondary_structure_states_accepted": 0,
                }
            )
            continue

        rotamer_states_to_scan = min(
            len(DEFAULT_BP5_CHI_ROTAMERS),
            rotamer_scan_limit or len(DEFAULT_BP5_CHI_ROTAMERS),
        )
        _emit_progress(
            progress,
            (
                f"[{m_index}/{len(m_values)}] M={m}: scanning "
                f"{rotamer_states_to_scan} rotamer state(s) across "
                f"{len(placement.anchor_pairs)} BP5 site(s)"
            ),
        )
        rotamer_placement = place_bp5_rotamer_ensembles_around_nanoring(
            m=m,
            units=units,
            anchor_phase_offset=anchor_phase_offset,
            snap_virtual_carbons=snap_virtual_carbons,
            rotamer_scan_limit=rotamer_scan_limit,
            max_rotamers_per_site=max_rotamers_per_site,
            rotamer_clash_cutoff=rotamer_clash_cutoff,
            secondary_structure=secondary_structure,
            secondary_structure_scan_limit=secondary_structure_scan_limit,
            residues_before=residues_before,
            residues_after=residues_after,
            secondary_structure_clash_cutoff=secondary_structure_clash_cutoff,
            secondary_structure_cylinder_filter=secondary_structure_cylinder_filter,
            secondary_structure_cylinder_radius=secondary_structure_cylinder_radius,
        )
        _emit_progress(
            progress,
            (
                f"[{m_index}/{len(m_values)}] M={m}: accepted "
                f"{len(rotamer_placement.accepted_rotamer_states)}/"
                f"{len(rotamer_placement.rotamer_states)} rotamer state(s)"
            ),
        )
        rotamer_output_paths: dict[str, Path] = {}
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
                rotamer_output_paths[state.rotamer_name] = rotamer_path

        secondary_structure_output_paths: dict[str, Path] = {}
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
            secondary_structure_output_paths[state.rotamer_name] = segment_path
        if secondary_structure is not None:
            _emit_progress(
                progress,
                (
                    f"[{m_index}/{len(m_values)}] M={m}: accepted "
                    f"{len(rotamer_placement.accepted_secondary_structure_states)}/"
                    f"{len(rotamer_placement.secondary_structure_states)} "
                    f"{secondary_structure} state(s)"
                ),
            )
        if write_reports:
            rotamer_report_rows.extend(
                _rotamer_report_rows(
                    placement=rotamer_placement,
                    units=units,
                    anchor_phase_offset=anchor_phase_offset,
                    snap_virtual_carbons=snap_virtual_carbons,
                    output_paths=rotamer_output_paths,
                )
            )
            secondary_structure_report_rows.extend(
                _secondary_structure_report_rows(
                    placement=rotamer_placement,
                    units=units,
                    anchor_phase_offset=anchor_phase_offset,
                    snap_virtual_carbons=snap_virtual_carbons,
                    secondary_structure=secondary_structure,
                    residues_before=residues_before,
                    residues_after=residues_after,
                    output_paths=secondary_structure_output_paths,
                )
            )
        run_summaries.append(
            {
                "m": m,
                "anchor_count": len(rotamer_placement.anchor_pairs),
                "rotamer_states_scanned": len(rotamer_placement.rotamer_states),
                "rotamer_states_accepted": len(
                    rotamer_placement.accepted_rotamer_states
                ),
                "secondary_structure_states_scanned": len(
                    rotamer_placement.secondary_structure_states
                ),
                "secondary_structure_states_accepted": len(
                    rotamer_placement.accepted_secondary_structure_states
                ),
            }
        )
    if write_reports:
        rotamer_report_path = report_output_dir / "rotamer_scores.csv"
        secondary_structure_report_path = (
            report_output_dir / "secondary_structure_scores.csv"
        )
        run_metadata_path = report_output_dir / "run_metadata.json"
        _write_csv_report(
            path=rotamer_report_path,
            fieldnames=ROTAMER_REPORT_FIELDS,
            rows=rotamer_report_rows,
            overwrite=overwrite,
        )
        _write_csv_report(
            path=secondary_structure_report_path,
            fieldnames=SECONDARY_STRUCTURE_REPORT_FIELDS,
            rows=secondary_structure_report_rows,
            overwrite=overwrite,
        )
        _write_json_report(
            path=run_metadata_path,
            data={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "m_values": list(m_values),
                "units": units,
                "anchor_phase_offset": anchor_phase_offset,
                "snap_virtual_carbons": snap_virtual_carbons,
                "file_format": file_format,
                "enumerate_bp5_rotamers": enumerate_bp5_rotamers,
                "rotamer_scan_limit": rotamer_scan_limit,
                "max_rotamers_per_site": max_rotamers_per_site,
                "rotamer_clash_cutoff": rotamer_clash_cutoff,
                "rotamer_clash_cutoff_unit": "total_overlap_per_bp5_site",
                "secondary_structure": secondary_structure,
                "secondary_structure_scan_limit": secondary_structure_scan_limit,
                "residues_before": residues_before,
                "residues_after": residues_after,
                "secondary_structure_clash_cutoff": secondary_structure_clash_cutoff,
                "secondary_structure_clash_cutoff_unit": (
                    "total_overlap_per_bp5_site"
                ),
                "secondary_structure_cylinder_filter": (
                    secondary_structure_cylinder_filter
                ),
                "secondary_structure_cylinder_radius": (
                    secondary_structure_cylinder_radius
                ),
                "secondary_structure_cylinder_radius_unit": "Angstrom",
                "available_rotamer_states": len(DEFAULT_BP5_CHI_ROTAMERS),
                "summaries": run_summaries,
            },
            overwrite=overwrite,
        )
        written_paths.extend(
            [rotamer_report_path, secondary_structure_report_path, run_metadata_path]
        )
        _emit_progress(progress, f"Wrote reports under {report_output_dir}")
    return written_paths


def _rotamer_report_rows(
    placement: BP5NanoringRotamerPlacement,
    units: float | int,
    anchor_phase_offset: int,
    snap_virtual_carbons: bool,
    output_paths: dict[str, Path],
) -> list[dict[str, object]]:
    accepted_names = {state.rotamer_name for state in placement.accepted_rotamer_states}
    scan_index_by_name = {
        state.rotamer_name: scan_index
        for scan_index, state in enumerate(placement.rotamer_states, start=1)
    }
    score_rank_by_name = {
        state.rotamer_name: score_rank
        for score_rank, state in enumerate(
            sorted(placement.rotamer_states, key=_rotamer_state_score_key),
            start=1,
        )
    }
    rows: list[dict[str, object]] = []
    for state in placement.rotamer_states:
        for candidate in state.candidates:
            anchor_pair = placement.anchor_pairs[candidate.residue_id - 1]
            rows.append(
                {
                    "m": placement.m,
                    "units": units,
                    "anchor_phase_offset": anchor_phase_offset,
                    "snap_virtual_carbons": snap_virtual_carbons,
                    "residue_id": candidate.residue_id,
                    "anchor_angle_degrees": anchor_pair.angular_midpoint_degrees,
                    "rotamer_name": candidate.rotamer.name,
                    "chi1_target_degrees": candidate.rotamer.chi1_degrees,
                    "chi2_target_degrees": candidate.rotamer.chi2_degrees,
                    "chi1_degrees": candidate.chi1_degrees,
                    "chi2_degrees": candidate.chi2_degrees,
                    "chi2_validation_degrees": candidate.chi2_validation_degrees,
                    "state_scan_index": scan_index_by_name[candidate.rotamer.name],
                    "state_score_rank": score_rank_by_name[candidate.rotamer.name],
                    "accepted": candidate.rotamer.name in accepted_names,
                    "rotamer_clash_score": state.clash_score,
                    "rotamer_clash_score_per_site": (
                        _state_clash_score_per_site(state)
                    ),
                    "output_path": str(output_paths.get(candidate.rotamer.name, "")),
                }
            )
    return rows


def _secondary_structure_report_rows(
    placement: BP5NanoringRotamerPlacement,
    units: float | int,
    anchor_phase_offset: int,
    snap_virtual_carbons: bool,
    secondary_structure: SecondaryStructureType | None,
    residues_before: int,
    residues_after: int,
    output_paths: dict[str, Path],
) -> list[dict[str, object]]:
    if secondary_structure is None:
        return []

    accepted_names = {
        state.rotamer_name for state in placement.accepted_secondary_structure_states
    }
    scan_index_by_name = {
        state.rotamer_name: scan_index
        for scan_index, state in enumerate(placement.secondary_structure_states, start=1)
    }
    score_rank_by_name = {
        state.rotamer_name: score_rank
        for score_rank, state in enumerate(
            sorted(placement.secondary_structure_states, key=_secondary_state_score_key),
            start=1,
        )
    }
    rows: list[dict[str, object]] = []
    for state in placement.secondary_structure_states:
        for candidate in state.candidates:
            residue_id = candidate.rotamer_candidate.residue_id
            anchor_pair = placement.anchor_pairs[residue_id - 1]
            metrics = candidate.orientation_metrics
            rows.append(
                {
                    "m": placement.m,
                    "units": units,
                    "anchor_phase_offset": anchor_phase_offset,
                    "snap_virtual_carbons": snap_virtual_carbons,
                    "secondary_structure": secondary_structure,
                    "residues_before": residues_before,
                    "residues_after": residues_after,
                    "residue_id": residue_id,
                    "anchor_angle_degrees": anchor_pair.angular_midpoint_degrees,
                    "rotamer_name": candidate.rotamer_candidate.rotamer.name,
                    "state_scan_index": scan_index_by_name[
                        candidate.rotamer_candidate.rotamer.name
                    ],
                    "state_score_rank": score_rank_by_name[
                        candidate.rotamer_candidate.rotamer.name
                    ],
                    "accepted": candidate.rotamer_candidate.rotamer.name in accepted_names,
                    "cylinder_intrusion_count": (
                        state.cylinder_intrusion_score.intruding_atom_count
                    ),
                    "cylinder_intrusion_count_per_site": (
                        state.cylinder_intrusion_score.intruding_atom_count
                        / len(state.candidates)
                    ),
                    "cylinder_total_intrusion_depth": (
                        state.cylinder_intrusion_score.total_intrusion_depth
                    ),
                    "cylinder_total_intrusion_depth_per_site": (
                        state.cylinder_intrusion_score.total_intrusion_depth
                        / len(state.candidates)
                    ),
                    "cylinder_max_intrusion_depth": (
                        state.cylinder_intrusion_score.max_intrusion_depth
                    ),
                    "cylinder_radius": state.cylinder_intrusion_score.radius,
                    "cylinder_z_min": state.cylinder_intrusion_score.z_min,
                    "cylinder_z_max": state.cylinder_intrusion_score.z_max,
                    "secondary_clash_score": state.clash_score,
                    "secondary_clash_score_per_site": (
                        _state_clash_score_per_site(state)
                    ),
                    "scaffold_clash_score": state.scaffold_clash_score,
                    "scaffold_clash_score_per_site": (
                        state.scaffold_clash_score / len(state.candidates)
                    ),
                    "bp5_clash_score": state.bp5_clash_score,
                    "bp5_clash_score_per_site": (
                        state.bp5_clash_score / len(state.candidates)
                    ),
                    "neighboring_backbone_clash_score": (
                        state.neighboring_backbone_clash_score
                    ),
                    "neighboring_backbone_clash_score_per_site": (
                        state.neighboring_backbone_clash_score
                        / len(state.candidates)
                    ),
                    **_vector_report_columns(
                        "secondary_structure_direction",
                        metrics.secondary_structure_direction,
                    ),
                    **_vector_report_columns(
                        "n_terminal_exit_vector",
                        metrics.n_terminal_exit_vector,
                    ),
                    **_vector_report_columns(
                        "c_terminal_exit_vector",
                        metrics.c_terminal_exit_vector,
                    ),
                    "radial_alignment": metrics.radial_alignment,
                    "tangential_alignment": metrics.tangential_alignment,
                    "axial_alignment": metrics.axial_alignment,
                    "output_path": str(
                        output_paths.get(candidate.rotamer_candidate.rotamer.name, "")
                    ),
                }
            )
    return rows


def _vector_report_columns(prefix: str, vector: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_x": float(vector[0]),
        f"{prefix}_y": float(vector[1]),
        f"{prefix}_z": float(vector[2]),
    }


def _write_csv_report(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, object]],
    overwrite: bool,
) -> None:
    _raise_if_exists(path, overwrite=overwrite)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_report(path: Path, data: dict[str, object], overwrite: bool) -> None:
    _raise_if_exists(path, overwrite=overwrite)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def _raise_if_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")


def _emit_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _validate_positive_limit(value: int | None, name: str) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1")


def _limit_tuple(values: tuple[_T, ...], limit: int | None) -> tuple[_T, ...]:
    if limit is None:
        return values
    return values[:limit]


@dataclass(frozen=True)
class _BP5StateClashScore:
    scaffold_score: float
    bp5_score: float

    @property
    def total_overlap_score(self) -> float:
        return float(self.scaffold_score + self.bp5_score)


@dataclass(frozen=True)
class _SymmetricSecondaryStructureScore:
    clash_score: SecondaryStructureClashScore
    cylinder_intrusion_score: NanoringCylinderIntrusionScore


def _build_symmetric_rotamer_states(
    candidates: tuple[BP5RotamerPlacement, ...],
    nanoring: struc.AtomArray,
    anchor_pairs: tuple[NanoringAnchorPair, ...],
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
            anchor_pairs=tuple(
                anchor_pairs[candidate.residue_id - 1] for candidate in ordered_group
            ),
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
        if rotamer_clash_cutoff is None
        or _state_clash_score_per_site(state) <= rotamer_clash_cutoff
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
    cylinder_filter: bool,
) -> tuple[BP5SymmetricSecondaryStructureState, ...]:
    accepted_groups = [
        state
        for state in states
        if clash_cutoff is None or _state_clash_score_per_site(state) <= clash_cutoff
    ]
    if cylinder_filter:
        accepted_groups = [
            state
            for state in accepted_groups
            if state.cylinder_intrusion_score.passes
        ]
    return tuple(sorted(accepted_groups, key=_secondary_state_score_key))


def _state_clash_score_per_site(
    state: BP5SymmetricRotamerState | BP5SymmetricSecondaryStructureState,
) -> float:
    return float(state.clash_score / len(state.candidates))


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
    anchor_pairs: tuple[NanoringAnchorPair, ...],
    secondary_structure: SecondaryStructureType,
    residues_before: int,
    residues_after: int,
    starting_atom_id: int,
    cylinder_radius: float | None,
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
            anchor_pairs=anchor_pairs,
            cylinder_radius=cylinder_radius,
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
            cylinder_intrusion_score = score_nanoring_cylinder_intrusions(
                atom_array=segment.atom_array,
                nanoring=nanoring,
                cylinder_radius=cylinder_radius,
            )
            scored_candidates.append(
                BP5SecondaryStructurePlacement(
                    rotamer_candidate=candidate,
                    segment=segment,
                    orientation_metrics=orientation_metrics,
                    scaffold_clash_score=score.clash_score.scaffold_score,
                    bp5_clash_score=score.clash_score.bp5_score,
                    neighboring_backbone_clash_score=(
                        score.clash_score.neighboring_backbone_score
                    ),
                    clash_score=score.clash_score.total_overlap_score,
                    cylinder_intrusion_score=cylinder_intrusion_score,
                )
            )
        states.append(
            BP5SymmetricSecondaryStructureState(
                rotamer_name=rotamer_state.rotamer_name,
                candidates=tuple(scored_candidates),
                scaffold_clash_score=score.clash_score.scaffold_score,
                bp5_clash_score=score.clash_score.bp5_score,
                neighboring_backbone_clash_score=(
                    score.clash_score.neighboring_backbone_score
                ),
                clash_score=score.clash_score.total_overlap_score,
                cylinder_intrusion_score=score.cylinder_intrusion_score,
            ),
        )
    return tuple(states)


def _score_bp5_atom_group(
    atom_arrays: tuple[struc.AtomArray, ...],
    nanoring: struc.AtomArray,
    anchor_pairs: tuple[NanoringAnchorPair, ...],
) -> _BP5StateClashScore:
    bp5_arrays = tuple(
        _without_atom_names(atom_array, BP5_VIRTUAL_CARBON_ATOMS)
        for atom_array in atom_arrays
    )
    scaffold_score = _score_bp5_arrays_against_nanoring(
        bp5_arrays=bp5_arrays,
        nanoring=nanoring,
        anchor_pairs=anchor_pairs,
    )
    inter_bp5_score = _score_atom_array_pairs(bp5_arrays)
    return _BP5StateClashScore(
        scaffold_score=scaffold_score,
        bp5_score=inter_bp5_score,
    )


def _score_symmetric_secondary_structure_state(
    segment_pairs: tuple[tuple[BP5RotamerPlacement, SecondaryStructureSegment], ...],
    nanoring: struc.AtomArray,
    anchor_pairs: tuple[NanoringAnchorPair, ...],
    cylinder_radius: float | None,
) -> _SymmetricSecondaryStructureScore:
    segments = tuple(segment for _, segment in segment_pairs)
    bp5_arrays = tuple(_bp5_segment_atoms(segment) for segment in segments)
    generated_arrays = tuple(_generated_segment_atoms(segment) for segment in segments)
    bp5_score = _score_bp5_atom_group(
        atom_arrays=bp5_arrays,
        nanoring=nanoring,
        anchor_pairs=tuple(
            anchor_pairs[candidate.residue_id - 1] for candidate, _ in segment_pairs
        ),
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
                ignore_same_residue=True,
                ignore_inter_residue_backbone_n_c=True,
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
                ignore_same_residue=True,
                ignore_inter_residue_backbone_n_c=True,
            ).total_overlap_score

    clash_score = SecondaryStructureClashScore(
        scaffold_score=scaffold_score,
        bp5_score=float(
            bp5_score.bp5_score
            + own_generated_bp5_score
            + neighboring_generated_bp5_score
        ),
        neighboring_backbone_score=_score_atom_array_pairs(generated_arrays),
    )
    cylinder_intrusion_score = score_nanoring_cylinder_intrusions(
        atom_array=struc.concatenate([segment.atom_array for segment in segments]),
        nanoring=nanoring,
        cylinder_radius=cylinder_radius,
    )
    return _SymmetricSecondaryStructureScore(
        clash_score=clash_score,
        cylinder_intrusion_score=cylinder_intrusion_score,
    )


def _score_bp5_arrays_against_nanoring(
    bp5_arrays: tuple[struc.AtomArray, ...],
    nanoring: struc.AtomArray,
    anchor_pairs: tuple[NanoringAnchorPair, ...],
) -> float:
    score = 0.0
    for bp5_array, anchor_pair in zip(bp5_arrays, anchor_pairs, strict=True):
        score += score_heavy_atom_clashes(
            atom_array=bp5_array,
            other=nanoring,
            ignored_atom_index_pairs=_pd_anchor_atom_index_pairs(
                bp5_array,
                anchor_pair,
            ),
            ignore_same_residue=True,
            ignore_inter_residue_backbone_n_c=True,
        ).total_overlap_score
    return float(score)


def _pd_anchor_atom_index_pairs(
    bp5_array: struc.AtomArray,
    anchor_pair: NanoringAnchorPair,
) -> tuple[tuple[int, int], ...]:
    atom_names = bp5_array.atom_name.tolist()
    if "PD" not in atom_names:
        return ()
    pd_index = atom_names.index("PD")
    return tuple(
        (pd_index, anchor_index) for anchor_index in anchor_pair.atom_indices
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
                ignore_same_residue=True,
                ignore_inter_residue_backbone_n_c=True,
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
                ignore_same_residue=True,
                ignore_inter_residue_backbone_n_c=True,
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
        "--write-reports",
        action="store_true",
        help="Write CSV score reports and JSON run metadata under reports/.",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help=(
            "Only scan the first N deterministic rotamer states and grow at most "
            "the first N secondary-structure states."
        ),
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
        default=DEFAULT_ROTAMER_CLASH_CUTOFF_PER_SITE,
        help=(
            "Reject rotamer states with total overlap per BP5 site above this "
            f"value. Defaults to {DEFAULT_ROTAMER_CLASH_CUTOFF_PER_SITE}."
        ),
    )
    parser.add_argument(
        "--secondary-structure-clash-cutoff",
        type=float,
        default=DEFAULT_SECONDARY_STRUCTURE_CLASH_CUTOFF_PER_SITE,
        help=(
            "Reject grown secondary-structure states with total overlap per BP5 "
            "site above this value. Defaults to "
            f"{DEFAULT_SECONDARY_STRUCTURE_CLASH_CUTOFF_PER_SITE}."
        ),
    )
    parser.add_argument(
        "--secondary-structure-cylinder-radius",
        type=float,
        default=None,
        help=(
            "Cylinder radius for secondary-structure interior exclusion. "
            "Defaults to the innermost nanoring carbon radial distance."
        ),
    )
    parser.add_argument(
        "--allow-secondary-structure-cylinder-intrusions",
        action="store_true",
        help=(
            "Do not reject secondary-structure states whose atom centers enter "
            "the nanoring interior cylinder."
        ),
    )
    parser.add_argument(
        "--no-clash-cutoffs",
        action="store_true",
        help="Disable default rotamer and secondary-structure overlap cutoffs.",
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
        default=DEFAULT_SECONDARY_STRUCTURE_RESIDUES_BEFORE,
        help=(
            "Number of residues to grow before BP5 in secondary-structure mode. "
            f"Defaults to {DEFAULT_SECONDARY_STRUCTURE_RESIDUES_BEFORE}."
        ),
    )
    parser.add_argument(
        "--residues-after",
        type=int,
        default=DEFAULT_SECONDARY_STRUCTURE_RESIDUES_AFTER,
        help=(
            "Number of residues to grow after BP5 in secondary-structure mode. "
            f"Defaults to {DEFAULT_SECONDARY_STRUCTURE_RESIDUES_AFTER}."
        ),
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

    rotamer_clash_cutoff = (
        None if args.no_clash_cutoffs else args.rotamer_clash_cutoff
    )
    secondary_structure_clash_cutoff = (
        None if args.no_clash_cutoffs else args.secondary_structure_clash_cutoff
    )
    written_paths = write_bp5_nanoring_series(
        output_dir=args.output_dir,
        m_values=args.m,
        units=args.units,
        anchor_phase_offset=args.anchor_phase_offset,
        snap_virtual_carbons=args.snap_virtual_carbons,
        file_format=args.format,
        overwrite=args.overwrite,
        enumerate_bp5_rotamers=args.enumerate_bp5_rotamers,
        write_reports=args.write_reports,
        rotamer_scan_limit=args.scan_limit,
        max_rotamers_per_site=args.max_rotamers_per_site,
        rotamer_clash_cutoff=rotamer_clash_cutoff,
        secondary_structure=(
            None if args.secondary_structure == "none" else args.secondary_structure
        ),
        secondary_structure_scan_limit=args.scan_limit,
        residues_before=args.residues_before,
        residues_after=args.residues_after,
        secondary_structure_clash_cutoff=secondary_structure_clash_cutoff,
        secondary_structure_cylinder_filter=(
            not args.allow_secondary_structure_cylinder_intrusions
        ),
        secondary_structure_cylinder_radius=args.secondary_structure_cylinder_radius,
        progress=lambda message: print(message, file=sys.stderr),
    )
    for path in written_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
