from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np

from tuber.writers import write_structure

from .active_site import build_bp5_palladium_active_site
from .ligands import DEFAULT_LIGAND_DIR
from .nanoring import generate_armchair_nanoring

DEFAULT_M_VALUES = (18, 24, 30, 36)
DEFAULT_GENERATED_DATA_DIR = Path("data/generated")
DEFAULT_NANORING_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "nanoring"
DEFAULT_THEOZYME_OUTPUT_DIR = DEFAULT_GENERATED_DATA_DIR / "theozyme"
RING_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


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


def write_bp5_nanoring_series(
    output_dir: str | Path = DEFAULT_GENERATED_DATA_DIR,
    m_values: Iterable[int] = DEFAULT_M_VALUES,
    units: float | int = 1.5,
    anchor_phase_offset: int = 1,
    snap_virtual_carbons: bool = False,
    file_format: str = "cif",
    overwrite: bool = False,
) -> list[Path]:
    """Write nanoring-only and BP5-placed structures for each requested M value."""
    output_dir = Path(output_dir)
    nanoring_output_dir = output_dir / "nanoring"
    theozyme_output_dir = output_dir / "theozyme"
    nanoring_output_dir.mkdir(parents=True, exist_ok=True)
    theozyme_output_dir.mkdir(parents=True, exist_ok=True)

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
    return written_paths


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
    )
    for path in written_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
