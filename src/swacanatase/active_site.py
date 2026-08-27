from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np

from .ligands import DEFAULT_LIGAND_DIR, load_bp5_atom_array, load_bp5_bond_pairs

BP5_INTER_RING_BOND = ("C3", "C6")
BP5_DONOR_ATOMS = ("N1", "N2")
BP5_RING_1_ATOMS = ("N1", "C1", "C2", "C3", "C4", "C5")
BP5_RING_2_ATOMS = ("N2", "C6", "C7", "C8", "C9", "C11")

DEFAULT_PD_N_BOND_LENGTH = 2.018
DEFAULT_PD_C_BOND_LENGTH = 2.02


@dataclass(frozen=True)
class SquarePlanarGeometry:
    """Geometric summary for the appended Pd coordination site."""

    donor_atom_names: tuple[str, str]
    palladium_atom_name: str
    virtual_carbon_atom_names: tuple[str, str]
    pd_n_bond_length: float
    pd_c_bond_length: float
    coordination_plane_normal: np.ndarray


def load_active_bp5_atom_array(
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
    coordinate_set: str = "ideal",
) -> struc.AtomArray:
    """Load BP5 from CCD coordinates and convert bipyridine into the active cis form."""
    return make_bp5_active_conformer(
        atom_array=load_bp5_atom_array(cif_path=cif_path, coordinate_set=coordinate_set),
        bond_pairs=load_bp5_bond_pairs(cif_path=cif_path),
    )


def make_bp5_active_conformer(
    atom_array: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
    inter_ring_bond: tuple[str, str] = BP5_INTER_RING_BOND,
) -> struc.AtomArray:
    """Rotate the C6 side of BP5 180 degrees to put N1 and N2 on the same side.

    The source CCD conformer is kept traceable by applying a rigid rotation to the
    connected component on the second-ring side of the C3-C6 inter-ring bond.
    """
    atom_array = atom_array.copy()
    fixed_atom, moving_atom = inter_ring_bond
    name_to_index = _atom_indices(atom_array)
    moving_atom_names = _connected_side_atom_names(
        bond_pairs=bond_pairs,
        fixed_atom=fixed_atom,
        moving_atom=moving_atom,
    )
    moving_indices = np.array([name_to_index[name] for name in moving_atom_names], dtype=int)

    axis_start = atom_array.coord[name_to_index[fixed_atom]]
    axis_end = atom_array.coord[name_to_index[moving_atom]]
    atom_array.coord[moving_indices] = _rotate_about_axis(
        points=atom_array.coord[moving_indices],
        axis_start=axis_start,
        axis_end=axis_end,
        angle_radians=np.pi,
    )
    return atom_array


def build_bp5_palladium_active_site(
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
    coordinate_set: str = "ideal",
    pd_n_bond_length: float = DEFAULT_PD_N_BOND_LENGTH,
    pd_c_bond_length: float = DEFAULT_PD_C_BOND_LENGTH,
) -> struc.AtomArray:
    """Build active BP5 and append square-planar Pd plus two virtual carbons."""
    active_bp5 = load_active_bp5_atom_array(cif_path=cif_path, coordinate_set=coordinate_set)
    return append_square_planar_palladium_site(
        atom_array=active_bp5,
        pd_n_bond_length=pd_n_bond_length,
        pd_c_bond_length=pd_c_bond_length,
    )


def append_square_planar_palladium_site(
    atom_array: struc.AtomArray,
    donor_atom_names: tuple[str, str] = BP5_DONOR_ATOMS,
    pd_n_bond_length: float = DEFAULT_PD_N_BOND_LENGTH,
    pd_c_bond_length: float = DEFAULT_PD_C_BOND_LENGTH,
) -> struc.AtomArray:
    """Append Pd and virtual carbon positions in the BP5 coordination plane."""
    name_to_index = _atom_indices(atom_array)
    donor_1 = atom_array.coord[name_to_index[donor_atom_names[0]]]
    donor_2 = atom_array.coord[name_to_index[donor_atom_names[1]]]
    plane_normal = _best_fit_plane_normal(
        atom_array,
        atom_names=BP5_RING_1_ATOMS + BP5_RING_2_ATOMS,
    )
    pd_coord = _palladium_coord(
        atom_array=atom_array,
        donor_1=donor_1,
        donor_2=donor_2,
        plane_normal=plane_normal,
        pd_n_bond_length=pd_n_bond_length,
    )

    n1_direction = _unit(donor_1 - pd_coord)
    n2_direction = _unit(donor_2 - pd_coord)
    virtual_c_1 = pd_coord - n2_direction * pd_c_bond_length
    virtual_c_2 = pd_coord - n1_direction * pd_c_bond_length

    extra_atoms = _new_atoms(
        atom_names=("PD", "CV1", "CV2"),
        elements=("PD", "C", "C"),
        res_names=("PD", "VRT", "VRT"),
        coords=np.vstack((pd_coord, virtual_c_1, virtual_c_2)),
        starting_atom_id=int(np.max(atom_array.atom_id)) + 1
        if "atom_id" in atom_array.get_annotation_categories()
        else atom_array.array_length() + 1,
    )
    return struc.concatenate([atom_array, extra_atoms])


def describe_square_planar_geometry(
    atom_array: struc.AtomArray,
    donor_atom_names: tuple[str, str] = BP5_DONOR_ATOMS,
    palladium_atom_name: str = "PD",
    virtual_carbon_atom_names: tuple[str, str] = ("CV1", "CV2"),
) -> SquarePlanarGeometry:
    """Measure the active-site distances and coordination plane."""
    name_to_index = _atom_indices(atom_array)
    pd_coord = atom_array.coord[name_to_index[palladium_atom_name]]
    donor_coords = [atom_array.coord[name_to_index[name]] for name in donor_atom_names]
    virtual_coords = [
        atom_array.coord[name_to_index[name]] for name in virtual_carbon_atom_names
    ]
    plane_normal = _best_fit_normal(np.vstack((donor_coords, pd_coord, virtual_coords)))
    return SquarePlanarGeometry(
        donor_atom_names=donor_atom_names,
        palladium_atom_name=palladium_atom_name,
        virtual_carbon_atom_names=virtual_carbon_atom_names,
        pd_n_bond_length=float(
            np.mean([np.linalg.norm(coord - pd_coord) for coord in donor_coords])
        ),
        pd_c_bond_length=float(
            np.mean([np.linalg.norm(coord - pd_coord) for coord in virtual_coords])
        ),
        coordination_plane_normal=plane_normal,
    )


def _atom_indices(atom_array: struc.AtomArray) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, atom_name in enumerate(atom_array.atom_name.tolist()):
        if atom_name in indices:
            raise ValueError(f"Duplicate atom name {atom_name!r}")
        indices[atom_name] = index
    return indices


def _connected_side_atom_names(
    bond_pairs: Iterable[tuple[str, str]],
    fixed_atom: str,
    moving_atom: str,
) -> list[str]:
    blocked_bond = frozenset((fixed_atom, moving_atom))
    graph: dict[str, set[str]] = {}
    for atom_1, atom_2 in bond_pairs:
        if frozenset((atom_1, atom_2)) == blocked_bond:
            continue
        graph.setdefault(atom_1, set()).add(atom_2)
        graph.setdefault(atom_2, set()).add(atom_1)

    seen = {moving_atom}
    stack = [moving_atom]
    while stack:
        atom_name = stack.pop()
        for neighbor in graph.get(atom_name, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return sorted(seen)


def _rotate_about_axis(
    points: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    axis = _unit(axis_end - axis_start)
    shifted = points - axis_start
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    return (
        axis_start
        + shifted * cos_theta
        + np.cross(axis, shifted) * sin_theta
        + axis * np.dot(shifted, axis)[:, np.newaxis] * (1.0 - cos_theta)
    )


def _palladium_coord(
    atom_array: struc.AtomArray,
    donor_1: np.ndarray,
    donor_2: np.ndarray,
    plane_normal: np.ndarray,
    pd_n_bond_length: float,
) -> np.ndarray:
    donor_vector = donor_2 - donor_1
    donor_distance = float(np.linalg.norm(donor_vector))
    if donor_distance >= 2 * pd_n_bond_length:
        raise ValueError(
            "pd_n_bond_length is too short for the BP5 donor atom spacing "
            f"({pd_n_bond_length:.3f} Å for N-N distance {donor_distance:.3f} Å)"
        )

    donor_midpoint = (donor_1 + donor_2) / 2.0
    donor_axis = _unit(donor_vector)
    in_plane_perpendicular = _unit(np.cross(plane_normal, donor_axis))
    metal_direction = _bp5_donor_lone_pair_direction(atom_array)
    if np.dot(in_plane_perpendicular, metal_direction) < 0:
        in_plane_perpendicular *= -1
    height = np.sqrt(pd_n_bond_length**2 - (donor_distance / 2.0) ** 2)
    return donor_midpoint + in_plane_perpendicular * height


def _bp5_donor_lone_pair_direction(atom_array: struc.AtomArray) -> np.ndarray:
    name_to_index = _atom_indices(atom_array)
    ring_1_centroid = _atom_centroid(atom_array, BP5_RING_1_ATOMS)
    ring_2_centroid = _atom_centroid(atom_array, BP5_RING_2_ATOMS)
    n1_direction = _unit(atom_array.coord[name_to_index["N1"]] - ring_1_centroid)
    n2_direction = _unit(atom_array.coord[name_to_index["N2"]] - ring_2_centroid)
    return _unit(n1_direction + n2_direction)


def _atom_centroid(atom_array: struc.AtomArray, atom_names: tuple[str, ...]) -> np.ndarray:
    name_to_index = _atom_indices(atom_array)
    return np.mean([atom_array.coord[name_to_index[name]] for name in atom_names], axis=0)


def _best_fit_plane_normal(atom_array: struc.AtomArray, atom_names: tuple[str, ...]) -> np.ndarray:
    name_to_index = _atom_indices(atom_array)
    points = np.array([atom_array.coord[name_to_index[name]] for name in atom_names])
    return _best_fit_normal(points)


def _best_fit_normal(points: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(points - np.mean(points, axis=0))
    return _unit(vh[-1])


def _new_atoms(
    atom_names: tuple[str, ...],
    elements: tuple[str, ...],
    res_names: tuple[str, ...],
    coords: np.ndarray,
    starting_atom_id: int,
) -> struc.AtomArray:
    atoms = struc.AtomArray(len(atom_names))
    atoms.coord = coords.astype(float, copy=False)
    atoms.chain_id = np.full(len(atom_names), "L", dtype="U4")
    atoms.res_id = np.ones(len(atom_names), dtype=int)
    atoms.ins_code = np.full(len(atom_names), "", dtype="U1")
    atoms.res_name = np.array(res_names, dtype="U5")
    atoms.hetero = np.ones(len(atom_names), dtype=bool)
    atoms.atom_name = np.array(atom_names, dtype="U6")
    atoms.element = np.array(elements, dtype="U2")
    atoms.set_annotation(
        "atom_id",
        np.arange(starting_atom_id, starting_atom_id + len(atom_names), dtype=int),
    )
    atoms.set_annotation("occupancy", np.ones(len(atom_names), dtype=float))
    atoms.set_annotation("b_factor", np.zeros(len(atom_names), dtype=float))
    return atoms


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm
