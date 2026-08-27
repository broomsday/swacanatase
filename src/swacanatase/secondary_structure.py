from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import biotite.structure as struc
import numpy as np

from .bp5_rotamers import BP5RotamerPlacement

SecondaryStructureType = Literal["alpha_helix", "beta_strand"]

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")

N_CA_BOND_LENGTH = 1.458
CA_C_BOND_LENGTH = 1.525
C_N_BOND_LENGTH = 1.329
C_O_BOND_LENGTH = 1.229

C_N_CA_ANGLE_DEGREES = 121.7
N_CA_C_ANGLE_DEGREES = 111.2
CA_C_N_ANGLE_DEGREES = 116.2
CA_C_O_ANGLE_DEGREES = 120.8
TRANS_PEPTIDE_OMEGA_DEGREES = 180.0
CARBONYL_O_DIHEDRAL_DEGREES = 180.0


@dataclass(frozen=True)
class BackboneTorsionTargets:
    phi_degrees: float
    psi_degrees: float


@dataclass(frozen=True)
class SecondaryStructureSegment:
    secondary_structure_type: SecondaryStructureType
    atom_array: struc.AtomArray
    bp5_residue_id: int
    residues_before: int
    residues_after: int
    torsion_targets: BackboneTorsionTargets


SECONDARY_STRUCTURE_TARGETS: dict[SecondaryStructureType, BackboneTorsionTargets] = {
    "alpha_helix": BackboneTorsionTargets(phi_degrees=-60.0, psi_degrees=-45.0),
    "beta_strand": BackboneTorsionTargets(phi_degrees=-135.0, psi_degrees=135.0),
}


def build_regular_secondary_structure_segment(
    bp5_rotamer: BP5RotamerPlacement | struc.AtomArray,
    secondary_structure_type: SecondaryStructureType,
    residues_before: int,
    residues_after: int,
    chain_id: str = "A",
    starting_residue_id: int = 1,
    starting_atom_id: int = 1,
) -> SecondaryStructureSegment:
    """Grow a deterministic regular backbone segment around a fixed BP5 frame."""
    if residues_before < 0 or residues_after < 0:
        raise ValueError("residue counts must be non-negative")
    if secondary_structure_type not in SECONDARY_STRUCTURE_TARGETS:
        raise ValueError(
            "secondary_structure_type must be 'alpha_helix' or 'beta_strand'"
        )

    bp5 = (
        bp5_rotamer.atom_array.copy()
        if isinstance(bp5_rotamer, BP5RotamerPlacement)
        else bp5_rotamer.copy()
    )
    _require_atom_names(bp5, ("N", "CA", "C", "O"))

    torsion_targets = SECONDARY_STRUCTURE_TARGETS[secondary_structure_type]
    bp5_residue_id = starting_residue_id + residues_before
    backbone_coords = _grow_backbone_coordinates(
        bp5=bp5,
        residues_before=residues_before,
        residues_after=residues_after,
        targets=torsion_targets,
    )

    arrays: list[struc.AtomArray] = []
    next_atom_id = starting_atom_id
    for residue_id in range(starting_residue_id, bp5_residue_id):
        local_residue_id = residue_id - starting_residue_id + 1
        arrays.append(
            _new_backbone_residue(
                residue_id=residue_id,
                chain_id=chain_id,
                coordinates=backbone_coords[local_residue_id],
                starting_atom_id=next_atom_id,
            )
        )
        next_atom_id += len(BACKBONE_ATOM_NAMES)

    prepared_bp5 = _prepare_bp5_residue(
        bp5=bp5,
        residue_id=bp5_residue_id,
        chain_id=chain_id,
        starting_atom_id=next_atom_id,
        has_previous=residues_before > 0,
        has_next=residues_after > 0,
    )
    arrays.append(prepared_bp5)
    next_atom_id += prepared_bp5.array_length()

    for residue_id in range(bp5_residue_id + 1, bp5_residue_id + residues_after + 1):
        local_residue_id = residue_id - starting_residue_id + 1
        arrays.append(
            _new_backbone_residue(
                residue_id=residue_id,
                chain_id=chain_id,
                coordinates=backbone_coords[local_residue_id],
                starting_atom_id=next_atom_id,
            )
        )
        next_atom_id += len(BACKBONE_ATOM_NAMES)

    return SecondaryStructureSegment(
        secondary_structure_type=secondary_structure_type,
        atom_array=struc.concatenate(arrays),
        bp5_residue_id=bp5_residue_id,
        residues_before=residues_before,
        residues_after=residues_after,
        torsion_targets=torsion_targets,
    )


def _grow_backbone_coordinates(
    bp5: struc.AtomArray,
    residues_before: int,
    residues_after: int,
    targets: BackboneTorsionTargets,
) -> dict[int, dict[str, np.ndarray]]:
    bp5_residue_id = residues_before + 1
    bp5_coords = {
        atom_name: _atom_coord(bp5, atom_name).copy() for atom_name in BACKBONE_ATOM_NAMES
    }
    coordinates: dict[int, dict[str, np.ndarray]] = {bp5_residue_id: bp5_coords}

    for residue_id in range(bp5_residue_id - 1, 0, -1):
        next_coords = coordinates[residue_id + 1]
        c_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["C"],
            atom_2=next_coords["CA"],
            atom_3=next_coords["N"],
            bond_length=C_N_BOND_LENGTH,
            bond_angle_degrees=C_N_CA_ANGLE_DEGREES,
            dihedral_degrees=targets.phi_degrees,
        )
        ca_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["CA"],
            atom_2=next_coords["N"],
            atom_3=c_coord,
            bond_length=CA_C_BOND_LENGTH,
            bond_angle_degrees=CA_C_N_ANGLE_DEGREES,
            dihedral_degrees=TRANS_PEPTIDE_OMEGA_DEGREES,
        )
        n_coord = _place_internal_coordinate_atom(
            atom_1=next_coords["N"],
            atom_2=c_coord,
            atom_3=ca_coord,
            bond_length=N_CA_BOND_LENGTH,
            bond_angle_degrees=N_CA_C_ANGLE_DEGREES,
            dihedral_degrees=targets.psi_degrees,
        )
        o_coord = _place_carbonyl_oxygen(n_coord, ca_coord, c_coord)
        coordinates[residue_id] = {
            "N": n_coord,
            "CA": ca_coord,
            "C": c_coord,
            "O": o_coord,
        }

    for residue_id in range(bp5_residue_id + 1, bp5_residue_id + residues_after + 1):
        previous_coords = coordinates[residue_id - 1]
        n_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["N"],
            atom_2=previous_coords["CA"],
            atom_3=previous_coords["C"],
            bond_length=C_N_BOND_LENGTH,
            bond_angle_degrees=CA_C_N_ANGLE_DEGREES,
            dihedral_degrees=targets.psi_degrees,
        )
        ca_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["CA"],
            atom_2=previous_coords["C"],
            atom_3=n_coord,
            bond_length=N_CA_BOND_LENGTH,
            bond_angle_degrees=C_N_CA_ANGLE_DEGREES,
            dihedral_degrees=TRANS_PEPTIDE_OMEGA_DEGREES,
        )
        c_coord = _place_internal_coordinate_atom(
            atom_1=previous_coords["C"],
            atom_2=n_coord,
            atom_3=ca_coord,
            bond_length=CA_C_BOND_LENGTH,
            bond_angle_degrees=N_CA_C_ANGLE_DEGREES,
            dihedral_degrees=targets.phi_degrees,
        )
        o_coord = _place_carbonyl_oxygen(n_coord, ca_coord, c_coord)
        coordinates[residue_id] = {
            "N": n_coord,
            "CA": ca_coord,
            "C": c_coord,
            "O": o_coord,
        }

    return coordinates


def _place_carbonyl_oxygen(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
) -> np.ndarray:
    return _place_internal_coordinate_atom(
        atom_1=n_coord,
        atom_2=ca_coord,
        atom_3=c_coord,
        bond_length=C_O_BOND_LENGTH,
        bond_angle_degrees=CA_C_O_ANGLE_DEGREES,
        dihedral_degrees=CARBONYL_O_DIHEDRAL_DEGREES,
    )


def _place_internal_coordinate_atom(
    atom_1: np.ndarray,
    atom_2: np.ndarray,
    atom_3: np.ndarray,
    bond_length: float,
    bond_angle_degrees: float,
    dihedral_degrees: float,
) -> np.ndarray:
    axis = _unit(atom_3 - atom_2)
    normal = _unit(np.cross(atom_2 - atom_1, axis))
    perpendicular = np.cross(normal, axis)
    bond_angle = np.radians(bond_angle_degrees)
    dihedral = np.radians(dihedral_degrees)
    direction = (
        np.cos(np.pi - bond_angle) * axis
        + np.sin(np.pi - bond_angle)
        * (np.cos(dihedral) * perpendicular + np.sin(dihedral) * normal)
    )
    return atom_3 + bond_length * direction


def _new_backbone_residue(
    residue_id: int,
    chain_id: str,
    coordinates: dict[str, np.ndarray],
    starting_atom_id: int,
) -> struc.AtomArray:
    atom_count = len(BACKBONE_ATOM_NAMES)
    atoms = struc.AtomArray(atom_count)
    atoms.coord = np.array([coordinates[name] for name in BACKBONE_ATOM_NAMES])
    atoms.chain_id = np.full(atom_count, chain_id, dtype="U4")
    atoms.res_id = np.full(atom_count, residue_id, dtype=int)
    atoms.ins_code = np.full(atom_count, "", dtype="U1")
    atoms.res_name = np.full(atom_count, "GLY", dtype="U5")
    atoms.hetero = np.zeros(atom_count, dtype=bool)
    atoms.atom_name = np.array(BACKBONE_ATOM_NAMES, dtype="U6")
    atoms.element = np.array(("N", "C", "C", "O"), dtype="U2")
    atoms.set_annotation(
        "atom_id",
        np.arange(starting_atom_id, starting_atom_id + atom_count, dtype=int),
    )
    atoms.set_annotation("occupancy", np.ones(atom_count, dtype=float))
    atoms.set_annotation("b_factor", np.zeros(atom_count, dtype=float))
    return atoms


def _prepare_bp5_residue(
    bp5: struc.AtomArray,
    residue_id: int,
    chain_id: str,
    starting_atom_id: int,
    has_previous: bool,
    has_next: bool,
) -> struc.AtomArray:
    terminal_atom_names: set[str] = set()
    if has_previous:
        terminal_atom_names.add("H2")
    if has_next:
        terminal_atom_names.update(("OXT", "HXT"))

    prepared = bp5[~np.isin(bp5.atom_name, list(terminal_atom_names))].copy()
    prepared.chain_id = np.full(prepared.array_length(), chain_id, dtype="U4")
    prepared.res_id = np.full(prepared.array_length(), residue_id, dtype=int)
    prepared.set_annotation(
        "atom_id",
        np.arange(
            starting_atom_id,
            starting_atom_id + prepared.array_length(),
            dtype=int,
        ),
    )
    return prepared


def _require_atom_names(atom_array: struc.AtomArray, atom_names: tuple[str, ...]) -> None:
    available = set(atom_array.atom_name.tolist())
    missing = sorted(set(atom_names) - available)
    if missing:
        raise ValueError(f"atom_array is missing required atom names: {missing}")


def _atom_coord(atom_array: struc.AtomArray, atom_name: str) -> np.ndarray:
    atom_indices = np.flatnonzero(atom_array.atom_name == atom_name)
    if atom_indices.size != 1:
        raise ValueError(f"expected exactly one atom named {atom_name!r}")
    return atom_array.coord[int(atom_indices[0])]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm
