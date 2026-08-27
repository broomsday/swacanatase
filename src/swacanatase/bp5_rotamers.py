from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc

from .ligands import DEFAULT_LIGAND_DIR, load_bp5_bond_pairs
from .torsions import measure_dihedral, set_dihedral_by_rotating_component

BP5_CHI1_ATOMS = ("N", "CA", "C12", "C9")
BP5_CHI2_ATOMS = ("CA", "C12", "C9", "C8")
BP5_INTER_RING_TORSION_ATOMS = ("N1", "C3", "C6", "N2")
BP5_BIPYRIDINE_RING_ATOMS = (
    "N1",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "N2",
    "C6",
    "C7",
    "C8",
    "C9",
    "C11",
)


@dataclass(frozen=True)
class BP5ChiRotamer:
    name: str
    chi1_degrees: float
    chi2_degrees: float
    weight: float | None = None


@dataclass(frozen=True)
class BP5RotamerPlacement:
    residue_id: int
    rotamer: BP5ChiRotamer
    atom_array: struc.AtomArray
    chi1_degrees: float
    chi2_degrees: float
    clash_score: float


DEFAULT_BP5_CHI_ROTAMERS: tuple[BP5ChiRotamer, ...] = (
    BP5ChiRotamer("gminus_m90", -60.0, -90.0),
    BP5ChiRotamer("gminus_p90", -60.0, 90.0),
    BP5ChiRotamer("gplus_m90", 60.0, -90.0),
    BP5ChiRotamer("gplus_p90", 60.0, 90.0),
    BP5ChiRotamer("trans_m90", 180.0, -90.0),
    BP5ChiRotamer("trans_p90", 180.0, 90.0),
)


def enumerate_bp5_chi_rotamers(
    atom_array: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]] | None = None,
    rotamers: Iterable[BP5ChiRotamer] = DEFAULT_BP5_CHI_ROTAMERS,
    residue_id: int | None = None,
    clash_score: float = 0.0,
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
) -> tuple[BP5RotamerPlacement, ...]:
    """Enumerate deterministic chi1/chi2 rotamers from a fixed active-site unit."""
    if bond_pairs is None:
        bond_pairs = load_bp5_bond_pairs(cif_path=cif_path)
    bond_pairs = tuple(bond_pairs)
    if residue_id is None:
        residue_ids = set(atom_array.res_id.tolist())
        if len(residue_ids) != 1:
            raise ValueError("residue_id is required when atom_array has multiple res_ids")
        residue_id = int(next(iter(residue_ids)))

    placements: list[BP5RotamerPlacement] = []
    for rotamer in rotamers:
        placed = apply_bp5_chi_rotamer(
            atom_array=atom_array,
            bond_pairs=bond_pairs,
            rotamer=rotamer,
        )
        placements.append(
            BP5RotamerPlacement(
                residue_id=residue_id,
                rotamer=rotamer,
                atom_array=placed,
                chi1_degrees=measure_dihedral(placed, BP5_CHI1_ATOMS),
                chi2_degrees=measure_dihedral(placed, BP5_CHI2_ATOMS),
                clash_score=clash_score,
            )
        )
    return tuple(placements)


def apply_bp5_chi_rotamer(
    atom_array: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
    rotamer: BP5ChiRotamer,
) -> struc.AtomArray:
    """Apply one BP5 chi rotamer while preserving the fixed bipyridine/Pd frame."""
    chi2_set = set_dihedral_by_rotating_component(
        atom_array=atom_array,
        bond_pairs=bond_pairs,
        atom_names=BP5_CHI2_ATOMS,
        target_degrees=rotamer.chi2_degrees,
        moving_side_atom="C12",
    )
    return set_dihedral_by_rotating_component(
        atom_array=chi2_set,
        bond_pairs=bond_pairs,
        atom_names=BP5_CHI1_ATOMS,
        target_degrees=rotamer.chi1_degrees,
        moving_side_atom="CA",
    )
