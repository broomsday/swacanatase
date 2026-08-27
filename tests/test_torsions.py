from pathlib import Path

import numpy as np

from swacanatase.active_site import build_bp5_palladium_active_site
from swacanatase.bp5_rotamers import (
    BP5_CHI1_ATOMS,
    BP5_CHI2_ATOMS,
    BP5_INTER_RING_TORSION_ATOMS,
    BP5_BIPYRIDINE_RING_ATOMS,
    BP5ChiRotamer,
    apply_bp5_chi_rotamer,
)
from swacanatase.ligands import load_bp5_bond_pairs
from swacanatase.torsions import (
    connected_component_after_removing_bond,
    measure_dihedral,
    set_dihedral_by_rotating_component,
)


def test_connected_component_after_removing_bond_selects_chi2_backbone_side() -> None:
    bond_pairs = load_bp5_bond_pairs(Path("data/rcsb/BP5.cif"))

    component = connected_component_after_removing_bond(
        bond_pairs=bond_pairs,
        blocked_bond=("C12", "C9"),
        seed_atom="C12",
    )

    assert {"C12", "CA", "N", "C", "O", "OXT"}.issubset(component)
    assert {"C9", "C8", "C11", "N1", "N2"}.isdisjoint(component)


def test_setting_chi1_preserves_fixed_active_site_atoms() -> None:
    active_site = build_bp5_palladium_active_site(Path("data/rcsb/BP5.cif"))
    bond_pairs = load_bp5_bond_pairs(Path("data/rcsb/BP5.cif"))
    fixed_atom_names = ("C12", "C9", "N1", "N2", "PD", "CV1", "CV2")
    fixed_before = _coords_by_name(active_site, fixed_atom_names)

    rotated = set_dihedral_by_rotating_component(
        atom_array=active_site,
        bond_pairs=bond_pairs,
        atom_names=BP5_CHI1_ATOMS,
        target_degrees=-60.0,
        moving_side_atom="CA",
    )

    assert _angle_close(measure_dihedral(rotated, BP5_CHI1_ATOMS), -60.0)
    assert _coords_by_name(rotated, fixed_atom_names) == fixed_before


def test_setting_chi2_preserves_fixed_active_site_atoms() -> None:
    active_site = build_bp5_palladium_active_site(Path("data/rcsb/BP5.cif"))
    bond_pairs = load_bp5_bond_pairs(Path("data/rcsb/BP5.cif"))
    fixed_atom_names = ("C12", "C9", "N1", "N2", "PD", "CV1", "CV2")
    fixed_before = _coords_by_name(active_site, fixed_atom_names)

    rotated = set_dihedral_by_rotating_component(
        atom_array=active_site,
        bond_pairs=bond_pairs,
        atom_names=BP5_CHI2_ATOMS,
        target_degrees=90.0,
        moving_side_atom="C12",
    )

    assert _angle_close(measure_dihedral(rotated, BP5_CHI2_ATOMS), 90.0)
    assert _coords_by_name(rotated, fixed_atom_names) == fixed_before


def test_applying_chi2_then_chi1_reaches_both_targets_and_preserves_bipyridine() -> None:
    active_site = build_bp5_palladium_active_site(Path("data/rcsb/BP5.cif"))
    bond_pairs = load_bp5_bond_pairs(Path("data/rcsb/BP5.cif"))

    rotated = apply_bp5_chi_rotamer(
        atom_array=active_site,
        bond_pairs=bond_pairs,
        rotamer=BP5ChiRotamer("test", 180.0, -90.0),
    )

    assert _angle_close(measure_dihedral(rotated, BP5_CHI1_ATOMS), 180.0)
    assert _angle_close(measure_dihedral(rotated, BP5_CHI2_ATOMS), -90.0)
    assert abs(measure_dihedral(rotated, BP5_INTER_RING_TORSION_ATOMS)) < 0.1
    assert _bipyridine_plane_singular_value(rotated) < 0.01


def _coords_by_name(atom_array, atom_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(np.round(atom_array.coord[_atom_index(atom_array, atom_name)], 8))
        for atom_name in atom_names
    )


def _angle_close(actual: float, expected: float, atol: float = 1e-6) -> bool:
    return abs((actual - expected + 180.0) % 360.0 - 180.0) <= atol


def _bipyridine_plane_singular_value(atom_array) -> float:
    points = np.array(
        [
            atom_array.coord[_atom_index(atom_array, atom_name)]
            for atom_name in BP5_BIPYRIDINE_RING_ATOMS
        ]
    )
    _, singular_values, _ = np.linalg.svd(points - points.mean(axis=0))
    return float(singular_values[-1])


def _atom_index(atom_array, atom_name: str) -> int:
    return atom_array.atom_name.tolist().index(atom_name)
