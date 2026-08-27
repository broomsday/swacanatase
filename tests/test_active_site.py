from pathlib import Path

import numpy as np

from swacanatase.active_site import (
    BP5_DONOR_ATOMS,
    BP5_INTER_RING_BOND,
    build_bp5_palladium_active_site,
    describe_square_planar_geometry,
    load_active_bp5_atom_array,
)
from swacanatase.ligands import load_bp5_atom_array


def test_active_bp5_rotates_bipyridine_nitrogens_to_same_side() -> None:
    source = load_bp5_atom_array(Path("data/rcsb/BP5.cif"))
    active = load_active_bp5_atom_array(Path("data/rcsb/BP5.cif"))

    assert abs(_dihedral(source, "N1", "C3", "C6", "N2")) > 179
    assert abs(_dihedral(active, "N1", "C3", "C6", "N2")) < 0.1
    assert _same_side_of_inter_ring_bond(active, *BP5_DONOR_ATOMS)


def test_active_bp5_keeps_bipyridine_rings_coplanar() -> None:
    active = load_active_bp5_atom_array(Path("data/rcsb/BP5.cif"))
    ring_atoms = [
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
    ]
    points = np.array(
        [active.coord[_atom_index(active, atom_name)] for atom_name in ring_atoms]
    )
    centered = points - points.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered)

    assert singular_values[-1] < 0.01


def test_build_bp5_palladium_active_site_adds_square_planar_pseudo_atoms() -> None:
    active_site = build_bp5_palladium_active_site(Path("data/rcsb/BP5.cif"))
    geometry = describe_square_planar_geometry(active_site)
    names = active_site.atom_name.tolist()

    assert active_site.array_length() == 34
    assert names[-3:] == ["PD", "CV1", "CV2"]
    assert active_site.element[-3:].tolist() == ["PD", "C", "C"]
    assert np.isclose(geometry.pd_n_bond_length, 2.018)
    assert np.isclose(geometry.pd_c_bond_length, 2.02)
    assert (
        abs(_point_plane_distance(active_site, "PD", ["N1", "N2", "CV1", "CV2"]))
        < 1e-6
    )
    assert abs(_angle(active_site, "N1", "PD", "CV2") - 180) < 1e-5
    assert abs(_angle(active_site, "N2", "PD", "CV1") - 180) < 1e-5


def _dihedral(atom_array, atom_1: str, atom_2: str, atom_3: str, atom_4: str) -> float:
    p0, p1, p2, p3 = [
        atom_array.coord[_atom_index(atom_array, atom)]
        for atom in (atom_1, atom_2, atom_3, atom_4)
    ]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _same_side_of_inter_ring_bond(atom_array, atom_1: str, atom_2: str) -> bool:
    c3 = atom_array.coord[_atom_index(atom_array, BP5_INTER_RING_BOND[0])]
    c6 = atom_array.coord[_atom_index(atom_array, BP5_INTER_RING_BOND[1])]
    n1 = atom_array.coord[_atom_index(atom_array, atom_1)]
    n2 = atom_array.coord[_atom_index(atom_array, atom_2)]
    ring_atoms = [
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
    ]
    points = np.array(
        [atom_array.coord[_atom_index(atom_array, atom_name)] for atom_name in ring_atoms]
    )
    _, _, vh = np.linalg.svd(points - points.mean(axis=0))
    normal = vh[-1]
    side_axis = np.cross(normal, c6 - c3)
    return bool(np.dot(n1 - c3, side_axis) * np.dot(n2 - c6, side_axis) > 0)


def _point_plane_distance(atom_array, atom_name: str, plane_atom_names: list[str]) -> float:
    point = atom_array.coord[_atom_index(atom_array, atom_name)]
    plane_points = np.array(
        [atom_array.coord[_atom_index(atom_array, name)] for name in plane_atom_names]
    )
    _, _, vh = np.linalg.svd(plane_points - plane_points.mean(axis=0))
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return float(np.dot(point - plane_points.mean(axis=0), normal))


def _angle(atom_array, atom_1: str, vertex: str, atom_2: str) -> float:
    p1 = atom_array.coord[_atom_index(atom_array, atom_1)]
    pv = atom_array.coord[_atom_index(atom_array, vertex)]
    p2 = atom_array.coord[_atom_index(atom_array, atom_2)]
    v1 = (p1 - pv) / np.linalg.norm(p1 - pv)
    v2 = (p2 - pv) / np.linalg.norm(p2 - pv)
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1, 1))))


def _atom_index(atom_array, atom_name: str) -> int:
    return atom_array.atom_name.tolist().index(atom_name)
