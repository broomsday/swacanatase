from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import biotite.structure as struc
import numpy as np


def measure_dihedral(atom_array: struc.AtomArray, atom_names: Sequence[str]) -> float:
    """Measure a signed dihedral angle in degrees."""
    if len(atom_names) != 4:
        raise ValueError("atom_names must contain exactly four atom names")

    name_to_index = _atom_indices(atom_array)
    p0, p1, p2, p3 = [
        atom_array.coord[name_to_index[atom_name]] for atom_name in atom_names
    ]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def set_dihedral_by_rotating_component(
    atom_array: struc.AtomArray,
    bond_pairs: Iterable[tuple[str, str]],
    atom_names: Sequence[str],
    target_degrees: float,
    moving_side_atom: str,
) -> struc.AtomArray:
    """Set a dihedral by rotating one bond-graph component around the central bond."""
    if len(atom_names) != 4:
        raise ValueError("atom_names must contain exactly four atom names")

    atom_1, axis_start_atom, axis_end_atom, atom_4 = tuple(atom_names)
    component = connected_component_after_removing_bond(
        bond_pairs=bond_pairs,
        blocked_bond=(axis_start_atom, axis_end_atom),
        seed_atom=moving_side_atom,
    )
    if moving_side_atom not in component:
        raise ValueError(f"moving_side_atom {moving_side_atom!r} is not in moving component")
    if atom_1 not in component and atom_4 not in component:
        raise ValueError(
            "moving component must contain one terminal atom from the requested dihedral"
        )

    current_degrees = measure_dihedral(atom_array, atom_names)
    delta_degrees = _signed_angle_delta(target_degrees, current_degrees)
    if atom_1 in component and atom_4 not in component:
        rotation_degrees = -delta_degrees
    elif atom_4 in component and atom_1 not in component:
        rotation_degrees = delta_degrees
    else:
        raise ValueError(
            "moving component must contain exactly one terminal atom from the dihedral"
        )

    rotated = atom_array.copy()
    name_to_index = _atom_indices(rotated)
    moving_atom_names = sorted(component - {axis_start_atom, axis_end_atom})
    moving_indices = np.array([name_to_index[name] for name in moving_atom_names], dtype=int)
    if moving_indices.size == 0:
        return rotated

    rotated.coord[moving_indices] = rotate_about_axis(
        points=rotated.coord[moving_indices],
        axis_start=rotated.coord[name_to_index[axis_start_atom]],
        axis_end=rotated.coord[name_to_index[axis_end_atom]],
        angle_radians=np.radians(rotation_degrees),
    )
    return rotated


def connected_component_after_removing_bond(
    bond_pairs: Iterable[tuple[str, str]],
    blocked_bond: tuple[str, str],
    seed_atom: str,
) -> set[str]:
    """Return the atom-name component reachable from seed after removing one bond."""
    blocked = frozenset(blocked_bond)
    graph: dict[str, set[str]] = defaultdict(set)
    for atom_1, atom_2 in bond_pairs:
        if frozenset((atom_1, atom_2)) == blocked:
            continue
        graph[atom_1].add(atom_2)
        graph[atom_2].add(atom_1)

    seen = {seed_atom}
    stack = [seed_atom]
    while stack:
        atom_name = stack.pop()
        for neighbor in graph.get(atom_name, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def rotate_about_axis(
    points: np.ndarray,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    """Rotate points around an axis with Rodrigues' rotation formula."""
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


def _atom_indices(atom_array: struc.AtomArray) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, atom_name in enumerate(atom_array.atom_name.tolist()):
        if atom_name in indices:
            raise ValueError(f"Duplicate atom name {atom_name!r}")
        indices[atom_name] = index
    return indices


def _signed_angle_delta(target_degrees: float, current_degrees: float) -> float:
    return float((target_degrees - current_degrees + 180.0) % 360.0 - 180.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / norm
