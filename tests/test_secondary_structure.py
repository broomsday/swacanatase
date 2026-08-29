from pathlib import Path

import biotite.structure as struc
import numpy as np

from swacanatase.bp5_rotamers import enumerate_bp5_chi_rotamers
from swacanatase.placement import place_bp5_sidechains_around_nanoring
from swacanatase.secondary_structure import (
    CA_C_BOND_LENGTH,
    CA_C_N_ANGLE_DEGREES,
    C_N_BOND_LENGTH,
    C_N_CA_ANGLE_DEGREES,
    N_H_BOND_LENGTH,
    N_CA_BOND_LENGTH,
    N_CA_C_ANGLE_DEGREES,
    OXT_HXT_BOND_LENGTH,
    RAMACHANDRAN_ALLOWED,
    RAMACHANDRAN_DISALLOWED,
    RAMACHANDRAN_FAVORED,
    SECONDARY_STRUCTURE_TARGETS,
    BackboneTorsionTargets,
    build_regular_secondary_structure_segment,
    measure_secondary_structure_orientation,
    phi_psi_grid_values,
    score_nanoring_cylinder_intrusions,
    score_secondary_structure_segment_clashes,
    secondary_structure_phi_psi_scan_matrix,
    secondary_structure_phi_psi_scan_targets,
)


def test_alpha_helix_and_beta_strand_builders_produce_requested_residue_counts() -> None:
    bp5_rotamer = _first_bp5_rotamer()

    helix = build_regular_secondary_structure_segment(
        bp5_rotamer,
        secondary_structure_type="alpha_helix",
        residues_before=2,
        residues_after=3,
    )
    strand = build_regular_secondary_structure_segment(
        bp5_rotamer,
        secondary_structure_type="beta_strand",
        residues_before=1,
        residues_after=1,
    )

    assert len(set(helix.atom_array.res_id.tolist())) == 6
    assert len(set(strand.atom_array.res_id.tolist())) == 3
    assert helix.bp5_residue_id == 3
    assert strand.bp5_residue_id == 2
    assert helix.atom_array[helix.atom_array.res_id == helix.bp5_residue_id].res_name[
        0
    ] == "BP5"
    assert set(helix.atom_array.chain_id.tolist()) == {"A"}


def test_generated_backbone_geometry_matches_regular_secondary_structure_targets() -> None:
    bp5_rotamer = _first_bp5_rotamer()

    for secondary_structure_type in ("alpha_helix", "beta_strand"):
        segment = build_regular_secondary_structure_segment(
            bp5_rotamer,
            secondary_structure_type=secondary_structure_type,
            residues_before=2,
            residues_after=2,
        )
        targets = SECONDARY_STRUCTURE_TARGETS[secondary_structure_type]

        for residue_id in range(2, 5):
            residue = _residue(segment.atom_array, residue_id)
            previous_residue = _residue(segment.atom_array, residue_id - 1)
            next_residue = _residue(segment.atom_array, residue_id + 1)

            if residue.res_name[0] == "GLY":
                assert np.isclose(
                    _distance(residue, "N", residue, "CA"),
                    N_CA_BOND_LENGTH,
                )
                assert np.isclose(
                    _distance(residue, "CA", residue, "C"),
                    CA_C_BOND_LENGTH,
                )
                assert np.isclose(
                    _angle(residue, "N", residue, "CA", residue, "C"),
                    N_CA_C_ANGLE_DEGREES,
                )
            assert np.isclose(_distance(residue, "C", next_residue, "N"), C_N_BOND_LENGTH)
            assert np.isclose(
                _angle(previous_residue, "C", residue, "N", residue, "CA"),
                C_N_CA_ANGLE_DEGREES,
            )
            assert np.isclose(
                _angle(residue, "CA", residue, "C", next_residue, "N"),
                CA_C_N_ANGLE_DEGREES,
            )
            assert _angle_close(
                _dihedral(previous_residue, "C", residue, "N", residue, "CA", residue, "C"),
                targets.phi_degrees,
            )
            assert _angle_close(
                _dihedral(residue, "N", residue, "CA", residue, "C", next_residue, "N"),
                targets.psi_degrees,
            )


def test_phi_psi_scan_matrix_is_sparse_on_a_5_degree_grid() -> None:
    matrix = secondary_structure_phi_psi_scan_matrix(
        "alpha_helix",
        step_degrees=5.0,
    )
    grid_values = phi_psi_grid_values(step_degrees=5.0)
    phi_index = grid_values.tolist().index(-60.0)
    psi_index = grid_values.tolist().index(-45.0)

    assert matrix.shape == (72, 72)
    assert matrix[phi_index, psi_index] == RAMACHANDRAN_FAVORED
    assert matrix[0, 0] == RAMACHANDRAN_DISALLOWED
    assert np.count_nonzero(matrix == RAMACHANDRAN_ALLOWED) > 0
    assert np.count_nonzero(matrix) < matrix.size / 5


def test_phi_psi_scan_matrix_uses_conservative_favored_cores() -> None:
    grid_values = phi_psi_grid_values(step_degrees=5.0)
    grid_index = {value: index for index, value in enumerate(grid_values.tolist())}
    alpha_matrix = secondary_structure_phi_psi_scan_matrix(
        "alpha_helix",
        step_degrees=5.0,
    )
    beta_matrix = secondary_structure_phi_psi_scan_matrix(
        "beta_strand",
        step_degrees=5.0,
    )

    assert np.count_nonzero(alpha_matrix == RAMACHANDRAN_FAVORED) == 29
    assert np.count_nonzero(beta_matrix == RAMACHANDRAN_FAVORED) == 113
    assert alpha_matrix[grid_index[-60.0], grid_index[-70.0]] == RAMACHANDRAN_ALLOWED
    assert beta_matrix[grid_index[-170.0], grid_index[135.0]] == RAMACHANDRAN_ALLOWED


def test_phi_psi_scan_targets_are_ordered_from_ideal_center_outward() -> None:
    targets = secondary_structure_phi_psi_scan_targets(
        "beta_strand",
        step_degrees=5.0,
        ramachandran_level="favored",
    )

    assert targets[0] == BackboneTorsionTargets(
        phi_degrees=-135.0,
        psi_degrees=135.0,
        label="phi_m135_psi_p135",
        ramachandran_level="favored",
    )
    assert 1 < len(targets) < 72 * 72
    assert all(target.ramachandran_level == "favored" for target in targets)
    assert all(target.phi_degrees % 5.0 == 0.0 for target in targets)
    assert all(target.psi_degrees % 5.0 == 0.0 for target in targets)


def test_custom_phi_psi_targets_drive_secondary_structure_growth() -> None:
    bp5_rotamer = _first_bp5_rotamer()
    target = BackboneTorsionTargets(
        phi_degrees=-65.0,
        psi_degrees=-40.0,
        label="phi_m065_psi_m040",
        ramachandran_level="favored",
    )

    segment = build_regular_secondary_structure_segment(
        bp5_rotamer,
        secondary_structure_type="alpha_helix",
        residues_before=2,
        residues_after=2,
        torsion_targets=target,
    )

    assert segment.torsion_targets == target
    for residue_id in range(2, 5):
        residue = _residue(segment.atom_array, residue_id)
        previous_residue = _residue(segment.atom_array, residue_id - 1)
        next_residue = _residue(segment.atom_array, residue_id + 1)
        assert _angle_close(
            _dihedral(previous_residue, "C", residue, "N", residue, "CA", residue, "C"),
            target.phi_degrees,
        )
        assert _angle_close(
            _dihedral(residue, "N", residue, "CA", residue, "C", next_residue, "N"),
            target.psi_degrees,
        )


def test_bp5_residue_frame_remains_fixed_after_segment_growth() -> None:
    bp5_rotamer = _first_bp5_rotamer()
    original_frame = _coords_by_name(bp5_rotamer.atom_array, ("N", "CA", "C", "O"))

    segment = build_regular_secondary_structure_segment(
        bp5_rotamer,
        secondary_structure_type="alpha_helix",
        residues_before=2,
        residues_after=2,
    )
    bp5_residue = _residue(segment.atom_array, segment.bp5_residue_id)

    assert _coords_by_name(bp5_residue, ("N", "CA", "C", "O")) == original_frame
    assert "OXT" not in bp5_residue.atom_name.tolist()
    assert "HXT" not in bp5_residue.atom_name.tolist()
    assert "H2" not in bp5_residue.atom_name.tolist()


def test_generated_peptide_terminals_are_normalized() -> None:
    segment = build_regular_secondary_structure_segment(
        _first_bp5_rotamer(),
        secondary_structure_type="alpha_helix",
        residues_before=1,
        residues_after=1,
    )

    n_terminal_residue = _residue(segment.atom_array, 1)
    bp5_residue = _residue(segment.atom_array, segment.bp5_residue_id)
    c_terminal_residue = _residue(segment.atom_array, 3)

    assert {"H", "H2"}.issubset(n_terminal_residue.atom_name.tolist())
    assert "OXT" not in n_terminal_residue.atom_name.tolist()
    assert "H" in bp5_residue.atom_name.tolist()
    assert "H2" not in bp5_residue.atom_name.tolist()
    assert "OXT" not in bp5_residue.atom_name.tolist()
    assert "HXT" not in bp5_residue.atom_name.tolist()
    assert "H" in c_terminal_residue.atom_name.tolist()
    assert {"OXT", "HXT"}.issubset(c_terminal_residue.atom_name.tolist())

    assert np.isclose(
        _distance(n_terminal_residue, "N", n_terminal_residue, "H"),
        N_H_BOND_LENGTH,
    )
    assert np.isclose(
        _distance(c_terminal_residue, "OXT", c_terminal_residue, "HXT"),
        OXT_HXT_BOND_LENGTH,
    )
    assert segment.atom_array.atom_id.tolist() == list(
        range(1, segment.atom_array.array_length() + 1)
    )


def test_neighboring_secondary_structure_backbone_clashes_are_scored() -> None:
    segment = build_regular_secondary_structure_segment(
        _first_bp5_rotamer(),
        secondary_structure_type="alpha_helix",
        residues_before=1,
        residues_after=1,
    )

    score = score_secondary_structure_segment_clashes(
        segment,
        neighboring_segments=(segment,),
    )

    assert score.scaffold_score == 0.0
    assert score.bp5_score == 0.0
    assert score.neighboring_backbone_score > 0.0
    assert score.total_overlap_score == score.neighboring_backbone_score
    assert not score.passes


def test_nanoring_cylinder_intrusion_score_counts_atoms_inside_finite_volume() -> None:
    nanoring = struc.AtomArray(4)
    nanoring.coord = np.array(
        [
            [5.0, 0.0, -1.0],
            [0.0, 5.0, -1.0],
            [5.0, 0.0, 1.0],
            [0.0, 5.0, 1.0],
        ],
        dtype=float,
    )
    nanoring.element = np.array(["C", "C", "C", "C"], dtype="U2")
    atoms = struc.AtomArray(5)
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [6.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    score = score_nanoring_cylinder_intrusions(atoms, nanoring)

    assert score.radius == 5.0
    assert score.z_min == -1.0
    assert score.z_max == 1.0
    assert score.intruding_atom_count == 2
    assert score.total_intrusion_depth == 6.0
    assert score.max_intrusion_depth == 5.0
    assert not score.passes


def test_secondary_structure_orientation_reports_frame_and_exit_vectors() -> None:
    segment = build_regular_secondary_structure_segment(
        _first_bp5_rotamer(),
        secondary_structure_type="alpha_helix",
        residues_before=2,
        residues_after=2,
    )

    metrics = measure_secondary_structure_orientation(
        segment,
        radial_direction=np.array([1.0, 0.0, 0.0]),
        tangential_direction=np.array([0.0, 1.0, 0.0]),
        ring_axis=np.array([0.0, 0.0, 1.0]),
    )
    bp5_residue = _residue(segment.atom_array, segment.bp5_residue_id)
    n_terminal_residue = _residue(segment.atom_array, 1)
    c_terminal_residue = _residue(segment.atom_array, 5)
    expected_n_exit = _unit(
        _atom_coord(n_terminal_residue, "CA") - _atom_coord(bp5_residue, "CA")
    )
    expected_c_exit = _unit(
        _atom_coord(c_terminal_residue, "CA") - _atom_coord(bp5_residue, "CA")
    )

    assert np.isclose(np.linalg.norm(metrics.secondary_structure_direction), 1.0)
    assert np.isclose(np.linalg.norm(metrics.n_terminal_exit_vector), 1.0)
    assert np.isclose(np.linalg.norm(metrics.c_terminal_exit_vector), 1.0)
    assert np.isclose(
        metrics.radial_alignment**2
        + metrics.tangential_alignment**2
        + metrics.axial_alignment**2,
        1.0,
    )
    assert np.allclose(metrics.n_terminal_exit_vector, expected_n_exit)
    assert np.allclose(metrics.c_terminal_exit_vector, expected_c_exit)


def _first_bp5_rotamer():
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )
    residue = placement.sidechains[placement.sidechains.res_id == 1]
    return enumerate_bp5_chi_rotamers(residue, residue_id=1)[0]


def _residue(atom_array, residue_id: int):
    return atom_array[atom_array.res_id == residue_id]


def _coords_by_name(atom_array, atom_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(np.round(atom_array.coord[_atom_index(atom_array, atom_name)], 8))
        for atom_name in atom_names
    )


def _distance(atom_array_1, atom_name_1: str, atom_array_2, atom_name_2: str) -> float:
    return float(
        np.linalg.norm(
            atom_array_1.coord[_atom_index(atom_array_1, atom_name_1)]
            - atom_array_2.coord[_atom_index(atom_array_2, atom_name_2)]
        )
    )


def _angle(atom_array_1, atom_name_1: str, atom_array_2, atom_name_2: str, atom_array_3, atom_name_3: str) -> float:
    coord_1 = atom_array_1.coord[_atom_index(atom_array_1, atom_name_1)]
    coord_2 = atom_array_2.coord[_atom_index(atom_array_2, atom_name_2)]
    coord_3 = atom_array_3.coord[_atom_index(atom_array_3, atom_name_3)]
    vector_1 = _unit(coord_1 - coord_2)
    vector_2 = _unit(coord_3 - coord_2)
    return float(np.degrees(np.arccos(np.clip(np.dot(vector_1, vector_2), -1, 1))))


def _dihedral(atom_array_1, atom_name_1: str, atom_array_2, atom_name_2: str, atom_array_3, atom_name_3: str, atom_array_4, atom_name_4: str) -> float:
    p0 = atom_array_1.coord[_atom_index(atom_array_1, atom_name_1)]
    p1 = atom_array_2.coord[_atom_index(atom_array_2, atom_name_2)]
    p2 = atom_array_3.coord[_atom_index(atom_array_3, atom_name_3)]
    p3 = atom_array_4.coord[_atom_index(atom_array_4, atom_name_4)]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = _unit(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _angle_close(actual: float, expected: float, atol: float = 1e-4) -> bool:
    return abs((actual - expected + 180.0) % 360.0 - 180.0) <= atol


def _atom_index(atom_array, atom_name: str) -> int:
    return atom_array.atom_name.tolist().index(atom_name)


def _atom_coord(atom_array, atom_name: str) -> np.ndarray:
    return atom_array.coord[_atom_index(atom_array, atom_name)]


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
