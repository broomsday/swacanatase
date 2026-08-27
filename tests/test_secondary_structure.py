from pathlib import Path

import numpy as np

from swacanatase.bp5_rotamers import enumerate_bp5_chi_rotamers
from swacanatase.placement import place_bp5_sidechains_around_nanoring
from swacanatase.secondary_structure import (
    CA_C_BOND_LENGTH,
    CA_C_N_ANGLE_DEGREES,
    C_N_BOND_LENGTH,
    C_N_CA_ANGLE_DEGREES,
    N_CA_BOND_LENGTH,
    N_CA_C_ANGLE_DEGREES,
    SECONDARY_STRUCTURE_TARGETS,
    build_regular_secondary_structure_segment,
    score_secondary_structure_segment_clashes,
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


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)
