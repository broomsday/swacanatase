from pathlib import Path

import numpy as np

from swacanatase.bp5_rotamers import (
    BP5_CHI2_ATOMS,
    BP5_CHI2_VALIDATION_ATOMS,
    DEFAULT_BP5_CHI_ROTAMERS,
    enumerate_bp5_chi_rotamers,
)
from swacanatase.clashes import score_heavy_atom_clashes
from swacanatase.torsions import measure_dihedral
from swacanatase.ligands import load_bp5_bond_pairs
from swacanatase.placement import (
    BP5_VIRTUAL_CARBON_ATOMS,
    _score_bp5_arrays_against_nanoring,
    _without_atom_names,
    place_bp5_rotamer_ensembles_around_nanoring,
    place_bp5_sidechains_around_nanoring,
)


def test_default_bp5_rotamer_grid_has_six_deterministic_states() -> None:
    assert [
        (rotamer.name, rotamer.chi1_degrees, rotamer.chi2_degrees)
        for rotamer in DEFAULT_BP5_CHI_ROTAMERS
    ] == [
        ("gminus_m90", -60.0, -90.0),
        ("gminus_p90", -60.0, 90.0),
        ("gplus_m90", 60.0, -90.0),
        ("gplus_p90", 60.0, 90.0),
        ("trans_m90", 180.0, -90.0),
        ("trans_p90", 180.0, 90.0),
    ]


def test_enumerating_rotamers_preserves_anchor_fit_and_realizes_chi_targets() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )
    residue = placement.sidechains[placement.sidechains.res_id == 1]
    fixed_before = _coords_by_name(residue, ("C9", "N1", "N2", "PD", "CV1", "CV2"))

    rotamers = enumerate_bp5_chi_rotamers(
        atom_array=residue,
        bond_pairs=load_bp5_bond_pairs(Path("data/rcsb/BP5.cif")),
        residue_id=1,
    )

    assert len(rotamers) == 6
    for candidate, expected in zip(rotamers, DEFAULT_BP5_CHI_ROTAMERS, strict=True):
        assert candidate.residue_id == 1
        assert candidate.rotamer == expected
        assert _angle_close(candidate.chi1_degrees, expected.chi1_degrees)
        assert _angle_close(candidate.chi2_degrees, expected.chi2_degrees)
        assert (
            _coords_by_name(
                candidate.atom_array,
                ("C9", "N1", "N2", "PD", "CV1", "CV2"),
            )
            == fixed_before
        )
        assert candidate.atom_array.chain_id.tolist() == residue.chain_id.tolist()
        assert candidate.atom_array.res_id.tolist() == residue.res_id.tolist()
        assert candidate.atom_array.atom_id.tolist() == residue.atom_id.tolist()


def test_chi2_c11_branch_torsion_is_retained_as_validation_partner() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )
    residue = placement.sidechains[placement.sidechains.res_id == 1]
    rotamers = enumerate_bp5_chi_rotamers(residue, residue_id=1)
    expected_validation_offset = _signed_angle_delta(
        measure_dihedral(residue, BP5_CHI2_VALIDATION_ATOMS),
        measure_dihedral(residue, BP5_CHI2_ATOMS),
    )

    validation_offsets = {
        round(
            _signed_angle_delta(
                candidate.chi2_validation_degrees,
                candidate.chi2_degrees,
            ),
            6,
        )
        for candidate in rotamers
    }

    assert len(validation_offsets) == 1
    assert np.isclose(next(iter(validation_offsets)), expected_validation_offset)
    for candidate in rotamers:
        assert np.isclose(
            candidate.chi2_validation_degrees,
            measure_dihedral(candidate.atom_array, BP5_CHI2_VALIDATION_ATOMS),
        )


def test_nanoring_rotamer_ensemble_produces_six_raw_candidates_per_site() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )

    assert len(placement.anchor_pairs) == 9
    assert len(placement.rotamer_states) == 6
    assert len(placement.accepted_rotamer_states) == 6
    assert len(placement.rotamer_candidates) == 9 * 6
    assert len(placement.accepted_rotamer_candidates) == 9 * 6
    assert {candidate.residue_id for candidate in placement.rotamer_candidates} == set(
        range(1, 10)
    )
    assert all(
        candidate.clash_score >= 0.0 for candidate in placement.rotamer_candidates
    )
    for state in placement.rotamer_states:
        assert len(state.candidates) == 9
        assert state.sidechains.array_length() == 9 * 34
        assert {candidate.residue_id for candidate in state.candidates} == set(
            range(1, 10)
        )
        assert all(
            np.isclose(candidate.clash_score, state.clash_score)
            for candidate in state.candidates
        )
    state_scores = {
        state.rotamer_name: state.clash_score for state in placement.rotamer_states
    }
    assert state_scores["gminus_m90"] > state_scores["gplus_m90"]
    assert state_scores["gminus_p90"] > state_scores["gplus_p90"]


def test_nanoring_rotamer_ensemble_can_keep_top_k_symmetric_states() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        max_rotamers_per_site=2,
    )

    assert len(placement.rotamer_candidates) == 9 * 6
    assert [state.rotamer_name for state in placement.accepted_rotamer_states] == [
        "gplus_m90",
        "gplus_p90",
    ]
    assert len(placement.accepted_rotamer_candidates) == 9 * 2
    rotamer_names_by_residue = {
        residue_id: tuple(
            candidate.rotamer.name
            for candidate in placement.accepted_rotamer_candidates
            if candidate.residue_id == residue_id
        )
        for residue_id in range(1, 10)
    }
    assert len(set(rotamer_names_by_residue.values())) == 1
    for residue_id in range(1, 10):
        residue_scores = [
            candidate.clash_score
            for candidate in placement.accepted_rotamer_candidates
            if candidate.residue_id == residue_id
        ]
        assert residue_scores == sorted(residue_scores)


def test_bp5_nanoring_clash_score_ignores_virtual_atoms_and_pd_anchor_contacts() -> None:
    placement = place_bp5_sidechains_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
    )
    residue = placement.sidechains[placement.sidechains.res_id == 1]
    anchor_pair = placement.anchor_pairs[0]
    scored_residue = _without_atom_names(residue, BP5_VIRTUAL_CARBON_ATOMS)
    pd_index = scored_residue.atom_name.tolist().index("PD")
    ignored_anchor_pairs = tuple(
        (pd_index, anchor_index) for anchor_index in anchor_pair.atom_indices
    )

    placement_score = _score_bp5_arrays_against_nanoring(
        bp5_arrays=(scored_residue,),
        nanoring=placement.nanoring,
        anchor_pairs=(anchor_pair,),
    )
    manual_score = score_heavy_atom_clashes(
        scored_residue,
        other=placement.nanoring,
        ignored_atom_index_pairs=ignored_anchor_pairs,
        ignore_same_residue=True,
        ignore_inter_residue_backbone_n_c=True,
    ).total_overlap_score
    unfiltered_score = score_heavy_atom_clashes(
        residue,
        other=placement.nanoring,
    ).total_overlap_score

    assert np.isclose(placement_score, manual_score)
    assert placement_score < unfiltered_score


def test_nanoring_rotamer_ensemble_scores_secondary_structure_candidates() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        max_rotamers_per_site=1,
        secondary_structure="alpha_helix",
        residues_before=1,
        residues_after=1,
    )

    assert len(placement.accepted_rotamer_candidates) == 9
    assert len(placement.accepted_rotamer_states) == 1
    assert len(placement.secondary_structure_states) == 1
    assert len(placement.accepted_secondary_structure_states) == 1
    assert len(placement.secondary_structure_candidates) == 9
    assert len(placement.accepted_secondary_structure_candidates) == 9
    secondary_state = placement.secondary_structure_states[0]
    assert len(secondary_state.candidates) == 9
    assert secondary_state.cylinder_intrusion_score.passes
    assert secondary_state.segments.array_length() == sum(
        candidate.segment.atom_array.array_length()
        for candidate in secondary_state.candidates
    )
    assert len(set(secondary_state.segments.atom_id.tolist())) == (
        secondary_state.segments.array_length()
    )
    assert len(set(secondary_state.segments.res_id.tolist())) == 9 * 3
    assert np.isclose(
        secondary_state.clash_score,
        secondary_state.scaffold_clash_score
        + secondary_state.bp5_clash_score
        + secondary_state.neighboring_backbone_clash_score,
    )
    for candidate in placement.secondary_structure_candidates:
        assert candidate.segment.residues_before == 1
        assert candidate.segment.residues_after == 1
        assert candidate.scaffold_clash_score >= 0.0
        assert candidate.bp5_clash_score >= 0.0
        assert candidate.neighboring_backbone_clash_score >= 0.0
        assert candidate.cylinder_intrusion_score.passes
        assert np.isclose(candidate.clash_score, secondary_state.clash_score)
        assert np.isclose(
            candidate.clash_score,
            candidate.scaffold_clash_score
            + candidate.bp5_clash_score
            + candidate.neighboring_backbone_clash_score,
        )
        assert np.isclose(
            np.linalg.norm(
                candidate.orientation_metrics.secondary_structure_direction
            ),
            1.0,
        )
        assert np.allclose(
            candidate.secondary_structure_direction,
            candidate.orientation_metrics.secondary_structure_direction,
        )
        assert np.isclose(
            np.linalg.norm(candidate.orientation_metrics.n_terminal_exit_vector),
            1.0,
        )
        assert np.isclose(
            np.linalg.norm(candidate.orientation_metrics.c_terminal_exit_vector),
            1.0,
        )


def test_default_secondary_structure_clash_cutoff_filters_high_overlap_states() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        secondary_structure="alpha_helix",
    )

    accepted_names = {
        state.rotamer_name for state in placement.accepted_secondary_structure_states
    }

    assert len(placement.secondary_structure_states) == 6
    assert accepted_names == {"gminus_p90", "gplus_m90", "trans_m90"}
    assert all(
        state.clash_score / len(state.candidates) <= 6.0
        for state in placement.accepted_secondary_structure_states
    )
    assert all(
        state.cylinder_intrusion_score.passes
        for state in placement.accepted_secondary_structure_states
    )


def test_secondary_structure_cylinder_filter_can_be_disabled() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        secondary_structure="alpha_helix",
        secondary_structure_cylinder_filter=False,
    )

    accepted_by_name = {
        state.rotamer_name: state
        for state in placement.accepted_secondary_structure_states
    }

    assert "gplus_p90" in accepted_by_name
    assert not accepted_by_name["gplus_p90"].cylinder_intrusion_score.passes


def test_secondary_structure_candidates_keep_symmetric_rotamer_states() -> None:
    placement = place_bp5_rotamer_ensembles_around_nanoring(
        m=18,
        cif_path=Path("data/rcsb/BP5.cif"),
        max_rotamers_per_site=2,
        secondary_structure="beta_strand",
        residues_before=1,
        residues_after=0,
    )

    accepted_rotamer_names_by_residue = {
        residue_id: tuple(
            candidate.rotamer.name
            for candidate in placement.accepted_rotamer_candidates
            if candidate.residue_id == residue_id
        )
        for residue_id in range(1, 10)
    }
    secondary_rotamer_names_by_residue = {
        residue_id: tuple(
            candidate.rotamer_candidate.rotamer.name
            for candidate in placement.accepted_secondary_structure_candidates
            if candidate.rotamer_candidate.residue_id == residue_id
        )
        for residue_id in range(1, 10)
    }

    assert len(placement.secondary_structure_candidates) == 9 * 2
    assert len(set(accepted_rotamer_names_by_residue.values())) == 1
    assert secondary_rotamer_names_by_residue == accepted_rotamer_names_by_residue


def _coords_by_name(atom_array, atom_names: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(np.round(atom_array.coord[_atom_index(atom_array, atom_name)], 8))
        for atom_name in atom_names
    )


def _angle_close(actual: float, expected: float, atol: float = 1e-4) -> bool:
    return abs((actual - expected + 180.0) % 360.0 - 180.0) <= atol


def _signed_angle_delta(target: float, current: float) -> float:
    return float((target - current + 180.0) % 360.0 - 180.0)


def _atom_index(atom_array, atom_name: str) -> int:
    return atom_array.atom_name.tolist().index(atom_name)
