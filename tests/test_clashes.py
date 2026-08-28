import biotite.structure as struc
import numpy as np

from swacanatase.clashes import score_heavy_atom_clashes


def test_heavy_atom_clash_score_detects_deliberate_overlap() -> None:
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([1, 1])
    atoms.atom_name = np.array(["C1", "C2"])
    atoms.element = np.array(["C", "C"])

    score = score_heavy_atom_clashes(atoms)

    assert score.clashing_pair_count == 1
    assert np.isclose(score.total_overlap_score, 1.2)
    assert np.isclose(score.worst_overlap, 1.2)
    assert score.worst_pair is not None


def test_heavy_atom_clash_score_ignores_bonded_pairs_and_hydrogens() -> None:
    atoms = struc.AtomArray(3)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A", "A"])
    atoms.res_id = np.array([1, 1, 1])
    atoms.atom_name = np.array(["C1", "C2", "H1"])
    atoms.element = np.array(["C", "C", "H"])

    score = score_heavy_atom_clashes(atoms, bonded_atom_pairs=(("C1", "C2"),))

    assert score.passes
    assert score.total_overlap_score == 0.0


def test_heavy_atom_clash_score_can_ignore_same_name_pairs() -> None:
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([1, 2])
    atoms.atom_name = np.array(["C", "C"])
    atoms.element = np.array(["C", "C"])

    score = score_heavy_atom_clashes(
        atoms,
        ignored_atom_name_pairs=(("C", "C"),),
    )

    assert score.passes
    assert score.total_overlap_score == 0.0


def test_heavy_atom_clash_score_handles_mixed_case_elements() -> None:
    atoms = struc.AtomArray(3)
    atoms.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ]
    )
    atoms.chain_id = np.array(["A", "A", "A"])
    atoms.res_id = np.array([1, 2, 3])
    atoms.atom_name = np.array(["PD", "C1", "H1"])
    atoms.element = np.array(["pd", "C", "h"])

    score = score_heavy_atom_clashes(atoms)

    assert score.passes
    assert score.total_overlap_score == 0.0


def test_heavy_atom_clash_score_can_ignore_same_residue_pairs() -> None:
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([1, 1])
    atoms.atom_name = np.array(["PD", "N1"])
    atoms.element = np.array(["PD", "N"])

    score = score_heavy_atom_clashes(atoms, ignore_same_residue=True)

    assert score.passes
    assert score.total_overlap_score == 0.0


def test_heavy_atom_clash_score_can_ignore_specific_cross_array_index_pairs() -> None:
    atoms = struc.AtomArray(1)
    atoms.coord = np.array([[0.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A"])
    atoms.res_id = np.array([1])
    atoms.atom_name = np.array(["PD"])
    atoms.element = np.array(["PD"])
    other = struc.AtomArray(2)
    other.coord = np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    other.chain_id = np.array(["B", "B"])
    other.res_id = np.array([1, 1])
    other.atom_name = np.array(["C1", "C2"])
    other.element = np.array(["C", "C"])

    score = score_heavy_atom_clashes(
        atoms,
        other=other,
        ignored_atom_index_pairs=((0, 1),),
    )

    assert score.clashing_pair_count == 1
    assert score.worst_pair is not None
    assert score.worst_pair.atom_2 == "B:1:C1"


def test_heavy_atom_clash_score_can_ignore_inter_residue_backbone_n_c_pairs() -> None:
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    atoms.chain_id = np.array(["A", "A"])
    atoms.res_id = np.array([1, 2])
    atoms.atom_name = np.array(["C", "N"])
    atoms.element = np.array(["C", "N"])

    score = score_heavy_atom_clashes(
        atoms,
        ignore_inter_residue_backbone_n_c=True,
    )

    assert score.passes
    assert score.total_overlap_score == 0.0
