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
