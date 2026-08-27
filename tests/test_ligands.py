from pathlib import Path

import numpy as np

from swacanatase.ligands import load_bp5_atom_array


def test_load_bp5_atom_array_from_checked_in_ccd() -> None:
    atom_array = load_bp5_atom_array(Path("data/rcsb/BP5.cif"))

    assert atom_array.array_length() == 31
    assert atom_array.res_name.tolist() == ["BP5"] * 31
    assert atom_array.atom_name[:4].tolist() == ["C9", "C8", "C7", "C6"]
    assert set(np.unique(atom_array.element)) == {"C", "H", "N", "O"}
    assert atom_array.coord.shape == (31, 3)
