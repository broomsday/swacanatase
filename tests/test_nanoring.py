from swacanatase.nanoring import generate_armchair_nanoring


def test_generate_armchair_nanoring_uses_tuber() -> None:
    atom_array = generate_armchair_nanoring(n=3, units=1)

    assert atom_array.array_length() > 0
    assert set(atom_array.res_name) == {"CNT"}
    assert set(atom_array.element) == {"C"}
