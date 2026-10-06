from __future__ import annotations

import json
from pathlib import Path

from biotite.structure.io import load_structure

from swacanatase.protein_generator import export_cyclic_bp5_backbones


def test_export_cyclic_bp5_backbones_from_m18_theozyme(tmp_path: Path) -> None:
    spike = export_cyclic_bp5_backbones(
        Path("data/generated/theozyme/nanoring_M18_bp5.cif"),
        tmp_path,
    )

    exported = load_structure(spike.input_pdb_path)
    assert spike.symmetry_order == 9
    assert exported.array_length() == 36
    assert set(exported.res_name) == {"ALA"}
    assert set(exported.chain_id) == {"A"}
    assert exported.res_id[::4].tolist() == [
        1,
        201,
        401,
        601,
        801,
        1001,
        1201,
        1401,
        1601,
    ]
    assert exported.atom_name[:4].tolist() == ["N", "CA", "C", "O"]
    assert spike.input_pdb_path.read_text().startswith("ATOM")

    manifest = json.loads(spike.manifest_path.read_text())
    assert manifest["symmetry_order"] == 9
    assert manifest["source_bp5_residue_ids"] == list(range(1, 10))
    assert manifest["contigs"] == [
        "A1-1",
        "12",
        "A201-201",
        "12",
        "A401-401",
        "12",
        "A601-601",
        "12",
        "A801-801",
        "12",
        "A1001-1001",
        "12",
        "A1201-1201",
        "12",
        "A1401-1401",
        "12",
        "A1601-1601",
        "12",
    ]
    assert "--predict_symmetric" in manifest["protein_generator_command"]
