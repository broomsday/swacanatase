from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np

BP5_COMPONENT_ID = "BP5"
RCSB_LIGAND_DOWNLOAD_BASE = "https://files.rcsb.org/ligands/download"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIGAND_DIR = PROJECT_ROOT / "data" / "rcsb"


@dataclass(frozen=True)
class RCSBLigandPaths:
    component_id: str
    definition_cif: Path
    ideal_sdf: Path


def download_bp5(
    data_dir: str | Path = DEFAULT_LIGAND_DIR,
    overwrite: bool = False,
) -> RCSBLigandPaths:
    return download_rcsb_ligand(
        component_id=BP5_COMPONENT_ID,
        data_dir=data_dir,
        overwrite=overwrite,
    )


def download_rcsb_ligand(
    component_id: str,
    data_dir: str | Path = DEFAULT_LIGAND_DIR,
    overwrite: bool = False,
) -> RCSBLigandPaths:
    component_id = component_id.upper()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    definition_cif = data_dir / f"{component_id}.cif"
    ideal_sdf = data_dir / f"{component_id}_ideal.sdf"

    _download_if_needed(
        url=f"{RCSB_LIGAND_DOWNLOAD_BASE}/{component_id}.cif",
        output_path=definition_cif,
        overwrite=overwrite,
    )
    _download_if_needed(
        url=f"{RCSB_LIGAND_DOWNLOAD_BASE}/{component_id}_ideal.sdf",
        output_path=ideal_sdf,
        overwrite=overwrite,
    )

    return RCSBLigandPaths(
        component_id=component_id,
        definition_cif=definition_cif,
        ideal_sdf=ideal_sdf,
    )


def load_bp5_atom_array(
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
    coordinate_set: str = "ideal",
) -> struc.AtomArray:
    return load_chemical_component_atom_array(
        cif_path=cif_path,
        component_id=BP5_COMPONENT_ID,
        coordinate_set=coordinate_set,
    )


def load_bp5_bond_pairs(
    cif_path: str | Path = DEFAULT_LIGAND_DIR / "BP5.cif",
) -> list[tuple[str, str]]:
    return load_chemical_component_bond_pairs(
        cif_path=cif_path,
        component_id=BP5_COMPONENT_ID,
    )


def load_chemical_component_atom_array(
    cif_path: str | Path,
    component_id: str,
    coordinate_set: str = "ideal",
) -> struc.AtomArray:
    """Load an RCSB CCD ligand CIF into a Biotite AtomArray."""
    component_id = component_id.upper()
    category = pdbx.CIFFile.read(str(cif_path))[component_id]["chem_comp_atom"]

    if coordinate_set == "ideal":
        x_name = "pdbx_model_Cartn_x_ideal"
        y_name = "pdbx_model_Cartn_y_ideal"
        z_name = "pdbx_model_Cartn_z_ideal"
    elif coordinate_set == "model":
        x_name = "model_Cartn_x"
        y_name = "model_Cartn_y"
        z_name = "model_Cartn_z"
    else:
        raise ValueError("coordinate_set must be 'ideal' or 'model'")

    atom_names = category["atom_id"].as_array(str)
    elements = np.char.upper(category["type_symbol"].as_array(str))
    coordinates = np.column_stack(
        (
            _required_float_column(category, x_name),
            _required_float_column(category, y_name),
            _required_float_column(category, z_name),
        )
    )

    atom_array = struc.AtomArray(len(atom_names))
    atom_array.coord = coordinates
    atom_array.chain_id = np.full(len(atom_names), "L", dtype="U4")
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.ins_code = np.full(len(atom_names), "", dtype="U1")
    atom_array.res_name = np.full(len(atom_names), component_id, dtype="U5")
    atom_array.hetero = np.ones(len(atom_names), dtype=bool)
    atom_array.atom_name = atom_names.astype("U6", copy=False)
    atom_array.element = elements.astype("U2", copy=False)
    atom_array.set_annotation("atom_id", np.arange(1, len(atom_names) + 1, dtype=int))
    atom_array.set_annotation("occupancy", np.ones(len(atom_names), dtype=float))
    atom_array.set_annotation("b_factor", np.zeros(len(atom_names), dtype=float))
    return atom_array


def load_chemical_component_bond_pairs(
    cif_path: str | Path,
    component_id: str,
) -> list[tuple[str, str]]:
    """Load atom-id bond pairs from an RCSB CCD ligand CIF."""
    component_id = component_id.upper()
    category = pdbx.CIFFile.read(str(cif_path))[component_id]["chem_comp_bond"]
    atom_id_1 = category["atom_id_1"].as_array(str)
    atom_id_2 = category["atom_id_2"].as_array(str)
    return list(zip(atom_id_1.tolist(), atom_id_2.tolist(), strict=True))


def _required_float_column(category: pdbx.CIFCategory, column_name: str) -> np.ndarray:
    values = category[column_name].as_array(str)
    if np.any((values == "?") | (values == ".")):
        raise ValueError(f"Required coordinate column {column_name!r} contains missing values")
    return values.astype(float)


def _download_if_needed(url: str, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        return

    request = Request(url, headers={"User-Agent": "swacanatase/0.1"})
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with urlopen(request, timeout=30) as response:
            temp_path.write_bytes(response.read())
        if temp_path.stat().st_size == 0:
            raise ValueError(f"Downloaded empty file from {url}")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download BP5 from the RCSB CCD.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LIGAND_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = download_bp5(data_dir=args.data_dir, overwrite=args.overwrite)
    atom_array = load_bp5_atom_array(paths.definition_cif)
    print(f"Downloaded {paths.component_id} CCD CIF: {paths.definition_cif}")
    print(f"Downloaded {paths.component_id} ideal SDF: {paths.ideal_sdf}")
    print(f"Loaded {atom_array.array_length()} atoms from ideal CCD coordinates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
