"""Export Cn BP5-backbone motif inputs for ProteinGenerator experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
from biotite.structure.io import save_structure
from biotite.structure.io.pdbx import CIFFile, get_structure
import numpy as np

_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_RESIDUE_SPACING = 200


@dataclass(frozen=True)
class ProteinGeneratorSpike:
    """Files and command for one Cn ProteinGenerator backbone experiment."""

    input_pdb_path: Path
    manifest_path: Path
    contigs: tuple[str, ...]
    symmetry_order: int
    command: tuple[str, ...]


def export_cyclic_bp5_backbones(
    theozyme_path: Path,
    output_dir: Path,
    *,
    linker_length: int = 12,
    num_designs: int = 1,
    diffusion_steps: int = 25,
) -> ProteinGeneratorSpike:
    """Export all BP5 backbone motifs as one Cn ProteinGenerator input.

    ProteinGenerator accepts standard protein atoms only.  Each BP5 backbone is
    therefore represented as a single ALA residue with its source N/CA/C/O
    coordinates unchanged.  The Cn symmetry constraint is applied over the N
    equal linker gaps joining those fixed residues in a cyclic order.

    Nanoring, BP5 sidechain, and Pd coordinates are retained in the source CIF
    and manifest for post-generation geometry filtering; they are intentionally
    absent from the protein-only input PDB.
    """
    if linker_length < 1:
        raise ValueError("linker_length must be positive")
    if num_designs < 1:
        raise ValueError("num_designs must be positive")
    if diffusion_steps < 1:
        raise ValueError("diffusion_steps must be positive")

    structure = get_structure(CIFFile.read(str(theozyme_path)), model=1)
    input_structure, source_residue_ids = _protein_generator_input_structure(
        structure
    )
    symmetry_order = len(source_residue_ids)
    if symmetry_order < 2:
        raise ValueError("theozyme must contain at least two BP5 residues")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_pdb_path = output_dir / "bp5_backbone_cyclic_input.pdb"
    save_structure(str(input_pdb_path), input_structure)

    residues = tuple(1 + _RESIDUE_SPACING * index for index in range(symmetry_order))
    cyclic_contigs = tuple(
        item
        for residue in residues
        for item in (f"A{residue}-{residue}", str(linker_length))
    )
    command = (
        "python",
        "inference.py",
        "--pdb",
        str(input_pdb_path.resolve()),
        "--contigs",
        ",".join(cyclic_contigs),
        "--symmetry",
        str(symmetry_order),
        "--predict_symmetric",
        "--num_designs",
        str(num_designs),
        "--T",
        str(diffusion_steps),
        "--save_best_plddt",
        "--out",
        str((output_dir / "design").resolve()),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_theozyme": str(theozyme_path.resolve()),
                "input_pdb": str(input_pdb_path.resolve()),
                "source_bp5_residue_ids": list(source_residue_ids),
                "protein_generator_residue_ids": list(residues),
                "symmetry_order": symmetry_order,
                "linker_length": linker_length,
                "diffusion_steps": diffusion_steps,
                "num_designs": num_designs,
                "contigs": list(cyclic_contigs),
                "protein_generator_command": list(command),
                "excluded_from_generator": ["BP5 sidechain", "PD", "CNT"],
                "required_post_generation_filters": [
                    "BP5 N/CA/C motif RMSD",
                    "BP5/Pd/CNT heavy-atom clashes",
                    "nanoring cylinder intrusion",
                    "Cyclic symmetry RMSD",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return ProteinGeneratorSpike(
        input_pdb_path=input_pdb_path,
        manifest_path=manifest_path,
        contigs=cyclic_contigs,
        symmetry_order=symmetry_order,
        command=command,
    )


def _protein_generator_input_structure(
    structure: struc.AtomArray,
) -> tuple[struc.AtomArray, tuple[int, ...]]:
    bp5 = structure[structure.res_name == "BP5"]
    residue_ids = tuple(sorted(int(residue_id) for residue_id in set(bp5.res_id)))
    residues: list[struc.AtomArray] = []
    for index, residue_id in enumerate(residue_ids):
        residue = bp5[bp5.res_id == residue_id]
        atom_indices = {
            atom_name: np.where(residue.atom_name == atom_name)[0]
            for atom_name in _BACKBONE_ATOMS
        }
        missing = [
            atom_name for atom_name, indices in atom_indices.items() if len(indices) != 1
        ]
        if missing:
            raise ValueError(
                f"BP5 residue {residue_id} must contain exactly one of each backbone atom; "
                f"problem atoms: {', '.join(missing)}"
            )
        backbone = residue[
            np.array([atom_indices[atom_name][0] for atom_name in _BACKBONE_ATOMS])
        ].copy()
        backbone.res_name[:] = "ALA"
        backbone.chain_id[:] = "A"
        backbone.res_id[:] = 1 + _RESIDUE_SPACING * index
        backbone.hetero[:] = False
        residues.append(backbone)
    return sum(residues[1:], residues[0]), residue_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Cn BP5-backbone ProteinGenerator spike input."
    )
    parser.add_argument("--theozyme", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--linker-length", type=int, default=12)
    parser.add_argument("--num-designs", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=25)
    args = parser.parse_args(argv)
    spike = export_cyclic_bp5_backbones(
        args.theozyme,
        args.output_dir,
        linker_length=args.linker_length,
        num_designs=args.num_designs,
        diffusion_steps=args.diffusion_steps,
    )
    print(f"wrote {spike.input_pdb_path}")
    print(f"wrote {spike.manifest_path}")
    print(" ".join(spike.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
