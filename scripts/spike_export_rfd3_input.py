#!/usr/bin/env python3
"""SPIKE (throwaway): build a C9-symmetric RFdiffusion3 input from an accepted
turn-motif theozyme state, to validate the RFD3 symmetric motif-scaffolding
contract before writing the real exporter in src/swacanatase/rfdiffusion.py.

Validated end-to-end on 2026-11-07 (see RFDIFFUSION_EXPORT_NOTES.md). The
representation this script emits, and why each part is required:

  - Each of the nine turn motifs becomes its own subunit chain (A-I), renumbered
    1..motif_length. RFD3 derives its symmetry frames in
    rfd3/inference/symmetry/frames.py from atoms with chain_type == 6, so the
    input must contain nine *polypeptide* subunits. A lone BP5 residue per chain
    is not a polymer and the frame search fails with an empty min().
  - BP5 is written as ATOM, not HETATM. Its CCD type is L-PEPTIDE LINKING, so
    AtomWorks keeps it in the polymer; as a heteroatom it is split out and the
    remaining GLY residues become non-contiguous, failing AtomWorks' own
    chain-info consistency check.
  - Pd is a separate residue (motif_length + 1) on the same chain. AtomWorks
    moves it to its own chain during parsing, which is expected and harmless.
  - The carbon nanoring is a single ligand on chain L named 'C:T', deliberately
    invalid as a CCD code so AtomWorks takes our geometry verbatim rather than
    matching a CCD conformer. It is named in symmetry.is_unsym_motif so the C9
    operator does not replicate it: one shared molecule on the axis.
  - VRT (CV1/CV2) and hydrogens are dropped; OXT/HXT are dropped because the
    motif becomes internal to a designed chain.
  - Explicit CONECT for BP5 (from the CCD), the motif's peptide bonds, and the
    ring (by distance). Ring carbons are renumbered C1..CN because tuber names
    every nanoring carbon 'C' and duplicate names within a residue break both
    CONECT mapping and AtomWorks.

Usage:
  python scripts/spike_export_rfd3_input.py \
      --motif-cif data/generated/full_clash_0_0_m12_20/scans/turn_motifs/motifs/nanoring_M18_gplus_m90_beta_turn_ad_bp5pos2.cif \
      --out-dir data/generated/rfdiffusion3/c9_turn_motif_spike \
      --n-term 30 --c-term 30

Then, from the foundry checkout:
  rfd3 design inputs=<out-dir>/rfd3_<state>.json out_dir=<out-dir>/out \
      inference_sampler.kind=symmetry inference_mode=legacy \
      diffusion_batch_size=1 low_memory_mode=True prevalidate_inputs=True
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

from swacanatase.ligands import load_bp5_bond_pairs
from swacanatase.placement import BP5_VIRTUAL_CARBON_ATOMS

SUBUNIT_CHAINS = "ABCDEFGHI"
RING_CHAIN = "L"
RING_RESNAME = "C:T"
VIRTUAL_RESNAME = "VRT"
TERMINAL_ATOMS = frozenset({"OXT", "HXT"})
RING_BOND_CUTOFF = 1.8
SYMMETRY_ORDER = 9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motif-cif", required=True, type=Path,
                    help="an accepted turn-motif state from a turn_motif scan")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--bp5-cif", type=Path, default=Path("data/rcsb/BP5.cif"))
    ap.add_argument("--n-term", type=int, default=30,
                    help="designed residues before the motif, per subunit")
    ap.add_argument("--c-term", type=int, default=30,
                    help="designed residues after the motif, per subunit")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    structure = pdbx.get_structure(pdbx.CIFFile.read(str(args.motif_cif)), model=1)
    structure = _scrub(structure)

    ring_source = structure[structure.res_name == "CNT"]
    palladium = structure[structure.res_name == "PD"]
    peptide = structure[~np.isin(structure.res_name, ["CNT", "PD"])]
    if not ring_source.array_length():
        raise ValueError("no CNT nanoring in the motif state")

    residue_ids = sorted(set(int(r) for r in peptide.res_id))
    motif_length, remainder = divmod(len(residue_ids), SYMMETRY_ORDER)
    if remainder or motif_length < 1:
        raise ValueError(
            f"{len(residue_ids)} peptide residues is not {SYMMETRY_ORDER} equal motifs"
        )

    pieces, bp5_position = _subunit_pieces(
        peptide, palladium, residue_ids, motif_length
    )
    pieces.append(_ring_piece(ring_source))

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined + piece

    bonds = _bond_pairs(combined, motif_length, args.bp5_cif)
    state = args.motif_cif.stem
    pdb_path = args.out_dir / f"rfd3_{state}.pdb"
    _write_pdb(combined, bonds, pdb_path)

    spec_path = args.out_dir / f"rfd3_{state}.json"
    spec_path.write_text(
        json.dumps(
            _spec(state, pdb_path, motif_length, args.n_term, args.c_term), indent=4
        )
        + "\n"
    )

    asu = args.n_term + motif_length + args.c_term
    print(f"wrote {pdb_path}")
    print(f"wrote {spec_path}")
    print(f"  motif length: {motif_length} residues, BP5 at position {bp5_position}")
    print(f"  subunit chains: {SYMMETRY_ORDER} x {motif_length} residues + 1 Pd")
    print(f"  ring: {ring_source.array_length()} atoms as ligand {RING_RESNAME!r}"
          f" (chain {RING_CHAIN}, not symmetrized)")
    print(f"  CONECT records: {len(bonds)} bonds")
    print(f"  ASU contig: {args.n_term} + {motif_length} + {args.c_term} = {asu} residues")
    print(f"  assembly: {SYMMETRY_ORDER * asu} residues")


def _scrub(structure: struc.AtomArray) -> struc.AtomArray:
    """Drop hydrogens, the CV1/CV2 virtual carbons, and terminal OXT/HXT."""
    keep = structure.element != "H"
    keep &= structure.res_name != VIRTUAL_RESNAME
    keep &= ~np.isin(structure.atom_name, list(BP5_VIRTUAL_CARBON_ATOMS))
    keep &= ~np.isin(structure.atom_name, list(TERMINAL_ATOMS))
    return structure[keep]


def _subunit_pieces(
    peptide: struc.AtomArray,
    palladium: struc.AtomArray,
    residue_ids: list[int],
    motif_length: int,
) -> tuple[list[struc.AtomArray], int]:
    """Split one chain of nine consecutive motifs into nine subunit chains."""
    pieces: list[struc.AtomArray] = []
    bp5_position = 0
    for site in range(SYMMETRY_ORDER):
        block = residue_ids[site * motif_length:(site + 1) * motif_length]
        segment = peptide[np.isin(peptide.res_id, block)].copy()
        remap = {old: new for new, old in enumerate(block, start=1)}
        segment.res_id[:] = [remap[int(r)] for r in segment.res_id]
        segment.chain_id[:] = SUBUNIT_CHAINS[site]
        # BP5 is L-PEPTIDE LINKING and must stay inside the polymer.
        segment.hetero[:] = False

        positions = sorted(
            {int(r) for r, n in zip(segment.res_id, segment.res_name) if n == "BP5"}
        )
        if len(positions) != 1:
            raise ValueError(f"expected one BP5 per motif, found {positions}")
        if site == 0:
            bp5_position = positions[0]
        elif positions[0] != bp5_position:
            raise ValueError("BP5 sits at different positions across motifs")

        metal = palladium[palladium.res_id == block[positions[0] - 1]].copy()
        if metal.array_length() != 1:
            raise ValueError(f"expected one Pd for motif {site + 1}")
        metal.chain_id[:] = SUBUNIT_CHAINS[site]
        metal.res_id[:] = motif_length + 1
        metal.hetero[:] = True
        pieces += [segment, metal]
    return pieces, bp5_position


def _ring_piece(ring_source: struc.AtomArray) -> struc.AtomArray:
    ring = ring_source.copy()
    ring.chain_id[:] = RING_CHAIN
    ring.res_id[:] = 1
    ring.res_name[:] = RING_RESNAME
    ring.hetero[:] = True
    ring.atom_name[:] = [f"C{i + 1}" for i in range(ring.array_length())]
    return ring


def _bond_pairs(
    combined: struc.AtomArray, motif_length: int, bp5_cif: Path
) -> list[tuple[int, int]]:
    ccd_bonds = {frozenset(pair) for pair in load_bp5_bond_pairs(bp5_cif)}
    bonds: list[tuple[int, int]] = []
    for chain in SUBUNIT_CHAINS:
        indices = np.where(combined.chain_id == chain)[0]
        chain_atoms = combined[indices]
        for residue_id in sorted(
            set(int(r) for r in chain_atoms.res_id[chain_atoms.res_name == "BP5"])
        ):
            selected = indices[
                (chain_atoms.res_id == residue_id) & (chain_atoms.res_name == "BP5")
            ]
            by_name = {combined.atom_name[i]: int(i) for i in selected}
            for pair in ccd_bonds:
                first, second = tuple(pair)
                if first in by_name and second in by_name:
                    bonds.append((by_name[first], by_name[second]))
        for residue_id in range(1, motif_length):
            carbon = indices[
                (chain_atoms.res_id == residue_id) & (chain_atoms.atom_name == "C")
            ]
            nitrogen = indices[
                (chain_atoms.res_id == residue_id + 1) & (chain_atoms.atom_name == "N")
            ]
            if len(carbon) == 1 and len(nitrogen) == 1:
                bonds.append((int(carbon[0]), int(nitrogen[0])))
    bonds += _ring_bonds(combined)
    return bonds


def _ring_bonds(combined: struc.AtomArray) -> list[tuple[int, int]]:
    indices = np.where(combined.res_name == RING_RESNAME)[0]
    coords = combined.coord[indices]
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    rows, columns = np.triu_indices(len(indices), k=1)
    close = distances[rows, columns] <= RING_BOND_CUTOFF
    return [
        (int(indices[i]), int(indices[j]))
        for i, j in zip(rows[close], columns[close])
    ]


def _spec(
    state: str, pdb_path: Path, motif_length: int, n_term: int, c_term: int
) -> dict:
    motif_selection = f"A1-{motif_length}"
    return {
        f"{state}_{n_term}_{c_term}": {
            "input": str(pdb_path),
            "contig": f"{n_term},{motif_selection},{c_term}",
            "length": None,
            "ligand": f"PD,{RING_RESNAME}",
            "symmetry": {
                "id": f"C{SYMMETRY_ORDER}",
                "is_symmetric_motif": True,
                "is_unsym_motif": RING_RESNAME,
            },
            "select_fixed_atoms": {
                motif_selection: "ALL",
                "PD": "ALL",
                RING_RESNAME: "ALL",
            },
            "allow_ligand_on_existing_chain": True,
        }
    }


def _write_pdb(
    combined: struc.AtomArray, bonds: list[tuple[int, int]], path: Path
) -> None:
    """Write ATOM/HETATM plus explicit CONECT records.

    Written by hand because biotite drops the heteroatom bond records RFD3 needs
    for the non-CCD nanoring and the BP5 sidechain.
    """
    lines: list[str] = []
    for index in range(combined.array_length()):
        name = combined.atom_name[index]
        name_column = name if len(name) >= 4 else f" {name:<3}"
        record = "HETATM" if combined.hetero[index] else "ATOM  "
        x, y, z = combined.coord[index]
        lines.append(
            f"{record}{index + 1:>5} {name_column} {combined.res_name[index]:>3} "
            f"{combined.chain_id[index]:>1}{combined.res_id[index]:>4}{'':4}"
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{1.00:>6.2f}{0.00:>6.2f}{'':10}{combined.element[index]:>2}"
        )

    partners: dict[int, list[int]] = {}
    for first, second in bonds:
        partners.setdefault(first, []).append(second)
        partners.setdefault(second, []).append(first)
    for atom in sorted(partners):
        listed = sorted(partners[atom])
        for start in range(0, len(listed), 4):
            lines.append(
                f"CONECT{atom + 1:>5}"
                + "".join(f"{other + 1:>5}" for other in listed[start:start + 4])
            )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
