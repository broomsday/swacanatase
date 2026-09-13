#!/usr/bin/env python3
"""SPIKE (throwaway): build a single-active-site RFdiffusion2 input from an
M18 theozyme, to test the BP5->ALA + ligand representation before writing the
real exporter in src/swacanatase/rfdiffusion.py.

Non-symmetric, single ASU:
  - Active site #1 only (BP5[1] + PD[1]); VRT (CV1/CV2) is dropped.
  - BP5 backbone (N/CA/C/O) is relabeled to ALA and written as protein chain A.
  - BP5 bipyridine sidechain heavy atoms -> ligand fragment 'BP5' (chain X).
  - PD -> ligand fragment 'PD'.
  - A local arc of the nearest CNT ring carbons -> ligand fragment 'CNT'.
  - Explicit CONECT records for BP5 sidechain and the CNT arc.

Usage:
  python scripts/spike_export_rfd2_input.py \
      --theozyme data/generated/theozyme/nanoring_M18_bp5.cif \
      --out data/generated/rfdiffusion/spike/site1_input.pdb \
      --site 1 --arc 14
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import biotite.structure.io.pdbx as pdbx

# BP5 sidechain heavy-atom bonds (from data/rcsb/BP5.cif _chem_comp_bond,
# heavy atoms only; the C12-CA bond crosses into the protein backbone and is
# intentionally omitted here).
BP5_SIDECHAIN_BONDS = [
    ("C9", "C12"), ("C9", "C8"), ("C9", "C11"),
    ("C8", "C7"), ("C7", "C6"),
    ("C6", "N2"), ("C6", "C3"),
    ("C5", "C1"), ("C5", "C4"), ("C4", "N1"),
    ("C3", "C2"), ("C3", "N1"), ("C2", "C1"),
    ("C11", "N2"),
]
BP5_BACKBONE = ("N", "CA", "C", "O")  # kept as ALA; OXT/HXT dropped
BP5_SIDECHAIN = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
                 "C11", "C12", "N1", "N2")


def _pdb_atom(record, serial, name, resname, chain, resseq, xyz, element):
    # PDB atom name column: right-justify 1-2 char names into cols 14-15
    nm = name if len(name) >= 4 else f" {name:<3}"
    return (f"{record:<6}{serial:>5} {nm}{'':1}{resname:>3} {chain}"
            f"{resseq:>4}{'':1}   {xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
            f"{1.0:>6.2f}{0.0:>6.2f}{'':>10}{element:>2}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theozyme", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--site", type=int, default=1)
    ap.add_argument("--arc", type=int, default=14,
                    help="number of nearest CNT carbons to include")
    args = ap.parse_args()

    cif = pdbx.CIFFile.read(str(args.theozyme))
    arr = pdbx.get_structure(cif, model=1)
    heavy = arr[arr.element != "H"]

    def sel(resname, resid=None):
        m = heavy.res_name == resname
        if resid is not None:
            m &= heavy.res_id == resid
        return heavy[m]

    bp5 = sel("BP5", args.site)
    pd = sel("PD", args.site)
    cnt = sel("CNT")
    if bp5.array_length() == 0 or pd.array_length() == 0:
        raise SystemExit(f"site {args.site}: BP5/PD not found")

    # local nanoring arc: nearest CNT carbons to the BP5 sidechain centroid
    sc_mask = np.isin(bp5.atom_name, BP5_SIDECHAIN)
    centroid = bp5.coord[sc_mask].mean(axis=0)
    d = np.linalg.norm(cnt.coord - centroid, axis=1)
    arc_idx = np.argsort(d)[: args.arc]
    arc = cnt[arc_idx]

    lines: list[str] = ["REMARK  spike single-ASU RFdiffusion2 input"]
    serial = 1
    name_to_serial: dict[str, int] = {}

    # --- protein: ALA backbone from BP5 backbone atoms ---
    for name in BP5_BACKBONE:
        m = bp5.atom_name == name
        if not m.any():
            raise SystemExit(f"BP5 missing backbone atom {name}")
        a = bp5[m][0]
        lines.append(_pdb_atom("ATOM", serial, name, "ALA", "A", 1,
                               a.coord, a.element))
        serial += 1
    lines.append(f"TER   {serial:>5}      ALA A   1")
    serial += 1

    # --- ligand fragment: BP5 bipyridine sidechain ---
    for name in BP5_SIDECHAIN:
        m = bp5.atom_name == name
        if not m.any():
            continue
        a = bp5[m][0]
        lines.append(_pdb_atom("HETATM", serial, name, "BP5", "X", 1,
                               a.coord, a.element))
        name_to_serial[name] = serial
        serial += 1

    # --- ligand fragment: PD ---
    pd_a = pd[0]
    # Use the true element "Pd" (Z=46). RF2AA internally relabels it 'Pt' via a
    # deliberately-shifted atomnum2atomtype table (kept for fold&dock3 weight
    # compatibility; see rf2aa/chemical.py) and corrects it back to Pd on output
    # in util.writepdb(remap_atomtype=True). So Pd round-trips correctly.
    lines.append(_pdb_atom("HETATM", serial, "PD", "PD", "X", 2,
                           pd_a.coord, "Pd"))
    serial += 1

    # --- ligand fragment: CNT arc ---
    cnt_serials: list[int] = []
    cnt_coords: list[np.ndarray] = []
    for i in range(arc.array_length()):
        a = arc[i]
        nm = f"C{i+1}"
        lines.append(_pdb_atom("HETATM", serial, nm, "CNT", "X", 3,
                               a.coord, "C"))
        cnt_serials.append(serial)
        cnt_coords.append(a.coord)
        serial += 1

    # --- ORI centering token (dummy atom, element X) at active-site centroid ---
    # Required by the `aa` inference config for centering; not a model ligand
    # (handled separately by rf_diffusion/inference/centering.py).
    site_atoms = np.vstack([bp5.coord[sc_mask], pd_a.coord[None, :],
                            np.array(cnt_coords)])
    ori = site_atoms.mean(axis=0)
    lines.append(_pdb_atom("HETATM", serial, "ORI", "ORI", "B", 332, ori, "X"))
    serial += 1

    # --- CONECT: BP5 sidechain bonds ---
    conect: list[tuple[int, int]] = []
    for x, y in BP5_SIDECHAIN_BONDS:
        if x in name_to_serial and y in name_to_serial:
            conect.append((name_to_serial[x], name_to_serial[y]))

    # --- CONECT: CNT arc bonds by distance (C-C < 1.8 A) ---
    cc = np.array(cnt_coords)
    for i in range(len(cc)):
        for j in range(i + 1, len(cc)):
            if np.linalg.norm(cc[i] - cc[j]) < 1.8:
                conect.append((cnt_serials[i], cnt_serials[j]))

    for a, b in conect:
        lines.append(f"CONECT{a:>5}{b:>5}")
    lines.append("END")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    print(f"  ALA motif: 1 residue (chain A)")
    print(f"  ligand BP5 sidechain atoms: {len(name_to_serial)}")
    print(f"  ligand PD atoms: 1")
    print(f"  ligand CNT arc atoms: {len(cnt_serials)}")
    print(f"  CONECT records: {len(conect)}")


if __name__ == "__main__":
    main()
