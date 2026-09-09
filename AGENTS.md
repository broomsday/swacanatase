# AGENTS.md

## Project Intent

This repository explores structural requirements for a tube-shaped enzyme that
could synthesize a carbon nanoring within its inner tube volume. The current
modeling focus is BP5/Pd active-site geometry and deterministic placement of
the RCSB CCD non-canonical amino acid `BP5` around armchair M=N
carbon nanohoop/nanotube scaffolds.

## Local Context

- The sibling project `../tuber` generates finite carbon nanotube structures
  from chiral indices `(n,m)` and writes PDB/CIF through Biotite.
- This project depends on `tuber` through `uv` as an editable local path source.
- RCSB BP5 ligand data is stored locally under `data/rcsb/`, but `data/` is
  gitignored. Hydrate it with `uv run swacanatase-download-bp5 --overwrite` or
  restore it from private backup before running tests.
- Generated nanoring and theozyme outputs are written under
  `data/generated/nanoring/` and `data/generated/theozyme/`.
- Reference notes and papers may be stored locally under `data/references/`;
  these are also ignored by git.
- Prefer Biotite `AtomArray`/structure APIs for coordinate manipulation and
  interrogation.

## Code Map

- `src/swacanatase/ligands.py`: downloads BP5 CCD/SDF data and loads CCD atom
  and bond records into structured Python/Biotite objects.
- `src/swacanatase/active_site.py`: derives the cis BP5 donor conformer, appends
  square-planar `PD`, `CV1`, and `CV2` atoms, and measures the resulting
  geometry.
- `src/swacanatase/nanoring.py`: wraps `tuber` armchair `(n,n)` generation and
  writing.
- `src/swacanatase/bp5_rotamers.py`: enumerates deterministic BP5 chi1/chi2
  rotamers while preserving the placed BP5/Pd anchor frame.
- `src/swacanatase/secondary_structure.py`: builds regular alpha/beta backbone
  segments and finite turn motifs around a fixed placed BP5 residue; owns
  Ramachandran scan targets, BetaTurnLib18 turn definitions, turn perturbation
  grids, cis-turn metadata, clash helpers, and orientation/cylinder metrics.
- `src/swacanatase/placement.py`: generates M=N scaffolds, selects central-band
  anchor pairs, places `M/2` BP5/Pd sidechains, expands symmetric rotamer,
  secondary-structure, and turn-motif states, and writes scaffold/theozyme,
  rotamer, secondary-structure, turn-motif, and report series.

## Setup

Use `uv` from the repository root:

```bash
uv sync --extra dev
```

Download or refresh BP5:

```bash
uv run swacanatase-download-bp5 --overwrite
```

Run tests:

```bash
uv run --extra dev pytest
```

## Modeling Guidelines

- Treat `(n,n)` scaffolds as the first-pass representation of M=N
  nanorings/nanohoops. The default M series is `18, 24, 30, 36`, and the
  default scaffold length is `1.5` armchair units, producing `6M` carbon atoms.
- Keep generated carbon structures aligned to the global `Z` axis unless a
  placement experiment explicitly rotates/translates them.
- Keep BP5 source coordinates traceable to the RCSB Chemical Component
  Dictionary; do not hand-edit downloaded ligand files.
- The implemented active BP5 conformer is derived by rotating the connected
  `C6` side of the ligand 180 degrees about the `C3-C6` inter-ring bond so
  `N1` and `N2` are on the same side.
- The implemented Pd model appends `PD`, `CV1`, and `CV2` using default
  distances `PD-N = 2.018 Angstrom` and `PD-CV = 2.020 Angstrom`.
- Placement code should continue to keep coordinate frames explicit: ring axis,
  radial direction, tangential direction, anchor atom indices, and BP5 virtual
  carbon atoms should be named in code and tests.
- Current BP5 placement uses central inter-benzene linker anchor pairs by
  default. Para-like central-band pairs remain available for comparison.
- By default, rigid placement preserves BP5/Pd virtual-carbon geometry and
  minimizes residuals against scaffold anchors. Use `snap_virtual_carbons` or
  `--snap-virtual-carbons` only for experiments that require exact `CV1`/`CV2`
  overlap with scaffold carbons.
- Turn-motif anchoring keeps the placed BP5/Pd residue fixed on its nanoring
  anchor pair and grows the finite motif around that fixed residue. Treat
  `bp5_motif_offset`/`--turn-bp5-position` as the 1-based residue position of
  BP5 within the 3-residue gamma turn or 4-residue beta turn.
- Turn-motif scans expand complete symmetric states as rotamer state x motif x
  BP5 motif offset. Default turn motifs are `beta_turn_ad`, `beta_turn_ab1`,
  `beta_turn_ab2`, and `gamma_turn_inverse`; additional cis-turn motifs require
  `--include-cis-turns`.
- Cis-Pro turn motifs currently enforce fixed residue identity and non-trans
  peptide omega values, but generated Pro residues are backbone-only.
- Generated complexes use chain `A` for BP5/Pd sidechains and chain `B` for the
  carbon scaffold.
- Prefer small, testable geometry functions over notebooks as the canonical
  implementation. Notebooks are fine for inspection, but reusable logic belongs
  in `src/swacanatase/`.

## Coding Conventions

- Use Python 3.11+.
- Use NumPy arrays for numerical geometry and Biotite for structure objects.
- Avoid ad hoc parsing of PDB/mmCIF where Biotite or structured CCD fields are
  available.
- Keep tests deterministic and avoid live network access in tests. If a test
  needs BP5, use local files in `data/rcsb/`; do not make tests download from
  RCSB.
- Run `uv run --extra dev pytest` before handing off code changes.
