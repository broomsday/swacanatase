# AGENTS.md

## Project Intent

This repository explores structural requirements for a tube-shaped enzyme that
could synthesize a carbon nanoring within its inner tube volume. The immediate
modeling focus is positioning the RCSB CCD non-canonical amino acid `BP5`
around armchair carbon nanohoop/nanotube scaffolds.

## Local Context

- The sibling project `../tuber` generates finite carbon nanotube structures
  from chiral indices `(n,m)` and writes PDB/CIF through Biotite.
- This project depends on `tuber` through `uv` as an editable local path source.
- RCSB BP5 ligand data is stored under `data/rcsb/`.
- Prefer Biotite `AtomArray`/structure APIs for coordinate manipulation and
  interrogation.

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
  nanorings/nanohoops.
- Keep generated carbon structures aligned to the global `Z` axis unless a
  placement experiment explicitly rotates/translates them.
- Keep BP5 source coordinates traceable to the RCSB Chemical Component
  Dictionary; do not hand-edit downloaded ligand files.
- For future BP5 placement code, make coordinate frames explicit: ring axis,
  radial direction, tangential direction, and BP5 anchor atoms should be named
  in code and tests.
- Prefer small, testable geometry functions over notebooks as the canonical
  implementation. Notebooks are fine for inspection, but reusable logic belongs
  in `src/swacanatase/`.

## Coding Conventions

- Use Python 3.11+.
- Use NumPy arrays for numerical geometry and Biotite for structure objects.
- Avoid ad hoc parsing of PDB/mmCIF where Biotite or structured CCD fields are
  available.
- Keep tests deterministic and avoid live network access in tests. If a test
  needs BP5, use the checked-in files in `data/rcsb/`.
- Run `uv run --extra dev pytest` before handing off code changes.
