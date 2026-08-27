# Swacanatase

Exploratory tooling for designing a tube-shaped enzyme that positions catalytic
groups around carbon nanohoops/nanorings.

The first working target is to generate armchair `(n,n)` carbon scaffolds via
the sibling `../tuber` project, load and manipulate structures with Biotite, and
use the RCSB Chemical Component Dictionary record for the non-canonical amino
acid BP5 (`3-(2,2'-BIPYRIDIN-5-YL)-L-ALANINE`) as the ligand/catalytic residue
geometry source.

## Setup

From this directory:

```bash
uv sync --extra dev
```

This project depends on `tuber` as an editable local path dependency:
`../tuber`.

## BP5 Data

Download BP5 Chemical Component Dictionary files from RCSB:

```bash
uv run swacanatase-download-bp5
```

The command writes:

- `data/rcsb/BP5.cif`: CCD definition containing model and ideal coordinates
- `data/rcsb/BP5_ideal.sdf`: ideal-coordinate SDF from RCSB

Load the ideal CCD coordinates as a Biotite `AtomArray`:

```python
from swacanatase import load_bp5_atom_array

bp5 = load_bp5_atom_array()
print(bp5.coord.shape)
```

## Armchair Nanohoop Scaffolds

Generate an armchair `(n,n)` carbon scaffold using `tuber`:

```bash
uv run swacanatase-generate-armchair --n 6 --units 1 --output data/generated/armchair_6_6.cif
```

Or from Python:

```python
from swacanatase import generate_armchair_nanoring

ring = generate_armchair_nanoring(n=6, units=1)
print(ring.array_length())
```

## Data Backup

Local data under `data/` is intentionally not tracked by git. Use S3 sync for
backup/restore once a private bucket exists:

```bash
export SWACANATASE_DATA_S3_URI=s3://your-bucket/swacanatase/data
scripts/sync-data-s3 upload --dry-run
scripts/sync-data-s3 upload
scripts/sync-data-s3 download
```

The sync script accepts an explicit URI instead of the environment variable:

```bash
scripts/sync-data-s3 upload s3://your-bucket/swacanatase/data
```

`--delete` is supported for pruning the destination, but run it with
`--dry-run` first.

## Tests

```bash
uv run --extra dev pytest
```
