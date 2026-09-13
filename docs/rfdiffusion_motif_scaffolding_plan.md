# RFdiffusion Motif Scaffolding Implementation Plan

## Goal

Add a pipeline stage that starts from generated BP5/Pd/nanoring motif states and
uses RFdiffusion-family tooling to create a single-chain protein scaffold around
the motif. The scaffold must preserve the fixed motif geometry and account for
the non-canonical BP5 sidechain atoms, Pd atom, BP5 virtual carbon anchors, and
carbon nanoring atoms as occupied volume so generated protein does not occlude
the active site or tube interior.

The near-term target is an export-and-run integration that can generate first
designs from one selected motif state. The longer-term target is a scalable
sweep over accepted rotamer and turn-motif states with deterministic reporting,
filtering, and downstream sequence-design handoff.

## Current Status (updated 2026-09-12)

A hands-on validation spike has been completed against a sibling RFdiffusion2
checkout (`/home/broom/AlphaCarbon/RFdiffusion2`). The plan below is largely
confirmed, with a few refinements folded in. Hard operational findings live in
`RFDIFFUSION_EXPORT_NOTES.md` (project root); the throwaway spike input builder
is `scripts/spike_export_rfd2_input.py`.

**Environment — validated and working:**

- RFdiffusion2 installed as a sibling checkout; `setup.py` downloaded all three
  containers (`rfdiffusion`/`chai`/`mlfold` sifs) and model weights.
- Apptainer 1.5.3 installed; GTX 1080 Ti runs the diffusion stage (~3 min/design
  for a ~100-residue scaffold, ~10 min for the 150-residue stock demo).
- Required invocation (host env leaks otherwise break OpenBabel and Hydra):
  `apptainer exec --cleanenv --env USER=$USER --nv <sif> <script> <args>`.
  Chai-1 refolding GPU compatibility on this card is still UNTESTED (Milestone 6).

**Representation decisions (agreed with project owner):**

- Scaffold around **all 9 active sites via C9 symmetry** (config
  `inference/sym.yaml`), not one giant asymmetric chain. This supersedes the
  "avoid symmetry at this stage" guidance in the Contig section below.
- `BP5` is a **non-canonical residue**, so it is NOT treated as an atomized
  protein motif. Instead: relabel its backbone to **ALA** (protein motif) and
  move the bipyridine sidechain, `PD`, and nanoring into **ligand** fragments.
- Scrub the `VRT` (`CV1`/`CV2`) virtual carbons before export (see notes doc).
- Nanoring (`CNT`) needs explicit `CONECT`; plan for `tuber` to emit these.

**Milestone 3 (first external run) — VALIDATED and HARDENED on M18:**

- A single active site exported as `ALA` motif + `BP5,PD,CNT` ligand fragments +
  an `ORI` centering token ran end-to-end and produced a 101-residue scaffold.
- Motif N/CA/C frame preserved to <0.02 A; ligand perfectly rigid; ligand bond
  graphs chemically correct with zero cross-fragment bonds.
- `Pd` open question RESOLVED: feed the true element; RF2AA's internal `Pt`
  relabel is a documented, training-consistent misnomer corrected on output.
- Motif geometry filter should use the **N/CA/C frame** (the carbonyl O
  legitimately reorients toward the generated neighbor).

**Recommended next steps (in order):**

1. **C9 symmetry spike** — confirm the theozyme point group (C9 vs D9) from
   coordinates, then drive `sym.yaml` with one ASU (motif + local ligand) to
   generate the full 9-chain assembly. This is the last research risk that could
   change the exporter's data model, so de-risk it BEFORE writing production code.
2. **Milestone 1 exporter** (`src/swacanatase/rfdiffusion.py`) — encode the
   validated single-ASU + C9 contract, emitting the `ORI` token, ALA-relabeled
   motif, block-diagonal ligand fragments with intra-fragment CONECT, and the
   `--cleanenv --env USER` run command.
3. Proceed with Milestones 4-6 (import/filtering, batch export, downstream).

## Recommended RFdiffusion Target

Use RFdiffusion2 as the primary implementation target.

RFdiffusion2 is the best match for this project because the intended input is
not only a fixed protein backbone motif. The active-site context includes
non-canonical sidechain atoms, a Pd coordination center, virtual carbon anchor
positions, and an extended carbon nanoring. Those requirements are closer to
atom-level active-site or ligand-context scaffolding than to classic backbone
motif scaffolding.

Keep the original RFdiffusion and RFdiffusionAA repositories as fallback or
comparison targets:

- Original RFdiffusion is useful for baseline backbone motif scaffolding but is
  less suitable for explicit non-protein occupied volume.
- RFdiffusionAA is useful if RFdiffusion2 setup or licensing creates a blocker,
  especially for ligand-aware scaffolding experiments.
- RFdiffusion2 should remain the first implementation target because it is
  designed around atomized active-site context and the public workflow includes
  downstream sequence design and validation stages.

Do not vendor RFdiffusion2 into this repository. Treat it as an external
runtime dependency with paths supplied by environment variables or config.

## External Installation Strategy

Install RFdiffusion2 outside `swacanatase`, preferably as a sibling checkout:

```bash
cd /home/broom/AlphaCarbon
git clone https://github.com/RosettaCommons/RFdiffusion2.git
cd RFdiffusion2
python setup.py
```

The public RFdiffusion2 workflow is container-oriented. The machine used for
production runs should have:

- NVIDIA GPU with a compatible driver.
- Apptainer or Singularity.
- RFdiffusion2 container image and model weights installed according to the
  upstream instructions.
- Enough local scratch space for generated structures and intermediate model
  outputs.

The `swacanatase` side should discover the external installation through config:

```bash
export RFDIFFUSION2_HOME=/home/broom/AlphaCarbon/RFdiffusion2
export RFDIFFUSION2_IMAGE=/home/broom/AlphaCarbon/RFdiffusion2/rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif
export RFDIFFUSION2_OUTPUT_ROOT=/home/broom/AlphaCarbon/swacanatase/data/generated/rfdiffusion
```

Optional variables:

```bash
export RFDIFFUSION2_USE_GPU=1
export RFDIFFUSION2_APPTAINER_BIN=apptainer
export RFDIFFUSION2_EXTRA_ARGS=""
```

Before wiring this project to RFdiffusion2, validate the external installation
with an upstream demo command from the RFdiffusion2 repository. This confirms
that the container, weights, GPU, and Python environment work independently from
our export logic.

## Repository Boundary

This repository should own:

- Selecting motif states from existing generated outputs and reports.
- Exporting RFdiffusion-ready input structures.
- Exporting RFdiffusion config files or command manifests.
- Optionally launching RFdiffusion2 through the container runtime.
- Importing generated scaffold outputs.
- Filtering generated structures for motif preservation, active-site clashes,
  and nanoring/tube occlusion.
- Reporting accepted and rejected designs.

RFdiffusion2 should own:

- Diffusion model code.
- Model weights.
- Container image.
- Internal inference scripts.
- Upstream sequence-design and prediction helpers unless we deliberately wrap
  them later.

Normal `pytest` should not require RFdiffusion2, its model weights, GPU access,
or network access.

## Data Model

Add a small set of local concepts around the RFdiffusion boundary.

### `RFDiffusionInput`

Represents one exported motif-scaffolding job.

Suggested fields:

- `job_id`: deterministic string derived from scan name, motif file stem, state
  index, scaffold length range, and export mode.
- `source_structure_path`: existing motif complex from
  `data/generated/motifs/` or `data/generated/scans/<scan-name>/motifs/`.
- `source_report_path`: optional report CSV row source.
- `input_pdb_path`: RFdiffusion-compatible exported PDB.
- `config_path`: generated RFdiffusion2 config or command file.
- `manifest_path`: local JSON manifest with all project-specific metadata.
- `output_dir`: target output directory for RFdiffusion results.
- `fixed_residue_ids`: motif residue identifiers that must remain fixed.
- `context_atom_ids`: BP5/Pd/nanoring atoms treated as occupied context.
- `length_range`: requested single-chain scaffold length range.
- `export_mode`: RFdiffusion2 encoding strategy.

### `RFDiffusionExportManifest`

Write a JSON manifest beside each exported input. This is important because PDB
format alone will not preserve all project-specific intent.

Suggested manifest content:

- source command/config that produced the motif.
- source motif path and checksum.
- nanoring `M`, units, scaffold length, and ring-axis frame if available.
- BP5 rotamer state identifier.
- turn motif name and BP5 motif offset when applicable.
- fixed protein motif residues.
- atomized context atom list.
- BP5 donor atoms, `PD`, `CV1`, `CV2`, and scaffold anchor carbon atom names.
- cylinder exclusion settings.
- clash cutoff settings.
- RFdiffusion2 command/config emitted by this repo.

## Input Structure Export

Create `src/swacanatase/rfdiffusion.py` for structure export, config generation,
command construction, and result import helpers.

The first implementation should write PDB rather than CIF because RFdiffusion
tooling usually accepts PDB-like inputs most reliably. Continue to use Biotite
for reading, writing, coordinate transforms, and atom selection rather than
ad hoc parsing.

Export rules:

- Keep generated RFdiffusion inputs under
  `data/generated/rfdiffusion/inputs/<job-id>/`.
- Use chain `A` for the fixed motif and generated scaffold target.
- Keep nanoring/context atoms in separate chain IDs if RFdiffusion2 accepts
  them for the chosen encoding. A reasonable first layout is:
  - chain `A`: motif protein residues including BP5 residue backbone.
  - chain `B`: carbon nanoring atoms.
  - chain `L` or equivalent: atomized BP5 sidechain/Pd/virtual context if the
    RFdiffusion2 input mode wants ligand-like atoms separated from protein.
- Preserve original coordinates exactly at export time.
- Preserve BP5 source atom names where the receiving RFdiffusion2 mode supports
  them.
- Preserve `PD`, `CV1`, and `CV2` names in the manifest even if the PDB export
  requires element/name normalization.
- Emit a clear warning or hard error if the selected RFdiffusion2 mode cannot
  represent an atom required for exclusion-volume behavior.

The initial export should include all heavy atoms needed for occlusion:

- BP5 sidechain heavy atoms.
- BP5 donor nitrogens.
- `PD`.
- `CV1` and `CV2`.
- all carbon nanoring atoms.
- fixed motif backbone atoms.

Hydrogen atoms are not required for the first export unless the RFdiffusion2
mode explicitly needs them.

## Encoding Strategies

Prototype two RFdiffusion2 input encodings and keep the export code explicit
about which one is being used.

### Strategy A: Atomized Active-Site Context

This is the preferred path.

Represent BP5 sidechain atoms, `PD`, `CV1`, `CV2`, and nanoring carbons as
atomized context or ligand-like atoms. Represent the fixed motif backbone as
protein residues. Ask RFdiffusion2 to build one protein chain around the fixed
motif and atomized context.

Expected advantages:

- Closest to the physical problem.
- Best chance of preventing direct occlusion during generation.
- Allows future partial-ligand or active-site-conditioning experiments.

Expected risks:

- RFdiffusion2 may require exact atom typing, residue naming, or input schema
  adaptation for Pd and virtual atoms.
- The nanoring may be larger than typical ligand context.
- Virtual atoms may need to be represented as carbon-like dummy atoms for
  occupied-volume behavior while retaining their original meaning in metadata.

### Strategy B: Fixed Protein Motif Plus Exclusion Ligand

Represent the motif as fixed protein context and represent BP5/Pd/nanoring as a
non-protein ligand or exclusion-volume object.

Expected advantages:

- Simpler to debug.
- Useful fallback if atomized context has strict chemistry requirements.

Expected risks:

- The model may not use the non-protein atoms as strongly as needed.
- More burden shifts to post-generation clash and cylinder filters.
- Less direct representation of the BP5/Pd active-site chemistry.

The implementation should make it easy to compare both modes on the same motif
state and length range.

## Contig and Scaffold Specification

> NOTE (2026-09-12): This section predates the decision to scaffold all 9 active
> sites via C9 symmetry (see Current Status). It remains accurate for the
> non-symmetric single-ASU validation spike, which is now complete. The
> production target is C9-symmetric; the single-ASU contract described here is
> the per-ASU building block.

The first usable target is a single-chain scaffold around one fixed motif state.
Avoid multichain or symmetry-aware RFdiffusion at this stage.

Use length ranges such as:

- `80-120`: small first-pass scaffold.
- `120-180`: default exploratory scaffold.
- `180-240`: larger enclosing scaffold.

For the first implementation, use fixed indexed motif placement:

```text
N-terminal generated segment + fixed motif residues + C-terminal generated segment
```

Later extensions can explore unindexed motif placement, multiple motif copies,
or explicit symmetry, but those should wait until the single-motif integration
is producing accepted outputs.

The export manifest should record:

- total length range.
- fixed motif residue numbers.
- generated N-terminal and C-terminal ranges if specified separately.
- whether RFdiffusion is allowed to redesign motif-adjacent backbone.
- whether the motif sequence is fixed.

## Command-Line Interface

Add two project commands.

### `swacanatase-rfdiffusion-export`

Creates one or more RFdiffusion input directories without running RFdiffusion.

Example:

```bash
uv run swacanatase-rfdiffusion-export \
  --motif data/generated/scans/default_turns/motifs/example.cif \
  --length 120-180 \
  --encoding atomized_context \
  --output-dir data/generated/rfdiffusion \
  --overwrite
```

Useful options:

- `--motif PATH`: explicit motif structure path.
- `--motif-report PATH`: report CSV used for selecting accepted states.
- `--scan-name NAME`: select rows from a config-run scan.
- `--state-id ID`: select a specific report row or motif state.
- `--max-jobs N`: cap exported jobs for early tests.
- `--length MIN-MAX`: total generated single-chain scaffold length range.
- `--encoding atomized_context|exclusion_ligand`.
- `--include-cv-atoms / --no-include-cv-atoms`.
- `--include-nanoring / --no-include-nanoring`.
- `--cylinder-radius FLOAT`: override nanoring cylinder exclusion radius.
- `--overwrite`.

### `swacanatase-rfdiffusion-run`

Runs or prints RFdiffusion2 commands for exported jobs.

Example:

```bash
uv run swacanatase-rfdiffusion-run \
  --input-dir data/generated/rfdiffusion/inputs/example-job \
  --num-designs 100
```

Useful options:

- `--input-dir PATH`: one exported job directory.
- `--manifest PATH`: explicit manifest path.
- `--num-designs N`.
- `--dry-run`: print commands only.
- `--apptainer-bin PATH`.
- `--rfdiffusion2-home PATH`.
- `--image PATH`.
- `--gpu / --no-gpu`.
- `--extra-arg TEXT`: pass-through argument for RFdiffusion2.

The run command should always write the exact command it executed to a log file
inside the job output directory.

## Python API

Keep the CLI thin and make reusable functions testable.

Suggested API:

```python
from pathlib import Path

from swacanatase.rfdiffusion import (
    RFDiffusionExportOptions,
    build_rfdiffusion_command,
    export_rfdiffusion_input,
    filter_rfdiffusion_outputs,
)

options = RFDiffusionExportOptions(
    motif_path=Path("data/generated/scans/default_turns/motifs/example.cif"),
    output_dir=Path("data/generated/rfdiffusion"),
    length_range=(120, 180),
    encoding="atomized_context",
)

job = export_rfdiffusion_input(options)
command = build_rfdiffusion_command(job, num_designs=100)
```

Implementation should favor dataclasses and plain dictionaries for manifests.
Avoid introducing a workflow engine until the commands and file contracts are
stable.

## Output Import and Filtering

Add post-RFdiffusion filtering before any design is considered accepted.

The first filters should be deterministic geometry filters:

- motif backbone RMSD to the exported fixed motif.
- BP5/Pd atom RMSD if these atoms are present in output.
- minimum distance from generated protein heavy atoms to BP5 heavy atoms.
- minimum distance from generated protein heavy atoms to `PD`.
- minimum distance from generated protein heavy atoms to nanoring carbons.
- minimum distance from generated protein heavy atoms to `CV1` and `CV2`.
- total heavy-atom clash score using the existing clash model where applicable.
- count and depth of generated atoms entering the nanoring finite cylinder.
- optional count of atoms entering a stricter inner catalytic volume around Pd.

Use existing project conventions where possible:

- organic heavy-atom contact cutoff near `2.2 Angstrom`.
- Pd-involving contact cutoff near `1.8 Angstrom`.
- finite cylinder aligned to the nanoring global `Z` axis unless the exported
  motif has an explicit transformed frame.
- ignore intended Pd contacts to scaffold anchor carbons.
- ignore `CV1`/`CV2` contacts only when measuring internal placement quality;
  include them when using them as occupied-volume sentinels for RFdiffusion
  output filtering.

Write filter results to:

```text
data/generated/rfdiffusion/reports/rfdiffusion_design_scores.csv
```

Suggested columns:

- `job_id`
- `design_id`
- `source_motif`
- `encoding`
- `length_range`
- `motif_backbone_rmsd`
- `bp5_context_rmsd`
- `min_bp5_distance`
- `min_pd_distance`
- `min_nanoring_distance`
- `min_cv_distance`
- `clash_score`
- `cylinder_intrusion_count`
- `max_cylinder_intrusion_depth`
- `accepted`
- `rejection_reason`

## Downstream Sequence Design

RFdiffusion outputs should be treated as scaffold backbones or partially
specified structures, not final designs.

Initial downstream path:

1. Generate RFdiffusion2 scaffold backbones.
2. Design sequences with the RFdiffusion2-recommended LigandMPNN path or a
   local LigandMPNN installation.
3. Predict/refold selected sequences with Chai-1 or the preferred local
   structure prediction stack.
4. Re-run the same BP5/Pd/nanoring geometry filters on predicted structures.
5. Compare predicted motif RMSD and nanoring clearance to the original
   RFdiffusion output.

Do not build this full downstream pipeline in the first implementation. The
first implementation should preserve enough metadata and paths that these steps
can be added cleanly.

## Testing Plan

Normal tests should cover only the `swacanatase` boundary and use small local
fixtures.

Unit tests:

- export preserves input coordinates for fixed motif atoms.
- export includes BP5 sidechain atoms, `PD`, `CV1`, `CV2`, and nanoring atoms.
- export manifest records source paths, fixed residues, context atoms, encoding,
  and length range.
- generated job IDs are deterministic.
- command builder uses configured RFdiffusion2 paths and does not require the
  external repository to exist in dry-run mode.
- clash/cylinder filters reject a synthetic design that deliberately overlaps
  BP5, Pd, or nanoring atoms.
- filters accept a synthetic design that is translated away from the occupied
  context.

Integration tests:

- gated behind an environment variable such as
  `SWACANATASE_RUN_RFDIFFUSION_TESTS=1`.
- skipped unless RFdiffusion2 home, container image, and GPU are available.
- run a tiny upstream-compatible example with `--num-designs 1`.
- validate that output files can be imported and scored.

Do not make live network access part of tests.

## Implementation Milestones

### Milestone 1: Export Contract

- Add `src/swacanatase/rfdiffusion.py`.
- Add `swacanatase-rfdiffusion-export`.
- Export one selected motif CIF to a PDB input plus JSON manifest.
- Add tests for deterministic export and manifest contents.

Success criterion: a selected existing motif state can be exported reproducibly
without RFdiffusion2 installed.

### Milestone 2: Command Builder

- Add `swacanatase-rfdiffusion-run --dry-run`.
- Build Apptainer command lines from environment variables and manifest data.
- Log commands into the job directory.
- Add tests for command construction.

Success criterion: the exact RFdiffusion2 command can be inspected and copied
to a workstation or cluster node.

### Milestone 3: First External RFdiffusion2 Run

- Validate RFdiffusion2 independently with an upstream demo.
- Run one exported `swacanatase` motif job with `--num-designs 1`.
- Record any required schema or atom-name adaptations.
- Update the exporter to make the accepted RFdiffusion2 input mode explicit.

Success criterion: RFdiffusion2 accepts a `swacanatase` motif/context input and
writes at least one single-chain scaffold output.

### Milestone 4: Output Import and Geometry Filters

- Import RFdiffusion2 outputs.
- Add deterministic clash, RMSD, and cylinder-occlusion scoring.
- Write `rfdiffusion_design_scores.csv`.
- Add synthetic tests for reject/accept behavior.

Success criterion: generated scaffolds are automatically accepted or rejected
based on active-site and nanoring clearance metrics.

### Milestone 5: Batch Export from Reports

- Select motif states from `turn_motif_scores.csv` and config-run reports.
- Add `--max-jobs`, `--scan-name`, and score-threshold selectors.
- Support length sweeps over the same motif state.

Success criterion: accepted motif states can be turned into a bounded batch of
RFdiffusion jobs without manual file selection.

### Milestone 6: Downstream Handoff

- Add optional helpers for LigandMPNN input preparation.
- Preserve RFdiffusion job metadata through sequence design.
- Reuse the same geometry filters on predicted/refolded structures.

Success criterion: a scaffold can move from motif state to RFdiffusion backbone
to sequence-designed candidate while retaining traceable active-site metrics.

## Initial Open Questions

- ~~Which RFdiffusion2 input mode most faithfully treats Pd and virtual carbon
  atoms as occupied volume?~~ RESOLVED (2026-09-12): atomized/ligand context with
  the true `Pd` element; RF2AA round-trips it correctly. Virtual carbons are
  scrubbed, not represented (see below).
- ~~Should `CV1` and `CV2` be exported as carbon-like dummy atoms...~~ RESOLVED:
  scrub them before export; retain in the manifest only for optional post-filter
  sentinels. See `RFDIFFUSION_EXPORT_NOTES.md`.
- Is the full nanoring too large for the atomized context mode, and if so,
  should the exporter use the full ring for post-filtering but only a local
  nanoring window for model conditioning?
- Should generated scaffolds be explicitly prevented from entering the entire
  tube interior, or only from clashing with the nanoring and BP5/Pd context?
- What length ranges are chemically plausible for the first single-chain
  enclosing scaffold around `M = 18, 24, 30, 36` rings?

## First Practical Experiment

Use the smallest controlled input that still exercises the important constraints:

1. Generate or select one accepted default turn motif around one `M = 18`
   nanoring state.
2. Export it with `--encoding atomized_context --length 120-180`.
3. Run RFdiffusion2 for one design.
4. Import the generated scaffold.
5. Score motif RMSD, BP5/Pd/nanoring clashes, and cylinder intrusion.
6. Inspect the structure manually before expanding to batches.

Only after this works should the pipeline scale across `M`, rotamer states, turn
classes, BP5 motif offsets, and scaffold length ranges.
