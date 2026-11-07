# Symmetric Backbone Generator Spike

## Decision

RFdiffusion3 is the production route for the rigid C9 backbone scaffold around
the M18 BP5/Pd/nanoring theozyme. It is the first evaluated generator that both
enforces coordinate symmetry and keeps all-atom ligand/metal conditioning, and
its input contract has a direct expression for a single shared, non-replicated
ligand on the symmetry axis (the carbon nanoring).

The blocker is no longer the method. A C9 assembly of nine 90-residue subunits
was generated and verified symmetric on the workstation GTX 1080 Ti (Pascal,
11 GB) in 388 seconds, but nine 180-residue subunits exhaust the card. The
production M18 target sits just above the validated size, so production sweeps
need a current data-center or workstation card even though local single designs
may be possible.

Earlier evaluations stand as negative results. RFdiffusion2 does not support
symmetry in its public inference interface. ProteinGenerator runs successfully
with a C9 sequence repeat, but its current symmetry code does not constrain the
generated coordinates to C9.

## RFdiffusion2 result

The sibling checkout was tested at upstream `origin/main` commit `d365cbf`
(2026-04-03). Its `sym.yaml` configuration is parsed but produces only one ASU;
the stock sampler never creates a symmetry manager. Initializing its dormant
symmetry path fails before denoising because `sym_indep.py` calls a missing
`SymIndex.set_kind()` method. An upstream maintainer confirmed in issue #34 that
RFdiffusion2 does not support symmetry.

The non-symmetric single-active-site M18 spike remains a valid asymmetric
control. It is not a production route for the C9 scaffold.

## ProteinGenerator result

ProteinGenerator was built and run from
`/home/broom/Software/protein_generator` using the project-pinned Python 3.8,
PyTorch 2.0, CUDA 11.7, and DGL 1.0.2 environment in an Apptainer image. The
base checkpoint is stored locally outside version control.

`swacanatase-protein-generator-export` writes the nine BP5 `N/CA/C/O` frames as
standard `ALA` `ATOM` records, spaced residue IDs, with one 12-residue gap after
each fixed motif. The valid C9 contig has nine motif-plus-gap units, hence 117
residues; duplicating the first motif to express closure produces 118 residues
and ProteinGenerator rejects it because the length is not divisible by nine.

The following GPU smoke run completed and wrote
`data/generated/protein_generator/c9_m18_spike/smoke_000000.pdb`:

```text
--symmetry 9 --predict_symmetric --num_designs 1 --T 2
```

The implementation of `--symmetry` repeats the sequence tensor. The
`--predict_symmetric` option repeats that sequence once more and makes one
final structure prediction; it does not generate one asymmetric unit and apply
rigid C9 rotations to its coordinates. In the smoke output, the nine
13-residue CA blocks have 3.21--3.48 Angstrom RMSD to the first block after
independent optimal rigid superposition. This is approximate structural repeat
similarity, not coordinate symmetry.

## RFdiffusion3 result (2026-11-06)

RFdiffusion3 ships inside the RosettaCommons `foundry` monorepo, cloned as a
sibling checkout at `/home/broom/AlphaCarbon/foundry` (upstream `origin/main`
commit `0d0b0ba`, 2026-11-04). RFD3 is at `foundry/models/rfd3`; the same repo
carries RF3 and ProteinMPNN/LigandMPNN for the downstream sequence-design and
refolding stages.

Symmetry is a supported inference mode, not a dormant configuration path:

- `inference_sampler.kind=symmetry` selects `SampleDiffusionWithSymmetry`
  (`models/rfd3/src/rfd3/model/inference_sampler.py`), which calls
  `apply_symmetry_to_X_L` on the denoised coordinates at every sampling step.
  This is coordinate-level symmetrization in the RFdiffusion1 sense, not the
  sequence repeat that ProteinGenerator provides.
- `symmetry.id` accepts cyclic and dihedral groups, so `C9` is expressible.
  Frames come from `get_cyclic_frames(order)`; an `input_defined` mode reads
  explicit frames from a `symmetry_file` if the generated axis convention ever
  needs to be overridden.
- Symmetric motif scaffolding requires the input motif to be pre-symmetrized
  and centered on the origin (`is_symmetric_motif: true`, the default, and the
  only supported case). The nine-site M18 theozyme already satisfies this.
- `is_unsym_motif` names motifs or ligands that must **not** be replicated by
  the symmetry operator. This is the direct expression for the carbon nanoring:
  one shared ligand on the symmetry axis plus nine symmetrized BP5/Pd motifs.
  Upstream ships the exact analogs: `indexed_unsym_C2_1bfr` (HEM unsymmetrized)
  and `unsym_C3_6t8h` (C3 protein around a DNA duplex), in
  `models/rfd3/docs/examples/symmetry.json`.
- Ligands and metals are conditioned as in RFdiffusion2 (`ligand: "ORO,ZN"`),
  with `select_fixed_atoms` pinning named atoms per residue, which is how the
  BP5 donor nitrogens and `PD` should be held.
- A non-CCD ligand is supported by giving it a residue name that is invalid for
  the Chemical Component Dictionary, for example `C:T`; AtomWorks then takes the
  geometry from the input file verbatim instead of failing to match a CCD
  conformer (`models/rfd3/docs/common_issues.md`). The carbon nanoring needs
  this. Whether bond perception for a large non-CCD ring is reliable is still an
  open question and must be verified against the loaded bond graph, exactly as
  was done for RFdiffusion2.

Two caveats affect the exactness of the symmetry:

- `SampleDiffusionWithSymmetry.sym_step_frac` defaults to `0.9`, so the final
  ten percent of sampling steps run unsymmetrized. Stock output is therefore
  near-symmetric. Set it to `1.0`, or keep the downstream rigid Cn projection
  plus geometry validation described below.
- Asymmetric motif inputs are rejected outright, and partial diffusion with
  symmetry requires symmetric inputs.

### Local environment, as installed

- Venv at `foundry/.venv` on managed CPython 3.12.13 (`uv python install 3.12`);
  the repo requires `>=3.12` and the host Python 3.14 is newer than anything
  upstream tests. Installed editable: `uv pip install -e ".[rfd3]"`.
- Checkpoint `rfd3_latest.ckpt` (2,690,316,669 bytes) in
  `~/.foundry/checkpoints`, fetched by `foundry install rfd3`.
- **`torch` must be pinned to `2.14.1+cu126`.** The default dependency
  resolution installs `2.15.0+cu132`, whose compiled arch list is
  `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. CUDA 13 dropped Pascal, so on the
  GTX 1080 Ti (`sm_61`) `torch.cuda.is_available()` still returns `True` but the
  first real kernel launch fails with `cudaErrorNoKernelImageForDevice`. The
  `cu126` build carries `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`, and sm_60
  cubins are binary-compatible with sm_61; fp32 and bf16 matmuls both run.
  `2.14.1` is the newest `cu126` build and satisfies foundry's `torch>=2.11`.
- **`inference_mode=legacy` is required on Pascal.** The default `fast` mode
  compiles through Inductor/Triton, which does not target pre-Volta. Legacy
  disables compilation and CUDA graphs and forces `kernels=torch` with vanilla
  dense attention (`models/rfd3/docs/inference_acceleration.md`).
- Local patch branch `local/atomworks-3.0.4-compat`, commit `a6e20a6`. Upstream
  `main` (pyproject `version = "1.0.0"`, unreleased; newest tag is `v0.2.0` and
  newest PyPI `rc-foundry` is `0.2.0`) calls `AnnotateCovalentModifications`
  with `p_reassign_modified_residue` / `atomize_backbone_modifications`. Those
  kwargs exist only on an internal AtomWorks branch (`ci/tagged-releases`); the
  public line, PyPI `atomworks` 3.0.4 and the `production` branch, takes
  `reassign_modified_residue` / `atomize_modified_residue`, whose `True`/`True`
  defaults are equivalent to probability 1.0 plus backbone atomization. Without
  the patch, inference aborts during Hydra instantiation of
  `build_atom14_base_pipeline`. The same call appears in rfd3, rfd3na, rf3, and
  mpnn; all four were patched. Drop the branch once upstream pins a matching
  AtomWorks release. Note that `v0.2.0` has no `checkpoints.py`, so pinning back
  to the last release would also mean finding a different checkpoint.

### Validated C9 run and symmetry measurement (GTX 1080 Ti)

**`length` is per subunit in symmetry mode, not per assembly.** A C9 job with
`length: 90` produces nine 90-residue chains, 810 residues and 4221 atoms total.
Mis-reading this as an assembly total understates every size by a factor of
nine.

C9 unconditional, `length: 90`, `inference_sampler.kind=symmetry`,
`inference_mode=legacy`, `diffusion_batch_size=1`, `low_memory_mode=True`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: **completed in 388 seconds**,
resident at about 7.2 GB of the card's 10.90 GB. Output is preserved at
`data/generated/rfdiffusion3/c9_smoke/`.

The output is coordinate-symmetric, which is the property RFdiffusion2 and
ProteinGenerator could not deliver:

- Nine chains of 90 residues. Per-subunit CA RMSD to chain A after independent
  optimal rigid superposition is 0.0342 to 0.0401 Angstrom. ProteinGenerator's
  equivalent numbers were 3.21 to 3.48 Angstrom.
- A 40 degree rotation about `[0.003, 0.001, -1.000]`, i.e. the Z axis through
  the origin, maps the assembly onto itself with mean closest-atom distance
  0.0564 Angstrom and maximum 0.1419 Angstrom. The symmetry axis convention is
  therefore Z, origin-centered, matching the requirement that motifs be
  pre-symmetrized about the origin.
- The residual 0.04 Angstrom is consistent with `sym_step_frac = 0.9` leaving
  the last ten percent of sampling steps unsymmetrized. Setting it to `1.0`, or
  applying the rigid Cn projection, should drive it to zero.

### Hardware verdict

The card runs real C9 jobs, but it is near its ceiling at the size this project
needs:

- `length: 90` (810 residues): fits, 388 seconds per design.
- `length: 180` (1620 residues): out of memory in a `torch.nn.Linear` forward
  pass, attempting a 2.50 GiB allocation with 7.00 GiB already in use of the
  10.90 GiB capacity. The desktop session holds roughly 1.5 GB of that.

The production M18 target is roughly nine subunits of about 100 residues, plus
the ~108-atom nanoring and nine atomized BP5/Pd sites. That sits just above the
largest job measured here, and atomized ligand atoms add tokens beyond the
residue count. Expect single designs to be marginal-to-feasible locally and
expect any real sweep to need a larger card. Pascal also has no native
bfloat16, and the engine selects bf16 AMP, so the arithmetic is emulated;
forcing fp32 is worth testing if this card is used for more than spot checks.
RFD3 publishes no official supported-hardware table; for comparison,
RFD4-Proteina documents Ampere through Blackwell only.

The production target is roughly nine subunits of about 100 residues plus a
108-atom nanoring, which is around an order of magnitude more tokens than the
180-residue job that already exhausted this card. Plan production C9 runs on a
40-80 GB class accelerator (A100/L40S/H100 or newer). RFD3 publishes no official
supported-hardware table; for comparison, RFD4-Proteina documents Ampere through
Blackwell only.

## Consequence

ProteinGenerator can provide sequence-symmetric candidate backbones for
exploration, subject to motif-frame, clash, pore-intrusion, and symmetry
filters. It cannot by itself meet the required C9 coordinate constraint.

The production route is RFdiffusion3 symmetry mode with the carbon nanoring
passed as an `is_unsym_motif` ligand. Remaining work splits cleanly by hardware:

- Local, on this workstation: build and validate the RFD3 input contract for the
  M18 theozyme (pre-symmetrized motif centered on the origin, non-CCD nanoring
  naming and bond-graph verification, `select_fixed_atoms` for the BP5 donors
  and `PD`), using small symmetry groups and short lengths as sanity checks.
- On a rented or shared current-generation GPU: production C9 M18 sweeps,
  followed by the rigid Cn projection and complete geometry validation that
  remain mandatory while `sym_step_frac` leaves the last sampling steps
  unsymmetrized.
