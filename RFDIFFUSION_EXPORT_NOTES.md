# RFdiffusion Export Notes

Decisions that affect how motif/active-site structures are prepared before they
are handed to RFdiffusion2. See `docs/rfdiffusion_motif_scaffolding_plan.md` for
the full pipeline plan.

## Validated runtime findings (single-ASU spike, M18)

A non-symmetric single-active-site spike was run end-to-end against RFdiffusion2
(sibling checkout at `../RFdiffusion2`). It succeeded: RFD2 accepted a
swacanatase-derived motif+ligand input and generated a ~100-residue scaffold in
~3 min on a GTX 1080 Ti. Concrete facts the real exporter/runner must honor:

1. **Apptainer invocation** — host env leaks break the container. Always run:
   `apptainer exec --cleanenv --env USER=$USER --nv <sif> <script> <args>`
   (`--cleanenv` stops host `BABEL_LIBDIR`/`MKL_THREADING_LAYER` from breaking
   OpenBabel; re-inject `USER` because a Hydra config interpolation needs it.
   Add `--env PYTHONPATH=<RFdiffusion2 repo>` when calling scripts directly.)

2. **`ORI` centering token is required.** The `aa` inference config asserts on a
   dummy `HETATM` with atom name/resname `ORI`, element `X`, placed at the
   centering origin (we use the active-site centroid). It is NOT passed in
   `inference.ligand`; centering handles it separately.

3. **Pd round-trips correctly.** Feed the true element `Pd` (Z=46). RF2AA
   internally relabels it `Pt` via a deliberately-shifted table (fold&dock3
   weight compatibility) but corrects it back to `Pd` on output. No workaround
   needed. See `../RFdiffusion2/rf2aa/chemical.py` and `util.writepdb`.

4. **Ligand fragments are parsed independently and block-diagonalized** — no
   inter-fragment bonds. Passing `inference.ligand='BP5,PD,CNT'` parses each
   separately. Verified: bond graphs are chemically correct per fragment (BP5
   bipyridine all 14 bonds, Pd isolated, nanoring connected), zero cross-fragment
   bonds. RFD2 feeds ALL `CONECT` lines to each per-resname parse, so OpenBabel
   emits benign "CONECT record ignored" warnings for the other fragments' lines.
   These are unavoidable (RFD2 takes a single `input_pdb`) and harmless — do NOT
   try to "fix" them by splitting PDBs. Just filter them from logs.

5. **Working contig for a fixed backbone motif** (no atomization):
   `contigmap.contigs=['50,A1-1,50']` with `inference.contig_as_guidepost=False`.

6. **Motif preservation (verified via .trb mapping + output coords):** the fixed
   motif's **N/CA/C backbone frame is preserved to <0.02 A** relative to the
   fixed ligand; the ligand is perfectly rigid (moved only by ORI centering). The
   **carbonyl O reorients (~0.85 A)** because its psi depends on residue i+1,
   which is generated. => The geometry filter should compute motif RMSD on the
   **N/CA/C frame**, not naive all-backbone-including-O. Input->output index map
   is in the `.trb` (`con_ref_pdb_idx` -> `con_hal_pdb_idx`), e.g. `A1 -> A51`.

The spike input builder lives at `scripts/spike_export_rfd2_input.py` (throwaway;
informs the real `src/swacanatase/rfdiffusion.py`).

## C9 symmetry spike (2026-10-04): RFdiffusion2 does not support symmetry

The validated M18 single-ASU input was run with `config/inference/sym.yaml`,
`sym.symid=C9`, `sym.max_nsub=9`, and the same `BP5,PD,CNT` ligand contract.
The diffusion job completed, but its PDB contains only the 101-residue ASU and
one copy of each ligand fragment. It is therefore a useful asymmetric control,
not a C9 result. The stock `rf_diffusion/run_inference.py` parses `sym.yaml` but
does not call `ipd.sym.create_sym_manager(conf)`, so no symmetry manager is
installed.

A spike-only launcher was then used to create the RFdiffusion symmetry manager
before calling the stock sampler. That reached the intended symmetry path, but
failed before the first denoising step in
`rf_diffusion/sym/sym_indep.py`: it calls `SymIndex.set_kind()`, which does not
exist on the `SymIndex` implementation bundled in this checkout. This is an
upstream RFdiffusion2/IPD API incompatibility, not an export or ligand-parsing
failure.

This behavior is confirmed upstream, not merely a local checkout problem. The
checkout is at its then-current `origin/main` (`d365cbf`, 2026-04-03), and an
RFdiffusion2 maintainer closed [issue #34](https://github.com/RosettaCommons/RFdiffusion2/issues/34)
with: “RFdiffusion2, unfortunately, does not support symmetry.” The dormant
symmetry configuration and partial implementation are therefore not a supported
inference interface.

Do not treat `data/generated/rfdiffusion/c9_spike/out/` as a symmetric design,
and do not build the production exporter around RFdiffusion2 C9 inference.
ProteinGenerator's protein-only `--symmetry` / `--predict_symmetric` workflow
was subsequently tested and is sequence-symmetric but not rigidly coordinate-
symmetric. See [the symmetric-backbone-generator spike](docs/symmetric_backbone_generator_spike.md)
for the complete cross-generator conclusion.

## Scrub the virtual carbons (`CV1`, `CV2`) after the motif is built

`CV1` and `CV2` are **theozyme construction scaffolding, not real atoms**. They
are the two virtual coupling-carbon positions placed in the BP5/Pd coordination
plane (see `src/swacanatase/active_site.py`) and used to define the catalytic
geometry while the active site is assembled.

Once the motif is built, they have served their purpose and **must be removed
before the structure is exported to RFdiffusion2**:

- RFdiffusion2 has no concept of a virtual / occupied-volume-only atom. Anything
  present in the ligand HETATM block is parsed by OpenBabel as a real atom, given
  a real (perceived) bond graph, and fed to the model as chemical context. Leaving
  `CV1`/`CV2` in would inject two spurious carbons with fabricated bonds into the
  active-site chemistry the network conditions on.
- Their geometric intent (where scaffold anchor carbons should sit) belongs in the
  export **manifest / metadata**, not in the coordinates handed to the model.

Guidance:

- Scrub `CV1`/`CV2` as the last step of motif construction, before RFdiffusion
  export.
- If they are useful as occupied-volume sentinels for **post-generation
  filtering**, retain their coordinates in the manifest only — do not include them
  in the RFdiffusion input PDB.
- `src/swacanatase/placement.py` already isolates them
  (`BP5_VIRTUAL_CARBON_ATOMS = {"CV1", "CV2"}`); reuse that set for the scrub.

## Carbon nanoring: write explicit `CONECT` records

The carbon nanoring is generated by the sibling **`tuber`** repo. RFdiffusion2
derives ligand bonds from OpenBabel, which perceives bonds **by interatomic
distance** unless explicit `CONECT` records are present. For a large ring this
distance-based perception is unreliable (mis-bonded / over-bonded).

Plan:

- Have `tuber` emit nanoring structures **with explicit `CONECT` records** for the
  intended ring connectivity.
- Make the corresponding change in `swacanatase` so the export path **preserves
  those `CONECT` records** through to the RFdiffusion2 input PDB rather than
  dropping them.

Note: the full ring (M = 18–36) is likely out-of-distribution for RFdiffusion2's
small-molecule ligand handling. This is an accepted risk for the first
experiments; explicit connectivity at least ensures the ring is represented with
the geometry we intend.
