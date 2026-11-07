# RFdiffusion Export Notes

Decisions that affect how motif/active-site structures are prepared before they
are handed to an RFdiffusion-family generator. See
`docs/rfdiffusion_motif_scaffolding_plan.md` for the full pipeline plan.

The C9 production target is now **RFdiffusion3**, not RFdiffusion2; see
[RFdiffusion3 is the C9 route](#rfdiffusion3-is-the-c9-route-2026-11-06) below
and [the symmetric-backbone-generator spike](docs/symmetric_backbone_generator_spike.md).
The RFdiffusion2 findings in this file remain valid for the asymmetric
single-ASU control and still describe how the motif itself must be prepared.

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

## RFdiffusion3 is the C9 route (2026-11-06)

RFdiffusion3 supports coordinate-level Cn symmetry *and* all-atom ligand/metal
conditioning, which is the combination this project needs and the combination
RFdiffusion2 lacks. It is cloned as a sibling checkout at
`/home/broom/AlphaCarbon/foundry` (RFD3 lives at `foundry/models/rfd3`).

**It does give us what we want, and it is verified.** A C9 unconditional job of
nine 90-residue subunits ran on the workstation GTX 1080 Ti in 388 seconds and
is symmetric to 0.04 Angstrom per-subunit CA RMSD about the Z axis through the
origin. Nine 180-residue subunits exhaust the card, and the production M18
target sits just above the validated size, so plan production sweeps on a
40-80 GB class card. Note that `length` in symmetry mode is **per subunit**, not
per assembly. Full installation pins, hardware measurements, and the local
upstream-compatibility patch are recorded in
[the symmetric-backbone-generator spike](docs/symmetric_backbone_generator_spike.md).

What changes for the exporter, relative to the RFdiffusion2 contract below:

1. **The nanoring is passed once, not nine times.** `is_unsym_motif` names
   motifs and ligands that the symmetry operator must not replicate. The nine
   BP5/Pd motifs are symmetrized; the nanoring is the shared on-axis ligand.
2. **The motif must be pre-symmetrized and centered on the origin.** RFD3 only
   supports `is_symmetric_motif: true` and rejects asymmetric motif inputs.
   There is no `ORI` token to emit; centering is on the origin by convention.
3. **Ligand chemistry is resolved through AtomWorks and the CCD, not
   OpenBabel.** A ligand whose residue name matches a CCD code is matched to the
   CCD; the documented escape hatch for anything else is a residue name that is
   invalid for the CCD (for example `C:T`), which makes AtomWorks use the input
   geometry verbatim. The carbon nanoring needs that escape hatch. This replaces
   the RFdiffusion2 OpenBabel bond-perception problem with a different one, so
   the loaded bond graph still has to be verified explicitly.
4. **Atom-level fixing is explicit.** `select_fixed_atoms` pins named atoms per
   residue (for example the BP5 donor nitrogens and `PD`) rather than relying on
   whole-residue motif semantics.
5. **The last sampling steps are not symmetrized.** `sym_step_frac` defaults to
   `0.9`. Either set it to `1.0` or keep a downstream rigid Cn projection plus
   complete geometry validation.

Unchanged from the RFdiffusion2 work: scrub `CV1`/`CV2` before export, keep
their coordinates in the manifest only, and compute motif RMSD on the `N/CA/C`
frame rather than naive all-backbone-including-`O`.

## Validated C9 turn-motif scaffolding run (2026-11-07)

An accepted M18 turn-motif state was scaffolded end-to-end with RFD3 symmetry
mode on the workstation GTX 1080 Ti. This is the first C9 result for this
project that preserves the theozyme.

State: `nanoring_M18_gplus_m90_beta_turn_ad_bp5pos2` (a default turn motif,
trans, BP5 at motif position 2, zero clash and zero ring-slab intrusion in the
`full_clash_0_0_m12_20` scan). Contig `30,A1-4,30`, so 64 residues per subunit
and 576 in the assembly, plus nine Pd and the 108-atom ring. Completed in 891
seconds.

The input builder is `scripts/spike_export_rfd3_input.py` (throwaway; informs
the real `src/swacanatase/rfdiffusion.py`). Outputs are preserved under
`data/generated/rfdiffusion3/c9_turn_motif_spike/`.

### Results

| Quantity | Value |
| --- | --- |
| Motif `N/CA/C` frame RMSD, all nine sites | 0.0246-0.0585 Angstrom |
| BP5 donor `N1`/`N2` deviation | 0.018-0.049 Angstrom |
| Pd-N distances, input -> output | 2.018 -> 1.983 +/- 0.022 and 2.026 +/- 0.016 Angstrom |
| Nanoring internal rigidity | 0.0199 Angstrom, with a 1.61 Angstrom rigid-body centering shift |
| Per-subunit CA RMSD across the nine chains | 0.032-0.042 Angstrom |
| Protein atoms in the ring slab | 0 |

RFD3 co-designs sequence with structure in the same pass, so the output chains
carry real residue identities rather than poly-glycine. That removes the
separate inverse-folding step the RFdiffusion2 plan assumed, though
LigandMPNN/RF3 remain available in the same `foundry` checkout for redesign and
refolding.

### Input contract: five rules the exporter must honor

1. **Nine polypeptide subunit chains.** `rfd3/inference/symmetry/frames.py`
   derives symmetry frames from atoms with `chain_type == 6`. A lone BP5 residue
   per chain is not a polymer, so the frame search finds no entities and fails
   with `ValueError: min() iterable argument is empty`. A real turn motif, four
   peptide-linked residues per chain, satisfies this. This is why turn-motif
   states, not bare BP5/Pd sites, are the correct starting point.
2. **BP5 is written as `ATOM`, not `HETATM`.** Its CCD type is
   `L-PEPTIDE LINKING`. As a heteroatom, AtomWorks splits it out of the chain,
   the remaining `GLY` residues become non-contiguous, and its own chain-info
   consistency check fails with a residue-name multiset mismatch.
3. **The nanoring needs a non-CCD residue name.** `C:T` is invalid as a CCD
   code, so AtomWorks keeps our geometry verbatim ("Residue 'C:T' is not in the
   CCD, keeping current atom order"). Name it in `symmetry.is_unsym_motif` and
   it stays a single shared molecule on the axis; the output contained exactly
   one 108-atom copy.
4. **Ring carbons need unique atom names.** `tuber` names every nanoring carbon
   `C`. Duplicate atom names inside one residue break both CONECT mapping and
   AtomWorks, so renumber them `C1..CN`.
5. **Pd rides on the subunit chain as its own residue.** AtomWorks moves it to a
   separate chain during parsing and RFD3 logs
   `Component 'PD' maps to multiple residues. If you are using Symmetry this is
   OK.` Both are expected.

### Output chain labels are a permutation of input chain labels

The symmetry operator does not preserve chain naming. In the validated run the
mapping was reflected: output `A->A`, `B->I`, `C->H`, `D->G`, `E->F`, `F->E`,
`G->D`, `H->C`, `I->B`. A geometry filter that compares output chain `X` to
input chain `X` reports 11-33 Angstrom errors on eight of the nine sites. Match
each output chain to its best input chain over all nine before scoring.

### `MIN_ATOMS_ALIGN` excludes small motifs on a stock checkout

`rfd3/inference/symmetry/checks.py` requires at least 30 atoms in the reference
entity to derive frames. A 4-residue beta turn ASU is 29 heavy atoms (three
`GLY` backbones at 4 each, plus BP5's 17), and a 3-residue gamma turn is 25.
**Both fall below the stock threshold.** Our `foundry` checkout lowers it to 12
on branch `local/atomworks-3.0.4-compat` (commit `c179d91`), which is safe here
because the nine subunit copies are exactly symmetric by construction, so the
rigid frame is heavily over-determined. On an unpatched checkout, only the
7-residue `alpha_helix`/`beta_strand` scan states are large enough.

### Ring-slab intrusion scoring is insufficient for scaffolded designs

`NanoringCylinderIntrusionScore` tests the ring's own thin z-slab, which the
validated design passes with zero atoms, and genuinely so near the active site:
the pore is clear through +/-5 Angstrom. Further along the axis it is not.

```text
pore radius 8.136 A:  ring slab 0 | +/-5 A 0 | +/-10 A 18 | unbounded 105 atoms
closest designed atom to the axis: LEU16 CD2, radius 5.89 A, 23.8 A above the ring plane
```

The protein caps the tube roughly 20-25 Angstrom out from the active site. For a
tube enzyme that must admit substrate or thread a growing nanoring along the
axis, that is a design-acceptability question, so extend the intrusion metric
along the axis (a tall cylinder, or unbounded) when filtering scaffolded
designs. The slab form passes trivially once the motif itself is clash-free.

Note also that the tightest protein-ring contacts, 2.93-2.95 Angstrom, are the
BP5 donor nitrogens themselves and are catalytic by design. No designed residue
comes closer, so a naive minimum-distance clash filter must exempt the motif.

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

Under RFdiffusion3 this guidance still holds but for a different reason: bonds
come from AtomWorks rather than OpenBabel, and a non-CCD residue name makes
AtomWorks take the input geometry verbatim, so explicit connectivity in the
input is what determines the ring's bond graph. Verify the loaded graph either
way.
