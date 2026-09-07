# Turn Motif Implementation Plan

## Goal

Extend the current BP5 secondary-structure analysis from regular repeated
alpha-helix and beta-strand torsion targets to stable non-regular tight-turn
motifs. The first implementation should test whether BP5 can occupy a central
position in canonical beta-turn and gamma-turn backbones while preserving the
existing BP5/Pd/nanoring catalytic geometry and the existing clash/cylinder
filters.

## Implementation Status

Complete:

- Phase 1: static motif library and turn builder.
- Phase 2: placement integration and reuse of existing clash/cylinder scoring.
- Phase 3: CLI, generated turn outputs, `turn_motif_scores.csv`, and metadata.
- Phase 4: BetaTurnLib18 18-cluster mode/medoid values, deterministic
  phi/psi perturbation scans, and opt-in cis-Pro motif support with non-trans
  omega and fixed Pro residue labels.
- Acceptance tests 1-7.

No pending implementation items remain for this plan.

Future extension: chemically complete Pro sidechain construction for fixed-Pro
cis motifs. The current Phase 4 support fixes residue identity and peptide
omega but emits backbone-only generated Pro residues.

## Background

The existing implementation treats secondary structure as a repeated phi/psi
target. This is appropriate for alpha helices and beta strands, but not for
tight turns. Turn motifs require per-position torsion targets:

- Beta turns are 4-residue motifs. The canonical geometry is defined mainly by
  the phi/psi values of residues 2 and 3, conventionally `i+1` and `i+2`.
- Gamma turns are 3-residue motifs. The canonical geometry is defined mainly by
  the phi/psi value of the middle residue, conventionally `i+1`.
- Cis-Pro beta-turn classes also require omega constraints and residue identity
  constraints, so they should not be treated as ordinary trans-peptide motifs.

Relevant existing code:

- `src/swacanatase/secondary_structure.py` currently defines
  `SecondaryStructureType = Literal["alpha_helix", "beta_strand"]` and
  `SECONDARY_STRUCTURE_TARGETS` as one phi/psi pair per motif.
- `build_regular_secondary_structure_segment()` grows a deterministic backbone
  around a fixed BP5 frame using one repeated torsion target.
- `src/swacanatase/placement.py` expands rotamer states by combining each BP5
  rotamer state with a `BackboneTorsionTargets` entry, then scores the resulting
  generated segments against BP5, neighboring backbone, scaffold, and nanoring
  cylinder constraints.

## Initial Motif Set

Use the following as the first deterministic motif library. Values are degrees.
The beta-turn values are classical first-pass targets, with names mapped to the
newer BetaTurnLib18 nomenclature where useful.

| Motif | Length | BP5 candidate positions | Torsion targets | Notes |
| --- | ---: | --- | --- | --- |
| `beta_turn_ad` | 4 | 2, 3 | r2 `(-60, -30)`, r3 `(-90, 0)` | Classical Type I; highest-priority baseline. |
| `beta_turn_pd` | 4 | 2 | r2 `(-60, 120)`, r3 `(+80, 0)` | Classical Type II, Gly-like at r3; BP5 best tested at r2 first. |
| `beta_turn_pa` | 4 | 2 | r2 `(-60, 120)`, r3 approx `(+60, +30)` | Modern Type II subtype; r3 often non-Gly but positive phi still risky for BP5. |
| `beta_turn_ad_prime` | 4 | optional | r2 `(+60, +30)`, r3 `(+90, 0)` | Classical Type I prime; likely poor for BP5 unless testing mirror-like cases. |
| `beta_turn_pd_prime` | 4 | 3 | r2 `(+60, -120)`, r3 `(-80, 0)` | Classical Type II prime; BP5 at r3 is plausible. |
| `beta_turn_ab1` | 4 | 2, 3 | r2 near `(-67, -31)`, r3 near `(-136, 162)` | Type VIII-derived modern subtype. |
| `beta_turn_ab2` | 4 | 2, 3 | r2 near `(-69, -30)`, r3 near `(-120, 128)` | Type VIII-derived modern subtype. |
| `beta_turn_az` | 4 | 2, 3 | r2 near `(-74, -28)`, r3 near `(-140, 75)` | Type VIII-derived modern subtype, often pre-Pro-like at following position. |
| `beta_turn_ag` | 4 | 2, 3 | r2 near `(-66, -19)`, r3 near `(-82, 63)` | Type VIII-derived modern subtype. |
| `gamma_turn_inverse` | 3 | 2 | middle residue approx `(-75, +65)` | Higher-priority gamma turn; negative phi is more BP5-compatible. |
| `gamma_turn_classic` | 3 | optional | middle residue approx `(+75, -65)` | Lower-priority due positive phi. |

Defer the cis-Pro classes (`PcisD`, `BcisP`, `PcisP`, `cisDA`, `cisDP`) until
the builder supports per-peptide omega targets and fixed residue identities.

## Data Model

Add turn-specific model classes instead of overloading the regular repeated
`BackboneTorsionTargets`.

Recommended structures:

```python
TurnMotifType = Literal[
    "beta_turn_ad",
    "beta_turn_pd",
    "beta_turn_pa",
    "beta_turn_ad_prime",
    "beta_turn_pd_prime",
    "beta_turn_ab1",
    "beta_turn_ab2",
    "beta_turn_az",
    "beta_turn_ag",
    "gamma_turn_inverse",
    "gamma_turn_classic",
]

@dataclass(frozen=True)
class ResidueTorsionTargets:
    residue_offset: int
    phi_degrees: float | None
    psi_degrees: float | None
    omega_after_degrees: float = 180.0

@dataclass(frozen=True)
class TurnMotifDefinition:
    motif_type: TurnMotifType
    length: int
    central_bp5_offsets: tuple[int, ...]
    residue_torsions: tuple[ResidueTorsionTargets, ...]
    label: str
    previous_name: str | None = None
    source: str = "classical"
```

Use 1-based residue offsets within each turn motif. For a 4-residue beta turn,
BP5 offsets 2 and 3 correspond to `i+1` and `i+2`.

Keep these definitions separate from the current
`SECONDARY_STRUCTURE_TARGETS`. A regular secondary structure has a repeatable
single-residue torsion target; a turn has a finite ordered torsion sequence.

## Builder Design

Add a new builder rather than modifying
`build_regular_secondary_structure_segment()` in-place:

```python
def build_turn_motif_segment(
    bp5_rotamer: BP5RotamerPlacement | struc.AtomArray,
    motif_type: TurnMotifType,
    bp5_motif_offset: int,
    chain_id: str = "A",
    starting_residue_id: int = 1,
    starting_atom_id: int = 1,
) -> TurnMotifSegment:
    ...
```

Implementation outline:

1. Load the selected `TurnMotifDefinition`.
2. Validate that `bp5_motif_offset` is allowed by `central_bp5_offsets`.
3. Treat the placed BP5 backbone atoms as the fixed residue at that offset.
4. Grow residues N-terminally and C-terminally using the existing internal
   coordinate helpers where possible.
5. Apply the residue-specific phi/psi values to the central motif residues.
6. Use trans omega by default.
7. Generate simple Gly residues for non-BP5 positions initially, because side
   chains are not needed for first-pass steric scoring and Gly avoids adding
   false side-chain clashes.
8. Add terminal atoms using the same conventions as the regular segment builder.
9. Return metadata containing motif type, BP5 offset, source torsions, residue
   count, and BP5 residue id.

If existing helper functions only support repeated phi/psi growth, factor out a
lower-level "append previous residue" and "append next residue" primitive that
accepts explicit target torsions. Keep the current regular builder as a thin
caller using repeated targets so existing tests remain stable.

## Scanning Strategy

Add turn motif scanning as a sibling mode to regular secondary-structure
scanning.

Recommended CLI shape:

```bash
uv run swacanatase-generate-bp5-nanorings \
  --turn-motif beta_turn_ad beta_turn_ab1 gamma_turn_inverse \
  --turn-bp5-position central \
  --write-reports
```

Suggested options:

- `--turn-motif`: one or more motif names; default can be a conservative first
  set of `beta_turn_ad`, `beta_turn_ab1`, `beta_turn_ab2`,
  `gamma_turn_inverse`.
- `--turn-bp5-position`: `central`, `all`, or an explicit 1-based motif offset.
  `central` means offsets 2 and 3 for beta turns, offset 2 for gamma turns.
- `--turn-scan-limit`: optional cap analogous to current `--scan-limit`, or
  reuse the global scan limit after documenting the behavior.
- `--include-cis-turns`: initially reject with a clear error until omega and
  residue identity constraints are implemented.

State expansion should become:

```text
rotamer state x turn motif x allowed BP5 motif offset
```

Each expanded state should build one full symmetric set of turn segments around
the nanoring, exactly like the current secondary-structure state builder does
for helices/strands.

## Scoring And Reports

Reuse existing scoring components:

- scaffold clash score
- BP5 clash score
- neighboring backbone clash score
- nanoring cylinder intrusion score
- orientation metrics

Add turn-specific report columns:

- `motif_type`
- `motif_length`
- `bp5_motif_offset`
- `turn_label`
- `turn_previous_name`
- `turn_source`
- `turn_phi_psi_r2`
- `turn_phi_psi_r3`
- `turn_middle_phi_psi` for gamma turns
- `turn_contains_positive_phi_for_bp5`
- `turn_requires_cis_peptide`

Keep regular secondary-structure CSV columns unchanged for backward
compatibility. Either write a new `turn_motif_scores.csv` report or add a
`motif_family` column only if all downstream report consumers can tolerate it.
A separate CSV is lower risk.

## Acceptance Tests

Add focused tests before broad generated-output tests:

1. `test_turn_motif_definitions_have_valid_bp5_offsets`
   - Every motif has length 3 or 4.
   - Every BP5 offset is within motif length.
   - Every non-cis first-pass motif has trans omega.

2. `test_beta_turn_builder_places_bp5_at_requested_central_offset`
   - Build `beta_turn_ad` with BP5 at offset 2 and offset 3.
   - Confirm total residue count is 4.
   - Confirm BP5 residue id matches the requested offset.

3. `test_generated_beta_turn_geometry_matches_targets`
   - Measure phi/psi of residues 2 and 3 for `beta_turn_ad`.
   - Assert values match the motif definition within the same angular tolerance
     used by regular secondary-structure tests.

4. `test_gamma_turn_builder_places_bp5_at_middle_position`
   - Build `gamma_turn_inverse`.
   - Confirm total residue count is 3.
   - Confirm BP5 is residue 2.
   - Measure middle phi/psi.

5. `test_turn_motif_segments_reuse_clash_and_cylinder_scoring`
   - Place at least one motif around an M=18 nanoring with one rotamer state.
   - Assert the state has candidates, scores, and cylinder metrics.

6. `test_turn_motif_cli_writes_outputs_and_report`
   - Run the placement CLI with `--turn-motif beta_turn_ad --scan-limit 1`.
   - Assert a turn output directory and `turn_motif_scores.csv` are created.

7. `test_cis_turns_must_be_explicitly_enabled`
   - Request a cis-Pro motif without the opt-in flag.
   - Assert the error tells the user to pass `--include-cis-turns`.

## Implementation Phases

### Phase 1: Static motif library and builder

- Add turn dataclasses and motif definitions to `secondary_structure.py` or a
  new `turn_motifs.py`.
- Factor reusable internal-coordinate growth helpers if needed.
- Implement `build_turn_motif_segment()`.
- Add unit tests for motif definitions and measured torsions.

### Phase 2: Placement integration

- Add turn motif state dataclasses mirroring the current secondary-structure
  placement state shape.
- Add a `_build_symmetric_turn_motif_states()` function in `placement.py`.
- Reuse the existing scaffold/BP5/backbone/cylinder scoring functions.
- Add `turn_motif_candidates`, `turn_motif_states`, and accepted-state fields
  to the placement result object.

### Phase 3: CLI and output reports

- Add CLI flags for turn motif scanning.
- Write generated turn complexes under `data/generated/turn_motif/`.
- Write `data/generated/reports/turn_motif_scores.csv`.
- Include turn scan metadata in `run_metadata.json`.

### Phase 4: Motif refinement

- Add BetaTurnLib18 modal values for the 18-cluster library, not just the
  classical first-pass set.
- Add optional medoid-vs-mode selection.
- Add optional small perturbation scans around motif modes, for example 5-degree
  grids within a narrow angular radius.
- Add support for cis-Pro turn classes by allowing non-180 omega and fixed Pro
  residue placement.

## Modeling Notes

- Treat BP5 at positive-phi positions as lower priority. Positive phi positions
  are often Gly-like in natural turns, and BP5 is bulky.
- For beta turns, BP5 at position 2 is likely most useful for Type I/AD,
  Type II/Pd/Pa, and Type VIII-derived AB/AZ/AG turns.
- BP5 at position 3 is likely most useful for Type I/AD, Type II prime/pD, and
  Type VIII-derived AB/AZ/AG turns.
- Gamma-turn inverse is more plausible than classical gamma for BP5 because its
  middle-residue phi is negative.
- Do not hand-edit BP5 ligand coordinates. The turn builder should only grow
  peptide backbone around the already placed BP5 rotamer frame.

## Sources

- Shapovalov M, Vucetic S, Dunbrack RL Jr. "A new clustering and nomenclature
  for beta turns derived from high-resolution protein structures." PLOS
  Computational Biology, 2019.
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006844
- BetaTurnLib18 motif library.
  https://github.com/sh-maxim/BetaTurn18/blob/master/BetaTurnLib18/BetaTurnLib18.txt
- Fang C, Shang Y, Xu D. "Improving Protein Gamma-Turn Prediction Using
  Inception Capsule Networks." Scientific Reports, 2018.
  https://www.nature.com/articles/s41598-018-34114-2
