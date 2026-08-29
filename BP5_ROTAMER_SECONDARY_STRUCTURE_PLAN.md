# BP5 Chi Rotamer and Secondary-Structure Implementation Plan

## Objective

Add a first-pass protein context around the current BP5/Pd nanoring placement by
sampling PHE-like `chi1` and `chi2` rotamers for BP5 before growing regular
alpha-helix or beta-strand backbone segments.

The active BP5/Pd/nanoring geometry remains the primary constraint. Rotamer
sampling should move the amino-acid backbone frame relative to the already placed
BP5 active-site group, not relax or re-place the nanoring-bound pharmacophore.

## Scope For This Pass

- Sample only BP5 `chi1` and `chi2`.
- Keep the bipyridine inter-ring state exactly as the implemented active
  coplanar cis conformer.
- Do not sweep the `C3-C6` inter-ring torsion.
- Do not introduce empirical fragment databases yet.
- Use deterministic rotamer enumeration and clash filtering before any
  secondary-structure growth.

## BP5 Atom Mapping

Status: Complete for the first implementation pass, including `C11` branch
validation as a measured partner torsion rather than an independent degree of
freedom.

The local RCSB CCD BP5 atom names map cleanly onto a PHE-like sidechain for the
first two chi angles:

| Concept | PHE-like atom | BP5 atom |
| --- | --- | --- |
| backbone amide nitrogen | `N` | `N` |
| alpha carbon | `CA` | `CA` |
| beta carbon | `CB` | `C12` |
| aromatic attachment carbon | `CG` | `C9` |
| first aromatic branch atom | `CD1` convention | `C8` |
| alternate aromatic branch atom | `CD2` convention | `C11` |

Use these signed torsion definitions:

- `chi1`: `N-CA-C12-C9`
- `chi2`: `CA-C12-C9-C8`

`C11` should be retained as a validation partner for the aromatic branch
orientation, but it should not define an independent degree of freedom.

The active bipyridine conformer should continue to satisfy:

- `N1-C3-C6-N2` near `0 degrees`
- all bipyridine ring atoms nearly coplanar
- appended `PD`, `CV1`, and `CV2` in the same square-planar coordination frame

## Core Modeling Principle

The current placement code fixes BP5/Pd to nanoring scaffold anchors through the
virtual carbons `CV1` and `CV2`. That part should remain unchanged.

After a BP5/Pd unit has been placed:

1. Keep the nanoring anchor fit fixed.
2. Keep the bipyridine rings, `PD`, `CV1`, and `CV2` fixed in space.
3. Enumerate backbone positions by changing `chi2` and then `chi1`.
4. Score/reject rotamer states based on clashes and secondary-structure
   compatibility.

This inverts the usual protein-building view. Instead of rotating the sidechain
off a fixed backbone, the catalytic sidechain is fixed by the nanoring model and
the residue backbone frame is rotated outward through allowed chi states.

## Torsion-Setting Mechanics

Status: Complete for regular alpha-helix/beta-strand builders and first-pass
post-growth candidate clash scoring.

Add a small, tested torsion utility rather than embedding torsion math directly
inside placement code.

Recommended module:

- `src/swacanatase/torsions.py`

Recommended functions:

- `measure_dihedral(atom_array, atom_names) -> float`
- `set_dihedral_by_rotating_component(atom_array, bond_pairs, atom_names, target_degrees, moving_side_atom) -> AtomArray`
- `connected_component_after_removing_bond(bond_pairs, blocked_bond, seed_atom) -> set[str]`

Implementation details:

- Use the structured BP5 bond records from `load_bp5_bond_pairs()`.
- Use Rodrigues rotation or the existing private axis-rotation logic from
  `active_site.py`/`placement.py`, promoted to a shared helper if useful.
- Do not parse PDB/mmCIF text by hand.
- Rotate only the intended connected component for each chi angle.

For fixed-active-site sampling:

1. Set `chi2` first.
   - Axis: `C12-C9`
   - Fixed side: `C9` plus bipyridine, `PD`, `CV1`, `CV2`
   - Moving side: `C12`, `CA`, backbone atoms, and `C12` hydrogens
   - Rotate all moving-side atoms except axis atoms `C12` and `C9`

2. Set `chi1` second.
   - Axis: `CA-C12`
   - Fixed side: `C12` plus bipyridine, `PD`, `CV1`, `CV2`
   - Moving side: `CA`, `N`, `C`, `O`, terminal atoms, and `CA` hydrogen
   - Rotate all moving-side atoms except axis atoms `CA` and `C12`

This order lets `chi2` choose where the whole residue frame exits the aromatic
system, then lets `chi1` orient the peptide backbone around the `CA-C12` bond.

## Rotamer Enumeration

Status: Complete for the first implementation pass.

Recommended module:

- `src/swacanatase/bp5_rotamers.py`

Recommended data structures:

```python
@dataclass(frozen=True)
class BP5ChiRotamer:
    name: str
    chi1_degrees: float
    chi2_degrees: float
    weight: float | None = None


@dataclass(frozen=True)
class BP5RotamerPlacement:
    residue_id: int
    rotamer: BP5ChiRotamer
    atom_array: struc.AtomArray
    chi1_degrees: float
    chi2_degrees: float
    clash_score: float
```

Initial PHE-like rotamer grid:

| name | `chi1` | `chi2` |
| --- | ---: | ---: |
| `gminus_m90` | `-60` | `-90` |
| `gminus_p90` | `-60` | `90` |
| `gplus_m90` | `60` | `-90` |
| `gplus_p90` | `60` | `90` |
| `trans_m90` | `180` | `-90` |
| `trans_p90` | `180` | `90` |

Keep this grid deterministic. Later, empirical PHE rotamer probabilities can be
added as weights or ranking priors without changing the geometric machinery.

## Placement Integration

Status: Complete for the first implementation pass. The opt-in nanoring rotamer
ensemble workflow is in place, including rigid BP5/Pd placement reuse, per-site
rotamer enumeration, realized chi measurement, full symmetric-state clash
scoring, cutoff filtering, top-`k` selection, secondary-structure candidate
generation, and full-state secondary-structure clash scoring.

Extend the current workflow in `src/swacanatase/placement.py` without replacing
the existing rigid BP5 placement behavior.

Recommended new public entry point:

- `place_bp5_rotamer_ensembles_around_nanoring(...) -> BP5NanoringRotamerPlacement`

Recommended behavior:

1. Generate the nanoring and select anchor pairs exactly as today.
2. Build the active BP5/Pd unit exactly as today.
3. Rigidly place one active BP5/Pd unit at each anchor exactly as today.
4. For each placed residue, enumerate the BP5 chi rotamer grid.
5. Measure `chi1`/`chi2` after each transform and store the realized values.
6. Clash-score each complete symmetric state against:
   - the carbon scaffold
   - BP5/Pd atoms within and across all retained positions
   - generated secondary-structure atoms against BP5 context and across all
     retained positions
7. Keep either all accepted symmetric rotamer states or the top `k` symmetric
   states, with every retained state represented at every BP5 site.

The existing `place_bp5_sidechains_around_nanoring()` should remain available as
the rigid baseline and as a source of regression tests.

## Clash Scoring

Status: Complete for the first implementation pass.

Start with a simple deterministic heavy-atom clash score.

Suggested rules:

- Ignore bonded atom pairs within a single BP5 residue.
- Ignore `CV1`/`CV2` overlap residuals with their intended scaffold anchors when
  evaluating the established active-site fit.
- Ignore all intra-residue atom pairs for placement cutoff scoring.
- Ignore Pd contacts to the two scaffold carbons targeted by `CV1` and `CV2`.
- Ignore backbone `N`/`C` atom pairs from different residues.
- Treat `PD` separately from organic atoms, with an explicit minimum distance
  threshold.
- Use conservative element-pair distance cutoffs rather than full force-field
  energies.
- Report both a boolean pass/fail and a numeric overlap score.

Recommended module:

- `src/swacanatase/clashes.py`

Initial output fields:

- total clash score
- number of clashing pairs
- worst overlap
- labels for the worst atom pair

## Secondary-Structure Growth

Status: Complete for the first implementation pass, including terminal atom
normalization, post-growth clash scoring, and orientation metrics.

Only after BP5 rotamer candidates are generated and filtered should regular
secondary structure be attached.

Recommended module:

- `src/swacanatase/secondary_structure.py`

Inputs:

- an accepted BP5 rotamer placement
- residue frame from BP5 `N`, `CA`, `C`, and `O`
- secondary-structure type: `alpha_helix` or `beta_strand`
- residue count before and after the BP5 residue

Initial deterministic backbone targets:

| type | `phi` | `psi` |
| --- | ---: | ---: |
| alpha helix | `-60` | `-45` |
| beta strand | `-135` | `135` |

Implemented follow-up feature: secondary-structure growth can scan explicit
`phi`/`psi` targets on a sparse grid around each ideal basin and persists the
selected `phi`/`psi` values in the secondary-structure score report. The default
scan level uses a conservative favored core; a broader allowed basin remains
available for exploratory runs.

For internal protein segments, replace terminal-only BP5 atoms as needed:

- [x] drop `OXT` and `HXT` when BP5 is embedded in a chain
- [x] normalize terminal hydrogens in generated peptides rather than preserving CCD
  terminal annotations blindly

Secondary-structure candidates are scored and measured after growth, not before:

- [x] backbone-scaffold clashes
- [x] backbone-BP5 clashes
- [x] backbone-backbone clashes between neighboring BP5 sites
- [x] helix axis or strand direction relative to nanoring radial/tangential/axial
  frames
- [x] N-terminal and C-terminal exit vectors

## Testing Plan

Add tests in small increments.

Torsion tests:

- [x] `measure_dihedral()` matches the existing test helper convention.
- [x] Setting `chi1` reaches the requested value without moving `C12`, `C9`, `N1`,
  `N2`, `PD`, `CV1`, or `CV2`.
- [x] Setting `chi2` reaches the requested value without moving `C9`, `N1`, `N2`,
  `PD`, `CV1`, or `CV2`.
- [x] Applying `chi2` then `chi1` reaches both requested values.
- [x] `N1-C3-C6-N2` remains near `0 degrees` after every rotamer transform.
- [x] The bipyridine ring atoms remain coplanar after every rotamer transform.

Rotamer tests:

- [x] The default BP5 rotamer grid has six deterministic states.
- [x] Enumerating rotamers from a placed active-site unit preserves `CV1`/`CV2`
  placement and Pd geometry.
- [x] Realized `chi1`/`chi2` values are within a tight angular tolerance of their
  targets.
- [x] Atom annotations, residue IDs, chain IDs, and atom IDs remain deterministic.

Placement tests:

- [x] The current rigid placement tests continue to pass unchanged.
- [x] For `M=18`, every anchor site can produce six raw rotamer candidates.
- [x] Clash filtering returns deterministic accepted/top-`k` candidate sets.
- [x] A deliberately overlapping candidate produces a nonzero clash score.

Secondary-structure tests:

- [x] Alpha-helix and beta-strand builders produce the requested residue counts.
- [x] Generated peptide bond lengths, backbone angles, and `phi`/`psi` values are
  within deterministic tolerances.
- [x] The BP5 residue frame remains aligned to the accepted rotamer after segment
  growth.
- [x] Post-growth clash scoring reports scaffold, BP5-context, and neighboring
  candidate-backbone components.
- [x] Post-growth orientation metrics report segment direction, anchor-frame
  alignment, and terminal exit vectors.

## CLI And Output Plan

Status: Complete for the first implementation pass.

Keep the existing generation CLI stable. Add rotamer and secondary-structure
outputs as opt-in modes.

Possible options:

- `--enumerate-bp5-rotamers`
- `--max-rotamers-per-site K`, interpreted as top `K` symmetric rotamer states
  across the full ring
- `--rotamer-clash-cutoff VALUE`, interpreted as total overlap per BP5 site
- `--secondary-structure {none,alpha_helix,beta_strand}`
- `--residues-before N`
- `--residues-after N`
- `--secondary-structure-clash-cutoff VALUE`, interpreted as total overlap per
  BP5 site
- `--no-clash-cutoffs`

Suggested output directories:

- `data/generated/theozyme/` for rigid BP5/Pd baseline structures
- `data/generated/rotamers/` for full symmetric BP5 chi-rotamer ensembles
- `data/generated/secondary_structure/` for full symmetric helix/strand segment
  models

Generated complexes should preserve the existing chain convention:

- chain `A`: BP5 residues and generated protein segment atoms
- chain `B`: carbon scaffold

## Implementation Stages

1. [x] Add shared torsion measurement and component-rotation helpers.
2. [x] Add BP5 chi constants and default PHE-like rotamer definitions.
3. [x] Add BP5 rotamer enumeration from an already placed active-site unit.
4. [x] Add active-site preservation and chi-realization tests.
5. [x] Add heavy-atom clash scoring.
6. [x] Integrate rotamer enumeration into nanoring placement as an opt-in workflow.
7. [x] Add regular alpha-helix and beta-strand builders from accepted BP5 residue
   frames.
8. [x] Add CLI options and generated-output paths.

## Acceptance Criteria

- [x] Existing rigid BP5/Pd nanoring placement behavior remains unchanged.
- [x] BP5 rotamer enumeration preserves the nanoring anchor fit and Pd coordination
  geometry.
- [x] Only `chi1` and `chi2` vary in this pass.
- [x] The inter-ring BP5 active conformer remains coplanar cis with no torsion
  sweep.
- [x] The code exposes deterministic, testable geometry functions rather than a
  notebook-only workflow.
- [x] `uv run --extra dev pytest` passes before handoff once implementation begins.
