# Symmetric Backbone Generator Spike

## Decision

Neither evaluated generator currently satisfies the project's requirement for a
rigid, high-precision C9 backbone scaffold around the M18 BP5/Pd/nanoring
theozyme.

RFdiffusion2 does not support symmetry in its public inference interface.
ProteinGenerator runs successfully with a C9 sequence repeat, but its current
symmetry code does not constrain the generated coordinates to C9.

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

## Consequence

ProteinGenerator can provide sequence-symmetric candidate backbones for
exploration, subject to motif-frame, clash, pore-intrusion, and symmetry
filters. It cannot by itself meet the required C9 coordinate constraint.

The next production-capable approach must either apply a downstream rigid Cn
projection followed by complete geometry validation, or use/modify a generator
that samples and scores one ASU before explicitly rotating it into the full C9
assembly.
