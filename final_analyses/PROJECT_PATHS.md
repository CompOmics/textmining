# Portable project paths

The committed SapBERT benchmark and PRIDE-context result manifests use the
literal `${HAMLET_ROOT}` prefix in place of the original cluster directory.
It denotes the HAMLET project directory containing `textmining/`, not the
`textmining` repository itself. Dataset identifiers, source file names,
checksums and numerical results are unchanged.

Set and export `HAMLET_ROOT` to your local HAMLET directory before submitting
the production and analysis batch scripts. Batch logs default to the submission
directory; use `sbatch --output=...` to select a different log destination.
Slurm directives do not expand shell variables.

JSON paths are descriptive strings, not automatically expanded environment
variables. Consumers must resolve this prefix explicitly. The SapBERT Staircase
figure builder resolves it against the local project root.

Raw production outputs, model caches and large runtime directories are not
included in this commit. The included comparison tables and summaries preserve
the data underlying the final figures; recomputing extraction or audits still
requires the corresponding source inputs and outputs.
