# Incremental PRIDE-context comparison, 1,737 projects

Recover earlier raw provider responses from the third-retry DocETL cache by exact
MD5 request keys, using preserved abstract/method inputs and the matching operation
configuration (600-second timeout, no timeout retry, bypass_cache true). Decode only
JSON string literals via pickletools; never execute pickle objects. Record database,
prompt/input/output hashes and per-PXD cache keys. Abort full reporting on missing
keys, conflicting cache payloads, nonempty WALs or incomplete latest output coverage.

Earlier inputs: 601 abstract+methods without PRIDE; 1,136 abstract+PRIDE-property
fallback. Latest: abstract/methods plus PRIDE properties and available sample/data
processing protocols. Latest raw output root is recorded in summary.json.

This is NOT a causal PRIDE-only ablation. Multi-value prompt/schema behavior changed,
and recovered provider responses do not restore final earlier QC/repair outputs.
No model inference, normalization, benchmark scoring or source-value correction occurs.

Compare populated fields separately from lexical value-set changes. Casefold/NFKC/
whitespace normalization and semicolon splitting apply symmetrically; no comma splitting
or embeddings. Lexical additions do not necessarily represent new scientific concepts.
Field coverage counts extraction schema fields, including any aliases, not independent
biological facts. Newly populated fields receive conservative evidence attribution:
literal quoted text in earlier input, only in newer PRIDE sections, both, or unverified.
Attribution measures source support/availability, not causal necessity or correctness.

Each run has a separate runs/job_JOBID directory with recovered_earlier_raw, mapping
provenance, missing cache keys, field_changes.csv, project_changes.json and summary.json.
COMPLETE is written only after all 5,211 responses map and all 1,737 projects compare.
No changes to the production output folders or original caches are required.
