# Agent reading and change rules

## Reading order

Read `README.md`, `docs/EXPERIMENTS.md`, and the target experiment's `docs/experiments/E*.md`.
Use `catalog/files.json` to find file roles, CLI flags, imports, symbols, source and publication hashes.
Every `.py` and `.sh` must have a linked per-file card under `docs/files/`.

## Version boundaries

This is an archive of distinct protocols, NOT one unified implementation.
Do not merge similarly named modules across E05/E06 or E08/E09/E22/E23.
Do not infer scientific results from filenames or successful syntax checks.
Respect missing/partial provenance. No invented original implementations or results.
Keep scoring definitions (EM/F1/lexical ACC/semantic review/FanOut metrics) explicit.
Do not silently alter budgets, tokenization, truncation, fallback, root reuse, seed, or search limits.

## Safety

By default inspect source and run `python3 tools/verify_repository.py` only.
Do not launch model/retriever services, install dependencies, download assets, call paid APIs,
or execute historical experiment/analysis entrypoints without a user request scoped to that work.
Some Python files perform I/O at import time: use AST parsing instead of importing for inspection.
Never commit raw trajectories, gold labels, model weights, indexes, credentials or private paths.
Do not use `git add .` without checking the exact diff and ignore rules.

## Updating documentation

For any source change update the corresponding file card and `catalog/files.json` published hash.
Retain the historical original hash; it is provenance, not the checksum of a modified publication copy.
New/changed experiments require an experiment page, inputs/outputs, protocol differences and validation notes.
Use per-directory README indexes for humans; use JSON catalogs for automated navigation.
Do not claim full runtime verification from static validation.
