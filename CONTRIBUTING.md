# Contributing

Thanks for your interest in SPECTRA.

## Development setup
```bash
make setup            # editable install + dev tools
pre-commit install    # enable formatting/lint hooks
make test
```

## Ground rules
- **Never commit data, checkpoints, or credentials.** `.gitignore` blocks common
  offenders and a pre-commit hook rejects files > 5 MB. Datasets are obtained
  separately (see `data/README.md`).
- Keep paths portable — read locations from `spectra.config.PATHS` / env vars,
  not hardcoded absolute paths.
- Run `make lint` and `make test` before opening a PR.

## Commit style
Short imperative subject lines (e.g. "Add hetero-graph structure backbone").
