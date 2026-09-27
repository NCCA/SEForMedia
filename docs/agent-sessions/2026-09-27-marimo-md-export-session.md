# 2026-09-27 marimo markdown export session

## Goal

Add a pre-commit hook that exports each marimo notebook to markdown in a separate folder and imports it into Joplin under the SEForMedia notebook.

## Files changed

- `scripts/export_marimo_md.py` (new) runs `marimo export md` on staged notebooks into `MarkdownExports/`, then creates or updates a note per notebook in Joplin via the Data API, with one sub-notebook per repo folder
- `.pre-commit-config.yaml` adds the local `export-marimo-markdown` hook
- `.gitignore` ignores `MarkdownExports/`

## Commands run

```bash
git worktree add .worktrees/marimo-md-export -b agent/marimo-md-export
ruff check scripts && ruff format scripts
python scripts/export_marimo_md.py --all --no-joplin     # 46 notebooks exported
python scripts/export_marimo_md.py IntroToMarimo/part1.py MNIST/Sketch/SketchNumbersMarimo.py Utils/__init__.py
python scripts/export_marimo_md.py IntroToMarimo/part1.py  # second run updates rather than duplicates
pre-commit validate-config
```

There is no test suite in the repo, so no tests were run.
