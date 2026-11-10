# AGENTS.md

## What this repo is

Documentation-only Quarto deliverable for the ADS 5984 final project (NFL "Opponent Tendencies Analyst" design). There is no application code, package manifest, test suite, lint config, or CI. Do not invent or run build/test/lint commands beyond Quarto.

All deliverables live in `docs/`:
- `docs/final_project_reqs_and_arch_zachary_and_adam.qmd` — the only editable source
- `docs/final_project_reqs_and_arch_zachary_and_adam.pdf` — rendered deliverable (committed)
- `docs/opponent-tendencies-sequence-diagram.png` — rendered sequence diagram (committed)

(The files currently also appear as root-level deletions / untracked `docs/` in `git status` because the move out of the repo root is uncommitted; don't recreate them at the root.)

## Build

Render from the repo root (Quarto 1.8.24 is installed at `/opt/quarto/bin/quarto`):

```bash
quarto render docs/final_project_reqs_and_arch_zachary_and_adam.qmd
```

The PDF is written next to the `.qmd` (`docs/`). It is a graded deliverable, so re-render and commit it whenever the `.qmd` changes. Never hand-edit the PDF or PNG; regenerate them.

Quick config check without rendering: `quarto inspect docs/final_project_reqs_and_arch_zachary_and_adam.qmd`.

## Editing gotchas

- Requirements are numbered 1–30 and issues 1–18. Later sections ("LLM / RAG Issues", "Solution Mitigation") reference those numbers explicitly. If you add, remove, or reorder an item, update every cross-reference.
- The Mermaid `sequenceDiagram` in the `.qmd` and `docs/opponent-tendencies-sequence-diagram.png` depict the same flow. If you change one, update the other.
- The diagram is referenced by a relative filename, so the PNG must stay in `docs/` beside the `.qmd`.
- Assignment text in the `.qmd` states the write-up must be produced without AI assistance. Treat content generation accordingly and limit agent help to repo mechanics unless the user says otherwise.
