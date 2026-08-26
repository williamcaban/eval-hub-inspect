# Contributing to eval-hub-inspect

All changes to `main` go through pull requests — direct pushes are blocked by branch protection.

## Quick start

```bash
git clone https://github.com/williamcaban/eval-hub-inspect
cd eval-hub-inspect
uv sync --group dev        # install dev tools
uv run pre-commit install  # wire up local quality hooks
```

## Branching convention

| Prefix | Use for |
|---|---|
| `feat/<name>` | New benchmark or feature |
| `fix/<name>` | Bug fix |
| `docs/<name>` | Documentation only |
| `chore/<name>` | Dependency bumps, CI changes |

```bash
git checkout -b feat/my-benchmark
```

## Development cycle

```bash
# 1. Make changes, then run quality gates manually before pushing
uv run ruff check . --fix   # lint (auto-fix safe violations)
uv run ruff format .         # format
uv run mypy src/ scripts/    # type check
uv run pytest                # tests
uv run python scripts/check_dep_cves.py --fix   # CVE audit (if deps changed)

# 2. Push and open a PR
git push -u origin feat/my-benchmark
gh pr create --title "feat: add <benchmark>" --body "..."
```

Pre-commit hooks run ruff and the CVE audit automatically on every `git commit`.

## Required CI checks

Every PR must pass all five checks before merging:

| Check | What it runs |
|---|---|
| **Ruff lint & format** | `ruff check . && ruff format --check .` |
| **mypy** | `mypy src/ scripts/` (strict) |
| **pytest** | `pytest tests/` |
| **Dependency CVE check** | `scripts/check_dep_cves.py --strict` against OSV |
| **Bandit security scan** | `bandit -r src/ scripts/ -ll` → SARIF to GitHub Security tab |

If a check is red, fix the issue and `git push` — CI re-runs automatically.

## Merging

With 0 required approvals configured, you can merge your own PR once all checks are green:

```bash
gh pr merge --squash        # squash-merge (keeps main history clean)
# or merge via the GitHub UI
```

Use squash-merge for feature/fix PRs. Use regular merge for release-style PRs where individual commits carry meaning.

## Adding a new benchmark

Follow the 12-step workflow in [AGENTS.md](AGENTS.md#workflow-add-a-new-benchmark). The short version:

1. `git checkout -b feat/<benchmark-name>`
2. Create `src/<benchmark_name>/` following the structure in `AGENTS.md`
3. Write `src/<benchmark_name>/README.md` (all 7 sections required — see `AGENTS.md`)
4. Add an entry to `src/BENCHMARKS.md`
5. Write tests in `tests/<benchmark_name>/`
6. Run `uv run python scripts/check_dep_cves.py --fix` if you added deps
7. Pass all quality gates
8. Open a PR

## Dependency changes

Whenever `pyproject.toml` or `uv.lock` changes:

```bash
uv add <package>                                  # adds to pyproject.toml + updates uv.lock
uv run python scripts/check_dep_cves.py --fix    # audit new minimum version
```

The CVE audit runs automatically in CI on every PR that touches dependency files and posts
a Markdown table comment if any minimum version has a known vulnerability.

## Branch protection rules (main)

| Rule | Value |
|---|---|
| Direct push to `main` | Blocked |
| Required PR approvals | 0 (self-merge after CI passes) |
| Required status checks | All 5 CI jobs must be green |
| Branch must be up-to-date | Yes (strict mode) |
| Force push | Blocked |
| Branch deletion | Blocked |
| Admin bypass | Available (emergency use only) |
