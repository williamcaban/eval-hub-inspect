# eval-hub-inspect

inspect_ai task implementations for AI evaluation benchmarks — runnable standalone
for research and EvalHub-compatible via the
[Inspect AI adapter](https://github.com/eval-hub/eval-hub-contrib).

## What this is

Each benchmark in this repository:
- implements one or more [inspect_ai](https://inspect.ai-safety-institute.org.uk/) `@task` functions
- can be run directly with `inspect eval` on any machine with Docker
- is eligible for registration in the [inspect_evals register](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/register) (benchmarks with arXiv papers)
- flows into [EvalHub](https://github.com/eval-hub/eval-hub) on RHOAI via the existing Inspect adapter without a separate native adapter

## Benchmarks

See **[src/BENCHMARKS.md](src/BENCHMARKS.md)** for the full index, quick-start commands, and
per-benchmark citations and licenses.

| Benchmark | Tasks | Domain | Status |
|---|---|---|---|
| [AgenticDataBench](src/agenticdatabench/README.md) | 246 public | Agentic data science | ✅ Available |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
git clone https://github.com/williamcaban/eval-hub-inspect
cd eval-hub-inspect

# Install dependencies (Python 3.12, inspect_ai, dev tools)
uv sync --group dev

# Verify everything is wired up
uv run python -c "import inspect_ai; print(inspect_ai.__version__)"
```

## Running evaluations

```bash
# Download benchmark data (one-time, per benchmark)
uv run python scripts/download_data.py --benchmark agenticdatabench

# Run a quick 1-sample sanity check
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
  --limit 1 --model openai/gpt-4o

# View results
uv run inspect view
```

## Code quality

All commits pass:
- **ruff** — lint + format (includes bandit-S security rules)
- **mypy** strict — type checking
- **bandit** — standalone security scan → GitHub Security tab (SARIF)
- **OSV CVE audit** — dependency minimums checked against the OSV database on every `pyproject.toml` change

Run locally:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/ scripts/
uv run python scripts/check_dep_cves.py --fix
```

Install pre-commit hooks (runs automatically on every commit):

```bash
uv run pre-commit install
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
Each benchmark's original code and data retains its own license;
see the benchmark's `README.md` for details.
