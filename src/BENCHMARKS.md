# Benchmark Index

All inspect_ai task implementations in this repository.
Each benchmark has its own subdirectory under `src/` with a `README.md` covering
the benchmark description, original resources, paper citations, and license.

## Available benchmarks

| Benchmark | Domain | Tasks | arXiv | inspect_ai task name | Status |
|---|---|---|---|---|---|
| [AgenticDataBench](agenticdatabench/README.md) | Agentic data science | 246 public | [2607.01647](https://arxiv.org/abs/2607.01647) | `agenticdatabench` | ✅ Available |

## Quick reference

### AgenticDataBench

Evaluates LLM agents on realistic multi-step data analysis tasks across 15 domains.
Agents write Python code, execute it in a sandboxed environment, and produce output files
that are compared against gold standards using domain-specific metrics.

```bash
# Data setup (required once — 27.3 GB download)
uv run python scripts/download_data.py --benchmark agenticdatabench

# Run evaluation
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
  --limit 5 --model openai/gpt-4o

# Domain filter
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
  -T domain=finance --model openai/gpt-4o
```

**Citation**: Sun et al. 2026 — arXiv:2607.01647
**License**: Apache 2.0 (benchmark data) · MIT (metrics module from DA-Code)

---

## Adding a new benchmark

1. Create `src/<benchmark_name>/` with:
   - `__init__.py` — exports the `@task` function
   - `task.py` — `@task`-decorated function
   - `README.md` — benchmark description, citations, license (see [AgenticDataBench](agenticdatabench/README.md) as template)
2. Add the benchmark to the table above.
3. Add the `@task` function name to `src/__init__.py`.
4. Run `uv run python scripts/check_dep_cves.py` if new dependencies were added.
5. Check inspect_evals register eligibility: the benchmark must have an arXiv paper
   and you must have at least one commit in this repo.
   See [register/README.md](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md).
