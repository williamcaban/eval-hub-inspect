# Agent Guide — eval-hub-inspect

This file is the primary entry point for AI coding agents (Claude Code, Codex, agentic
harnesses, etc.) working in this repository. Read it in full before making any changes.

If you are uncertain about anything while working, write your uncertainty to a file called
`UNCERTAINTIES.md` in the relevant benchmark directory before proceeding. A human reviewer
will resolve it during code review.

---

## What this repository is

`eval-hub-inspect` is a collection of [inspect_ai](https://inspect.ai-safety-institute.org.uk/)
`@task` implementations for AI evaluation benchmarks. Each benchmark lives in its own
subdirectory under `src/` and produces standardized inspect_ai evaluation results.

Two audiences, one codebase:

1. **Research / AISI community** — run `inspect eval` directly; benchmarks with arXiv papers
   can be registered in the [inspect_evals register](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/register).
2. **EvalHub on RHOAI** — the same inspect_ai tasks flow into EvalHub through the existing
   [Inspect AI adapter](https://github.com/eval-hub/eval-hub-contrib/tree/main/adapters/inspect)
   without a separate native adapter.

---

## Repository map

```
eval-hub-inspect/
│
├── AGENTS.md                    ← you are here; read first
├── README.md                    ← user-facing repo overview
├── pyproject.toml               ← dependencies, ruff/mypy/pytest config, uv settings
├── uv.lock                      ← pinned dependency lock file (always commit this)
├── .pre-commit-config.yaml      ← hooks: ruff, file hygiene, CVE check
│
├── scripts/
│   ├── check_dep_cves.py        ← CVE audit tool (run before committing new deps)
│   └── download_data.py         ← benchmark data download helper (to be created per benchmark)
│
├── src/
│   ├── BENCHMARKS.md            ← master index of all benchmarks (keep current)
│   └── <benchmark_name>/
│       ├── __init__.py          ← exports the @task function(s)
│       ├── task.py              ← @task entry point
│       ├── _dataset.py          ← dataset loading → inspect_ai Sample objects
│       ├── _scorer.py           ← Scorer implementation
│       ├── metrics/             ← ported/custom evaluation metric functions (if needed)
│       │   └── __init__.py
│       ├── docker/
│       │   ├── Dockerfile       ← agent execution environment
│       │   └── compose.yaml     ← inspect_ai sandbox configuration
│       └── README.md            ← benchmark description, citations, license (REQUIRED)
│
├── tests/
│   ├── conftest.py              ← shared fixtures
│   └── <benchmark_name>/
│       ├── test_dataset.py
│       ├── test_scorer.py
│       ├── test_metrics.py
│       └── test_task.py
│
└── .github/
    ├── dependabot.yml           ← automated dependency update PRs
    └── workflows/
        ├── ci.yml               ← lint, typecheck, bandit, CVE check, tests
        └── dep-cve-check.yml    ← standalone CVE audit on dep file changes
```

---

## Toolchain

Always use `uv run` — never bare `python`, `pip`, or `pytest`.

| Action | Command |
|---|---|
| Install / sync dev deps | `uv sync --group dev` |
| Run a Python script | `uv run python scripts/foo.py` |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check .` |
| Auto-fix lint | `uv run ruff check . --fix` |
| Format | `uv run ruff format .` |
| Type-check | `uv run mypy src/ scripts/` |
| Security scan | `uv run bandit -r src/ scripts/ -ll` |
| CVE audit (deps) | `uv run python scripts/check_dep_cves.py --fix` |
| Add a dependency | `uv add <package>` then commit `uv.lock` |

**Python version**: 3.12 minimum. Do not use syntax or APIs exclusive to 3.13+.

---

## Quality gates — must pass before any commit

Run these in order. A commit that fails any gate will be blocked by CI.

```bash
# 1. Lint (includes bandit-S security rules, isort, bugbear, pyupgrade)
uv run ruff check .

# 2. Format
uv run ruff format --check .

# 3. Type checking (strict)
uv run mypy src/ scripts/

# 4. Tests
uv run pytest

# 5. CVE audit — REQUIRED whenever pyproject.toml or uv.lock changes
uv run python scripts/check_dep_cves.py --fix
```

Pre-commit hooks run gates 1, 2, and 5 automatically. Install them once:

```bash
uv run pre-commit install
```

### CVE audit rules

- `check_dep_cves.py` queries the [OSV database](https://osv.dev) for every minimum version
  pinned in `pyproject.toml` and `requirements*.txt`.
- Exit code `0` = all clean. Exit code `1` = vulnerable version found.
- **Never** lower a version pin without re-running the CVE audit.
- If `--fix` suggests a safe minimum, apply it and re-audit before committing.
- The audit runs on every PR that touches `pyproject.toml`, `uv.lock`, or requirements files.
- Use `--strict` in CI (fails on network error); omit locally (warns when offline).

### Security annotation rule

The benchmark scorer uses `eval()` on strings from the curated benchmark dataset.
This is intentional and safe — the strings are static, not user input. Each call site
must carry an inline suppression with explanation:

```python
# S307: eval_func is a static string from the benchmark dataset, not user input.
result = eval(func_str, metric_namespace)  # noqa: S307
```

Do **not** add a global `S307` ignore in `pyproject.toml`. The per-site annotation
keeps the pattern visible during code review.

---

## Workflow: Run an existing benchmark

```bash
# 1. Download benchmark data (one-time per benchmark)
uv run python scripts/download_data.py --benchmark <name>

# 2. Run with a limit first to validate the setup
uv run inspect eval src/<name>/task.py@<task_name> --limit 1 --model openai/gpt-4o-mini

# 3. Examine the transcript — confirm the agent can submit a valid result
uv run inspect view

# 4. Full run
uv run inspect eval src/<name>/task.py@<task_name> --model openai/gpt-4o
```

A successful run means: the evaluation completes, scores are non-zero for at least some
samples, and the log contains no `ERRORED` samples from infrastructure failures.

---

## Workflow: Add a new benchmark

Follow every step in order. Skipping steps creates incomplete submissions that fail CI
or the inspect_evals register review.

### Step 1 — Research the benchmark

Before writing any code:

1. Read the benchmark's paper (find the arXiv link — **required** for inspect_evals register).
2. Clone or read the upstream repository to understand:
   - Task format (input structure, output format)
   - Evaluation logic (how scores are computed)
   - Dataset access (HuggingFace, direct download, or private)
   - License of the benchmark code and data
3. Confirm the license allows redistribution/porting. If not, note the limitation in `README.md`.
4. Identify the eval_func types (e.g., `compare_csv`, LLM judge, test runner).

### Step 2 — Create the directory structure

```bash
BENCH=<benchmark_name>   # lowercase, underscores, e.g. "agenticdatabench"
mkdir -p src/$BENCH/metrics src/$BENCH/docker tests/$BENCH
touch src/$BENCH/__init__.py
touch src/$BENCH/task.py
touch src/$BENCH/_dataset.py
touch src/$BENCH/_scorer.py
touch src/$BENCH/metrics/__init__.py
touch src/$BENCH/docker/Dockerfile
touch src/$BENCH/docker/compose.yaml
touch tests/$BENCH/__init__.py
touch tests/$BENCH/test_task.py
touch tests/$BENCH/test_dataset.py
touch tests/$BENCH/test_scorer.py
```

### Step 3 — Implement `task.py`

The `@task` function is the inspect_ai entry point. Required pattern:

```python
from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import bash
from ._dataset import load_dataset
from ._scorer import benchmark_scorer

@task
def <benchmark_name>(
    split: str = "dev",
    domain: str | None = None,
) -> Task:
    return Task(
        dataset=load_dataset(split=split, domain=domain),
        solver=[
            system_message(SYSTEM_PROMPT),
            use_tools(bash()),
            generate(),
        ],
        scorer=benchmark_scorer(),
        sandbox=("docker", "src/<benchmark_name>/docker/compose.yaml"),
        max_messages=30,
    )
```

Rules:
- The `@task` function name must match the directory name and the name used in `inspect eval`.
- Expose filtering parameters (e.g., `domain`, `split`) as typed keyword arguments.
- Set `max_messages` — never leave it unbounded for agentic evals.
- Use `sandbox=("docker", "compose.yaml_path")` for any eval that executes untrusted code.

### Step 4 — Implement `_dataset.py`

Return a list of `Sample` objects. Each sample must include:

```python
Sample(
    id=str(task["id"]),  # unique, stable identifier
    input=formatted_prompt,  # what the model/agent sees
    target="",  # use "" if scoring is file-based (not text match)
    metadata={  # everything the scorer needs
        "eval_func": ...,
        "output_file_name": ...,
        "gold_file_name": ...,
        "domain": ...,
    },
)
```

Load data from HuggingFace using `huggingface_hub.hf_hub_download()` with a pinned
`revision=` (commit SHA or tag). Never load from a mutable branch reference.

### Step 5 — Implement `_scorer.py`

```python
from inspect_ai.scorer import scorer, Score, Scorer, mean, stderr
from inspect_ai.solver import TaskState
from inspect_ai.scorer import Target
from inspect_ai.util import sandbox


@scorer(metrics=[mean(), stderr()])
def benchmark_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # Read output files from the sandbox
        content = await sandbox().read_file("/workspace/output.csv", text=False)
        # Compare against gold standard
        # Return Score with float value in [0, 1]
        return Score(value=0.0, explanation="...")

    return score
```

Scoring rules:
- `Score.value` must be a float in [0.0, 1.0].
- Return `Score.unscored()` when the evaluation instrument itself fails (e.g., gold file missing), not `Score(value=0.0)`. The latter falsely attributes instrument failure to the model.
- Return `Score(value=0.0)` when the model produced an output but it is wrong.
- Set `Score.explanation` to a human-readable string explaining the result.
- Always use `metrics=[mean(), stderr()]` unless the benchmark has a specific metric requirement.

### Step 6 — Write the Dockerfile and compose.yaml

The Docker image provides the agent's execution environment.

**`docker/Dockerfile`** minimum:
```dockerfile
FROM python:3.12-slim
WORKDIR /workspace
RUN pip install --no-cache-dir pandas numpy scikit-learn \
    && useradd -m agent
USER agent
```

**`docker/compose.yaml`** minimum:
```yaml
services:
  default:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ${BENCHMARK_DATA:-./data/<name>}/datasets:/datasets:ro
      - ${BENCHMARK_DATA:-./data/<name>}/gold:/gold:ro
    working_dir: /workspace
```

Rules:
- Mount dataset directories read-only (`:ro`).
- Never run the container as root in production; add a non-root `USER`.
- All dataset paths must be configurable via environment variables with local defaults.

### Step 7 — Write `__init__.py`

```python
from .task import <benchmark_name>

__all__ = ["<benchmark_name>"]
```

### Step 8 — Write `README.md` for the benchmark

**Every benchmark README must contain all of the following sections.** Missing sections
will cause the PR to be sent back.

```markdown
# <Benchmark Name> — inspect_ai wrapper

One-paragraph description of what the benchmark measures and why it matters.

## Benchmark overview

Table: Tasks, Domains, Scale, Score range, Evaluation metrics.

## Original benchmark resources

Table: GitHub, Project page (if exists), HuggingFace dataset (if public), arXiv paper.

## Citation

BibTeX for the benchmark paper. If the evaluation metrics were ported from a different
project, include a second BibTeX entry for that project.

## License

Three-column table: Component | License | Source
- Benchmark data and code
- Metrics module (if ported from elsewhere)
- This inspect_ai wrapper (Apache 2.0)

## Data setup

Exact commands to download and structure benchmark data.
Show the resulting directory tree.

## Running the evaluation

At minimum: single-sample sanity check command and full-run command.
Include any task parameters (split, domain, etc.).

## Module structure

Directory tree with one-line description of each file.
```

### Step 9 — Write tests

Every benchmark needs at minimum:

| Test file | What to test |
|---|---|
| `test_task.py` | `@task` returns a `Task`; `Task.dataset` is non-empty; `Task.sandbox` is set |
| `test_dataset.py` | Samples have required fields (`id`, `input`, `metadata`); no missing eval_func |
| `test_scorer.py` | Scorer returns `Score` in [0, 1]; `Score.unscored()` on missing files; known inputs give expected score |
| `test_metrics.py` | Each metric function (compare_csv, etc.) with fixture data; edge cases (empty file, missing column) |

Use `pytest` fixtures for synthetic samples — never require the full 27+ GB dataset in
unit tests. Mock `sandbox().read_file()` using `unittest.mock.AsyncMock`.

### Step 10 — Update indexes

1. Add benchmark to `src/BENCHMARKS.md` — one row in the table, one entry in the quick-reference section.
2. Export the `@task` from `src/<benchmark_name>/__init__.py`.

### Step 11 — Add dependencies and re-audit CVEs

If the benchmark requires new packages:

```bash
uv add <package>                                     # adds to pyproject.toml, updates uv.lock
uv run python scripts/check_dep_cves.py --fix        # audit the new minimum version
```

If the CVE check flags a vulnerability, bump the minimum to the suggested safe version
and re-audit before committing.

### Step 12 — Run all quality gates

```bash
uv run ruff check . --fix
uv run ruff format .
uv run mypy src/ scripts/
uv run pytest
uv run python scripts/check_dep_cves.py
```

All must pass with exit code 0. Fix failures before opening a PR.

---

## inspect_evals register eligibility

To register a benchmark in the [inspect_evals register](https://github.com/UKGovernmentBEIS/inspect_evals)
(which makes it discoverable via the AISI docs), the following must be true:

| Requirement | Check |
|---|---|
| arXiv paper | `README.md` has a valid `https://arxiv.org/abs/NNNN.NNNNN` link |
| Public dataset | Datasets downloadable without authentication |
| Assets pinned | HuggingFace loads use `revision=<sha>`, not `revision="main"` |
| You are a contributor | You have at least one commit in this repo |
| Eval runs end-to-end | `inspect eval ... --limit 1` completes without crashing |
| Log files | Two full run logs uploaded to the PR (can use inexpensive models) |

Benchmarks without an arXiv paper can still live in this repo and run locally or via
EvalHub — they just cannot be submitted to the inspect_evals register.

---

## Coding standards

### Imports

Group in this order (ruff `I` rule enforces this automatically):
1. stdlib
2. third-party
3. local (`from ._dataset import ...`)

### Type annotations

- All function signatures must be fully annotated.
- Use `X | None` not `Optional[X]` (Python 3.10+ union syntax).
- Use `list[X]`, `dict[K, V]` not `List[X]`, `Dict[K, V]`.
- `mypy --strict` must pass. Use `# type: ignore[<code>]` only when unavoidable; add an explanation comment.

### Async

inspect_ai scorers are async. Use `await sandbox().read_file(...)`, not blocking file I/O.
Do not mix sync and async file access in the same scorer.

### eval() usage

`eval()` in scorers is permitted for benchmark evaluation functions because the strings
come from a static, curated dataset — not user input. Every call site must carry:

```python
# S307: eval_func string is from the curated benchmark dataset, not user input.
result = eval(func_str, namespace)  # noqa: S307
```

### zip() usage

Always pass `strict=True` when zipping two lists that must have equal length:

```python
for sample, result in zip(samples, results, strict=True):
    ...
```

### No bare `except`

```python
# Bad
except Exception:
    pass

# Good — log the failure so the problem is visible
except Exception as exc:
    print(f"Warning: {exc}", file=sys.stderr)
```

---

## Common agent mistakes to avoid

| Mistake | Correct approach |
|---|---|
| Using `python` instead of `uv run python` | Always `uv run python` |
| Adding deps with `pip install` | `uv add <pkg>` |
| Ignoring CVE audit output | Fix the flagged version before committing |
| Returning `Score(value=0.0)` for infrastructure failures | Use `Score.unscored()` |
| Using `revision="main"` in HuggingFace loads | Use a pinned commit SHA |
| Writing `except Exception: pass` | Log the exception |
| Skipping `README.md` sections | All 7 sections are required |
| Forgetting to update `src/BENCHMARKS.md` | Update the index table |
| Using `Optional[X]` instead of `X | None` | Use `X | None` (Python 3.10+) |
| Using bare `zip()` | `zip(..., strict=True)` |
| `S307` global ignore in pyproject.toml | Per-site `# noqa: S307` with explanation |

---

## Getting help

If you are an agent and you are uncertain about anything:

1. Write `UNCERTAINTIES.md` in the affected benchmark directory listing each uncertainty.
2. Continue with what you are confident about.
3. Do not guess on licensing, citation authorship, or CVE mitigations — leave these for human review.
