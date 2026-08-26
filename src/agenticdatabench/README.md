# AgenticDataBench — inspect_ai wrapper

inspect_ai task implementation of **AgenticDataBench**, a comprehensive benchmark for evaluating LLM-based data agents on realistic, multi-step data science workflows with fine-grained skill-level labels.

## Benchmark overview

| Property | Value |
|---|---|
| Tasks (public) | 246 (of 344 total; 98 withheld as private test set) |
| Domains | 15 (agriculture, finance, healthcare, logistics, …) |
| B2B fintech use cases | 5 real-world scenarios |
| Data science skills | 433 distinct skills extracted from Stack Overflow |
| Input datasets | 97 datasets, 27.3 GB total |
| Evaluation metrics | `compare_csv`, `compare_json`, `compare_image`, `compare_model`, `compare_text`, `compare_sqlite` |
| Score range | 0.0 – 1.0 continuous (averaged across `eval_func` calls per task) |

### What it evaluates

Agents receive a data analysis question and a list of input CSV/database files. They must write and execute Python code that produces one or more output files (`output.csv`, `output.json`, …) whose contents are compared against gold-standard references using the benchmark's evaluation functions.

The benchmark distinguishes itself from narrow SQL-translation benchmarks by testing the full analytical pipeline: data loading, joining, cleaning, statistical computation, visualization, and business-context reasoning — at the level of individual data science skills, not just end-to-end accuracy.

## Original benchmark resources

| Resource | Link |
|---|---|
| GitHub | <https://github.com/AgenticDataBench/AgenticDataBench> |
| Project page | <https://agenticdatabench.github.io> |
| HuggingFace dataset | <https://huggingface.co/datasets/shawnzzzh/AgenticDataBench> |
| arXiv paper | <https://arxiv.org/abs/2607.01647> |

## Citation

If you use this wrapper or the underlying benchmark, please cite the original paper:

```bibtex
@misc{sun2026agenticdatabench,
  title        = {AgenticDataBench: A Comprehensive Benchmark for Data Agents},
  author       = {Zhaoyan Sun and Shan Zhong and Daizhou Wen and Jiaxing Han and
                  Guoliang Li and Ying Yan and Peng Zhang and Yu Su and Xiang Qi and
                  Baolin Sun and Chengyuan Yang and Tao Fang and Huaiyu Ruan},
  year         = {2026},
  eprint       = {2607.01647},
  archivePrefix = {arXiv},
  primaryClass  = {cs.DB},
  url          = {https://arxiv.org/abs/2607.01647}
}
```

The evaluation metrics (`compare_csv`, `compare_json`, etc.) in `metrics/` are ported from
[DA-Code](https://github.com/yiyihum/da-code) (EMNLP 2024). If you use the metrics module
directly, please also cite:

```bibtex
@inproceedings{luo2024dacode,
  title     = {DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models},
  author    = {Luo, Jianwen and others},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024},
  url       = {https://github.com/yiyihum/da-code}
}
```

## License

| Component | License | Source |
|---|---|---|
| AgenticDataBench benchmark data and code | [Apache 2.0](https://github.com/AgenticDataBench/AgenticDataBench/blob/main/LICENSE) | Tsinghua University / Ant Group |
| DA-Code evaluation metrics (ported to `metrics/`) | [MIT](https://github.com/yiyihum/da-code/blob/main/LICENSE) | Jianwen Luo et al. |
| This inspect_ai wrapper | [Apache 2.0](../../LICENSE) | eval-hub-inspect contributors |

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| [Docker Engine](https://docs.docker.com/engine/install/) | Any current | Required for the agent sandbox |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.11 | Python package manager |
| Model API key | — | e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` |
| Disk space | ~30 GB | For dataset download + Docker image |

## Step 1 — Install dependencies

```bash
# From the repo root
uv sync --extra agenticdatabench --group dev
```

This installs inspect_ai, pandas, scikit-learn, opencv, and all other required libraries.

## Step 2 — Get the data

### Option A — Download from HuggingFace (network-connected)

```bash
uv run python scripts/download_data.py --benchmark agenticdatabench
```

Downloads ~27.3 GB to `./data/agenticdatabench/`. Use `--data-dir` to override:

```bash
uv run python scripts/download_data.py --benchmark agenticdatabench \
    --data-dir /mnt/fast-ssd/agenticdatabench
```

### Option B — Airgapped / offline environments

Download the data on a connected machine, then transfer it to the target:

```bash
# 1. Download on a connected machine
uv run python scripts/download_data.py --benchmark agenticdatabench \
    --data-dir /tmp/adb-transfer

# 2. Transfer to the airgapped machine (example using rsync)
rsync -av /tmp/adb-transfer/ airgapped-host:/data/agenticdatabench/

# 3. On the airgapped machine, set the data path via environment variable
export AGENTICDATABENCH_DATA=/data/agenticdatabench
```

The `AGENTICDATABENCH_DATA` environment variable is read by `docker/compose.yaml` to mount
the data directories into the agent container. No other configuration is needed.

### Option C — Manual layout (bring-your-own data)

If you already have the files, arrange them as follows and pass `--data-dir`:

```
<data_root>/
├── tasks/
│   └── dev.jsonl            ← 246-task JSONL file
├── datasets/
│   └── <task_id>/           ← one sub-directory per task
│       └── <data_files>     ← CSV, parquet, shapefiles, …
└── gold/
    └── <task_id>/           ← one sub-directory per task
        └── result.csv       ← gold-standard output file(s)
```

## Step 3 — Build the Docker image

The first run will build the sandbox image automatically. To pre-build:

```bash
docker compose -f src/agenticdatabench/docker/compose.yaml build
```

## Step 4 — Run the evaluation

```bash
# Quick sanity check — 1 task
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=./data/agenticdatabench \
    --limit 1 \
    --model openai/gpt-4o-mini

# Larger run — 10 tasks
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=./data/agenticdatabench \
    --limit 10 \
    --model openai/gpt-4o

# All 246 public tasks
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=./data/agenticdatabench \
    --model openai/gpt-4o

# Filter to a single domain
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=./data/agenticdatabench \
    -T domain=agriculture \
    --model openai/gpt-4o

# Using an environment variable for the data path (airgapped / non-default location)
AGENTICDATABENCH_DATA=/data/agenticdatabench \
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=/data/agenticdatabench \
    --model anthropic/claude-sonnet-4-6

# Parallel execution (4 sandboxes simultaneously)
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=./data/agenticdatabench \
    --limit 20 \
    --max-sandboxes 4 \
    --model openai/gpt-4o
```

### Task parameters

| Parameter | Default | Description |
|---|---|---|
| `data_dir` | `None` (downloads from HuggingFace) | Path to pre-downloaded data root |
| `split` | `"dev"` | Dataset split (`"dev"` is the only public split) |
| `domain` | `None` (all domains) | Restrict to one domain (e.g. `"agriculture"`) |

### Available domains

`agriculture`, `biology`, `chemistry`, `economics`, `education`, `engineering`,
`environment`, `finance`, `healthcare`, `history`, `logistics`, `physics`, `sociology`,
`sports`, `technology`

## Step 5 — View results

```bash
# Open the interactive log viewer
uv run inspect view

# Or list recent runs
uv run inspect list logs
```

Results include per-task scores (0.0–1.0), error messages from the metric functions,
and the full agent transcript showing the code the agent wrote and executed.

---

## Data location summary

| Method | Where data lives | How compose.yaml finds it |
|---|---|---|
| Default download | `./data/agenticdatabench/` | `AGENTICDATABENCH_DATA` defaults to `./data/agenticdatabench` |
| `--data-dir /custom/path` | `/custom/path/` | Pass same path via `-T data_dir=` and set env var |
| Airgapped transfer | Anywhere on the target machine | `export AGENTICDATABENCH_DATA=/that/path` |

The Docker compose mounts two read-only volumes into each agent container:
- `$AGENTICDATABENCH_DATA/datasets/` → `/datasets/` (input data)
- `$AGENTICDATABENCH_DATA/gold/` → `/gold/` (reference files for scoring)

---

## CLI run status

| Check | Status |
|---|---|
| Task discoverable by `inspect list tasks` | ✅ Confirmed |
| Task loads with synthetic data (no network) | ✅ Confirmed |
| Pipeline reaches Docker sandbox correctly | ✅ Confirmed (`inspect eval` runs, stops at Docker requirement) |
| Full end-to-end with real data + model | Requires Docker Engine + data download |

To replicate the CLI sanity check without real data:

```bash
# Create minimal synthetic data
mkdir -p /tmp/adb-test/tasks /tmp/adb-test/datasets/smoke_01 /tmp/adb-test/gold/smoke_01
echo '{"id":"smoke_01","question":"Sum column value, save to output.csv","data_sources":["data.csv"],"skills":[],"domain":"test","output_file_name":["output.csv"],"gold_file_name":["result.csv"],"eval_func":["compare_csv(output_file_name='"'"'output.csv'"'"',gold_file_name='"'"'result.csv'"'"',ignore_order=True)"]}' \
    > /tmp/adb-test/tasks/dev.jsonl
printf "total\n10\n" > /tmp/adb-test/gold/smoke_01/result.csv
printf "value\n1\n2\n3\n4\n" > /tmp/adb-test/datasets/smoke_01/data.csv

# Run — confirms pipeline works up to Docker
AGENTICDATABENCH_DATA=/tmp/adb-test \
uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \
    -T data_dir=/tmp/adb-test --limit 1 --model mockllm/model
# Expected: "ERROR: Docker sandbox environments require Docker Engine"
# (means the pipeline is working correctly — just needs Docker to proceed)
```

---

## Module structure

```
src/agenticdatabench/
├── task.py          # @task entry point
├── _dataset.py      # HuggingFace dataset loading → inspect_ai Samples
├── _scorer.py       # File-comparison scorer using eval_func strings
├── metrics/         # Ported from DA-Code (MIT); used for host-side scoring
│   ├── table.py     # compare_csv, compare_sqlite
│   ├── myjson.py    # compare_json, compare_json_normalized
│   ├── image.py     # compare_image
│   ├── ml.py        # compare_model
│   ├── text.py      # compare_text
│   └── script/
│       └── ml_script.py  # CalculateML class (ML metric helpers)
└── docker/
    ├── Dockerfile   # Agent execution environment with data science stack
    └── compose.yaml # inspect_ai sandbox configuration
```
