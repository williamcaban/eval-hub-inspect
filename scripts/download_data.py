#!/usr/bin/env python3
"""
Download benchmark datasets from HuggingFace and structure them for use by
the Docker sandbox.

Default download location:
    ./data/<benchmark>/          (relative to the repo root)

Override via:
    --data-dir /path/to/data     (CLI flag)
    AGENTICDATABENCH_DATA=/path  (env var, read by compose.yaml at eval time)

Airgapped / offline deployments
────────────────────────────────
If the evaluation environment has no internet access, download the data on a
connected machine first, then transfer the directory to the target machine:

  # On a connected machine:
  uv run python scripts/download_data.py --benchmark agenticdatabench \\
      --data-dir /mnt/transfer/agenticdatabench

  # Copy /mnt/transfer/agenticdatabench to the airgapped machine, e.g.:
  rsync -av /mnt/transfer/agenticdatabench airgapped-host:/data/agenticdatabench

  # On the airgapped machine, point the env var at the transferred directory:
  export AGENTICDATABENCH_DATA=/data/agenticdatabench

  # Then run the eval as normal — no network access required:
  uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \\
      -T data_dir=/data/agenticdatabench --model <model>

Manual layout (if you already have the files)
──────────────────────────────────────────────
The eval expects this directory structure under the data root:

    <data_root>/
    ├── tasks/
    │   └── dev.jsonl            ← task definitions
    ├── datasets/
    │   └── <task_id>/           ← one directory per task
    │       └── <data_files>     ← CSV, parquet, shapefiles, …
    └── gold/
        └── <task_id>/           ← one directory per task
            └── result.csv       ← gold-standard output file(s)

Usage:
    uv run python scripts/download_data.py --benchmark agenticdatabench
    uv run python scripts/download_data.py --benchmark agenticdatabench \\
        --data-dir /custom/path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def download_agenticdatabench(data_dir: Path) -> None:
    """Download AgenticDataBench tasks, datasets, and gold files from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface-hub is required. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading AgenticDataBench to {data_dir} …")
    print("Dataset: https://huggingface.co/datasets/shawnzzzh/AgenticDataBench")
    print("Size: ~27.3 GB — this will take a while on a typical connection.")
    print()

    hf_cache = data_dir / "_hf_cache"
    snapshot_download(
        repo_id="shawnzzzh/AgenticDataBench",
        repo_type="dataset",
        local_dir=str(hf_cache),
        ignore_patterns=["*.git*"],
    )

    # Create symlinks from the expected layout to the HuggingFace cache structure
    for subdir in ("datasets", "gold", "tasks"):
        src = hf_cache / "testbed" / subdir
        dst = data_dir / subdir
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
            print(f"  Linked {dst} → {src}")
        elif not src.exists():
            print(f"  Warning: {src} not found in downloaded snapshot", file=sys.stderr)

    print()
    print(f"Done. Data available at {data_dir.resolve()}/")
    print()
    print("Directory layout:")
    print(f"  {data_dir}/tasks/dev.jsonl      ← 246 public task definitions")
    print(f"  {data_dir}/datasets/<task_id>/  ← input data files per task")
    print(f"  {data_dir}/gold/<task_id>/      ← gold reference files per task")
    print()
    print("Next step — run the evaluation:")
    print("  uv run inspect eval src/agenticdatabench/task.py@agenticdatabench \\")
    print(f"      -T data_dir={data_dir.resolve()} --limit 5 --model openai/gpt-4o-mini")


BENCHMARKS: dict[str, object] = {
    "agenticdatabench": download_agenticdatabench,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download benchmark data from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Airgapped usage:
  Download on a connected machine with --data-dir /mnt/transfer/<benchmark>,
  transfer the directory to the target, then set AGENTICDATABENCH_DATA=/path
  before running the evaluation.

Manual layout:
  <data_root>/tasks/dev.jsonl
  <data_root>/datasets/<task_id>/<files>
  <data_root>/gold/<task_id>/result.csv
""",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARKS),
        help="Which benchmark to download",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Download directory (default: ./data/<benchmark>)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path("data") / args.benchmark
    fn = BENCHMARKS[args.benchmark]
    fn(data_dir)  # type: ignore[operator]


if __name__ == "__main__":
    main()
