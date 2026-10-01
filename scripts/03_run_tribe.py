"""Run TRIBE v2 on stimulus shards (GPU, Slurm compute node only).

    python scripts/03_run_tribe.py --list                     # number of shards
    python scripts/03_run_tribe.py --shards 0                 # benchmark one shard
    python scripts/03_run_tribe.py --array-index 3 --per-task 4   # shards 12-15 (Slurm array)
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.inference import assign_shards, run_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*")
    ap.add_argument("--array-index", type=int)
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "tribe" / "shards"))
    a = ap.parse_args()
    cfg = load_config()
    table = pd.read_csv(ROOT / cfg["stimuli"]["output"])
    n = int(assign_shards(table, cfg["tribe"]["shard_size"]).max()) + 1
    if a.list:
        print(n)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run on a Slurm compute node, not a login node")
    shards = a.shards if a.shards is not None else range(a.array_index * a.per_task, min(n, (a.array_index + 1) * a.per_task))
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"tribeloc_{os.environ['SLURM_JOB_ID']}"
    print(f"{n} shards in total; this job: {list(shards)}", flush=True)
    log = run_shards(table, list(shards), cfg, a.out, cache)
    logdir = Path(a.out).parent / "timing"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / f"job_{os.environ['SLURM_JOB_ID']}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
