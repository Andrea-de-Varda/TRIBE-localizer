"""No-input baseline (GPU, Slurm compute node only): TRIBE's prediction with all text features set to zero.

Writes results/analysis/baseline/no_input.npz with the time course (vertices x time) and output times.
Check: two dummy timelines with different words must give identical zero-feature predictions, confirming that
the baseline does not depend on the stimulus.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.baseline import predict_no_input
from tribeloc.inference import load_model


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run on a Slurm compute node, not a login node")
    cfg = load_config()
    s, t = cfg["stimuli"], cfg["tribe"]
    rows = pd.DataFrame([dict(stim_id="noinput_a", text="the", onset_s=s["onset_seconds"]),
                         dict(stim_id="noinput_b", text="Where will it fall? A tower stands.", onset_s=s["onset_seconds"])])
    cache = Path(os.environ.get("TMPDIR", "/tmp")) / f"tribeloc_baseline_{os.environ['SLURM_JOB_ID']}"
    model = load_model(t["checkpoint"], cache, t["batch_size"], t["text_batch_size"])
    (ya, ta), (yb, tb) = predict_no_input(model, rows, s["word_seconds"], s["timeline_seconds"])
    diff = float(np.abs(ya - yb).max())
    assert np.array_equal(ta, tb) and diff < 1e-5, f"zero-feature predictions differ across timelines: {diff}"
    out = ROOT / cfg["paths"]["analysis"] / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "no_input.npz", timecourse=ya.astype(np.float32), times=ta)
    info = dict(job=os.environ["SLURM_JOB_ID"], max_abs_difference_between_dummy_timelines=diff, shape=list(ya.shape),
                times=[float(ta[0]), float(ta[-1])], mean=float(ya.mean()), sd_over_time=float(ya.std(1).mean()))
    (out / "no_input.json").write_text(json.dumps(info, indent=2))
    print("NO-INPUT BASELINE SAVED", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
