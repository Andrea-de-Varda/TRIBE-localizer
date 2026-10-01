"""Score every text with BrainGPT (GPU, Slurm compute node only). Writes results/scores.csv."""
import os
import time

import pandas as pd

from bglib import HERE, load_config, score_text


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run on a Slurm compute node, not a login node")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = load_config()
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["base"])
    model = AutoModelForCausalLM.from_pretrained(m["base"], torch_dtype=getattr(torch, m["dtype"]), device_map="auto")
    model = PeftModel.from_pretrained(model, m["adapter"]).eval()
    texts = pd.read_csv(HERE / "data" / "abstracts.csv")
    rows, start = [], time.time()
    for i, r in enumerate(texts.itertuples()):
        rows.append(score_text(model, tok, r.context, r.result))
        if i % 100 == 0:
            print(f"{i}/{len(texts)} scored ({time.time() - start:.0f} s)", flush=True)
    out = pd.concat([texts.drop(columns=["context", "result", "text"]), pd.DataFrame(rows)], axis=1)
    out.to_csv(HERE / "results" / "scores.csv", index=False)
    print(f"saved {len(out)} scores ({time.time() - start:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
