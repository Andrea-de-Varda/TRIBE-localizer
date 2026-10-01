"""Build data/stimuli/stimuli.csv.gz (one row per stimulus) and a per-task summary."""
import subprocess

from tribeloc import ROOT, load_config
from tribeloc.stimuli import build_table, load_tasks, summarize


def main():
    cfg = load_config()
    src = ROOT / cfg["source"]["local_dir"]
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert head == cfg["source"]["commit"], f"LLM_Modularity at {head}, expected {cfg['source']['commit']}; run scripts/00_fetch_source.sh"
    s = cfg["stimuli"]
    tasks = load_tasks(src, cfg["source"]["domains"])
    assert len(tasks) == 46, len(tasks)
    table = build_table(tasks, s["word_seconds"], s["onset_seconds"], s["half_seed"])
    assert table.text.is_unique and table.stim_id.is_unique
    assert table.end_s.max() + cfg["windows"]["full_tail"]["end"] <= s["timeline_seconds"]
    out = ROOT / s["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary = summarize(tasks, table)
    summary.to_csv(out.parent / "task_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\n{table.item.groupby(table.task).nunique().sum()} items, {len(table)} stimuli -> {out}")
    print(table.groupby("domain", sort=False).n_words.mean().round(1).to_string())


if __name__ == "__main__":
    main()
