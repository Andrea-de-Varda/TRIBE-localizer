"""Build data/localizers/localizers.csv.gz: the four text localizers (two physics variants).

Language and ToM stimuli are downloaded from their public sources and checked against pinned SHA-256s
(cached in data/external/localizers/); MD and physics stimuli are generated from seeds.
"""
import hashlib
import urllib.request

import pandas as pd

from tribeloc import ROOT, load_config
from tribeloc.localizers import build_table, language_rows, md_rows, physics_rows, tom_rows


def fetch(name, spec):
    path = ROOT / "data" / "external" / "localizers" / f"{name}.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0"})
        path.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == spec["sha256"], f"{name}: sha256 {digest} != {spec['sha256']}"
    return data


def main():
    cfg = load_config()
    lc, s = cfg["localizers"], cfg["stimuli"]
    rows = pd.concat([language_rows(fetch("language", lc["sources"]["language"])),
                      md_rows(lc["md"]["n"], lc["md"]["seed"]),
                      tom_rows(fetch("tom", lc["sources"]["tom"])),
                      physics_rows(lc["physics"]["n_towers"], lc["physics"]["seed"])], ignore_index=True)
    table = build_table(rows, s["word_seconds"], s["onset_seconds"], lc["half_seed"])
    assert table.stim_id.is_unique
    assert (table.end_s.max() + cfg["windows"]["full_tail"]["end"]) <= s["timeline_seconds"]
    out = ROOT / lc["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    summary = table.groupby(["localizer", "condition"], sort=False).agg(
        n=("stim_id", "size"), mean_words=("n_words", "mean"), min_words=("n_words", "min"), max_words=("n_words", "max"),
        unique_texts=("text", "nunique")).reset_index()
    summary.to_csv(out.parent / "summary.csv", index=False)
    print(summary.round(1).to_string(index=False), flush=True)
    for loc in table.localizer.unique():
        print(f"\n[{loc}]", *table[table.localizer == loc].groupby("condition", sort=False).text.first().tolist(), sep="\n  ")


if __name__ == "__main__":
    main()
