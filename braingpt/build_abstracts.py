"""Build data/abstracts.csv: every abstract x candidate x results paraphrase to score (CPU, local)."""
import pandas as pd

from bglib import HERE, ROOT, build_texts, load_config


def main():
    cfg = load_config()
    tasks = pd.read_csv(HERE / "tasks.tsv", sep="\t")
    stimuli = pd.read_csv(ROOT / "data" / "stimuli" / "stimuli.csv.gz")
    assert set(tasks.task) == set(stimuli.task) and tasks.task.is_unique, set(tasks.task) ^ set(stimuli.task)
    texts = build_texts(cfg, tasks, stimuli)
    texts.to_csv(HERE / "data" / "abstracts.csv", index=False)
    first = texts.drop_duplicates("task")
    (HERE / "data" / "abstracts_preview.txt").write_text("\n\n".join(
        f"[{r.domain}] {r.task}\n{r.context[len(cfg['prefix']):].strip()}" for r in first.itertuples()) + "\n")
    print(f"{len(texts)} texts, {texts.task.nunique()} tasks", flush=True)
    one = texts[(texts.task == "physics_temperature") & (texts["style"] == "anatomical") & (texts.paraphrase == 0)]
    print("\nexample context:\n" + one.context.iloc[0], *one.result, sep="\n  ")


if __name__ == "__main__":
    main()
