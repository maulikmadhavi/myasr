"""Stage: merge newly recorded sessions into the local mirror of the dataset.

Pulls the published dataset into dataset/, writes one parquet per recorded
session next to the old ones, validates everything and regenerates README.md.
"""
import json

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from .common import DATASET, DATASET_REPO, RATE, RECORDINGS, load_meta, load_rows, normalize, read_prompts

SCHEMA = pa.schema(
    [
        ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("sentence", pa.string()),
        ("id", pa.string()),
        ("session", pa.string()),
        ("duration", pa.float32()),
    ],
    metadata={
        "huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "audio": {"sampling_rate": RATE, "_type": "Audio"},
                        "sentence": {"dtype": "string", "_type": "Value"},
                        "id": {"dtype": "string", "_type": "Value"},
                        "session": {"dtype": "string", "_type": "Value"},
                        "duration": {"dtype": "float32", "_type": "Value"},
                    }
                }
            }
        )
    },
)

README = """---
language:
- en
task_categories:
- automatic-speech-recognition
configs:
- config_name: all
  default: true
  data_files:
  - split: train
    path: data/session*/*.parquet
{configs}
---

# Lapel-mic speech ASR dataset

Self-recorded read speech from a single speaker on a lapel microphone, for fine-tuning and evaluating ASR models.
Each recording session is a separate config; `all` (the default) combines them. All audio is 16 kHz mono.
There is no train/test split yet: every config has a single `train` split.

| config | clips | minutes |
|---|---|---|
{table}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")                 # all sessions
s2 = load_dataset("{repo}", "session0002", split="train")  # one session
s2 = ds.filter(lambda x: x["session"] == "session0002")    # same, from the combined set
```

## Fields

| field | description |
|---|---|
| `audio` | 16 kHz mono, 16-bit PCM WAV |
| `sentence` | prompt read aloud |
| `id` | `<session>_<prompt number>`, e.g. `session0002_0001` |
| `session` | recording session, e.g. `session0002` |
| `duration` | seconds |
"""


def build_readme(stats):
    """stats: {session: (clips, seconds)}"""
    names = sorted(stats)
    configs = "\n".join(f"- config_name: {s}\n  data_files:\n  - split: train\n    path: data/{s}/*.parquet" for s in names)
    rows = [f"| `{s}` | {stats[s][0]} | {stats[s][1] / 60:.1f} |" for s in names]
    rows.append(f"| `all` | {sum(c for c, _ in stats.values())} | {sum(d for _, d in stats.values()) / 60:.1f} |")
    return README.format(configs=configs, table="\n".join(rows), repo=DATASET_REPO)


def write_session(session):
    """recordings/<session>/ -> dataset/data/<session>/train-00000-of-00001.parquet"""
    meta = load_meta(session)
    folder = RECORDINGS / session
    rows = [
        {
            "audio": {"bytes": (folder / f"{cid}.wav").read_bytes(), "path": f"{cid}.wav"},
            "sentence": r["sentence"],
            "id": cid,
            "session": session,
            "duration": r["duration"],
        }
        for cid, r in sorted(meta.items())
    ]
    out = DATASET / "data" / session / "train-00000-of-00001.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), out)
    return len(rows)


def validate(rows):
    """Returns (errors, warnings). Errors stop the pipeline."""
    errors, warnings, seen_ids, seen_text = [], [], set(), {}
    for r in rows:
        cid = r["id"]
        if cid in seen_ids:
            errors.append(f"{cid}: duplicate id")
        seen_ids.add(cid)
        if not cid.startswith(r["session"] + "_"):
            errors.append(f"{cid}: id does not match session {r['session']}")
        if r["rate"] != RATE or r["audio"].ndim != 1:
            errors.append(f"{cid}: audio must be {RATE} Hz mono")
        seconds = len(r["audio"]) / r["rate"]
        if not 0.5 <= seconds <= 30:
            errors.append(f"{cid}: {seconds:.1f} s is outside 0.5-30 s (Whisper reads at most 30 s)")
        if abs(seconds - r["duration"]) > 0.05:
            errors.append(f"{cid}: duration field {r['duration']:.2f} does not match audio {seconds:.2f}")
        if not r["sentence"].strip():
            errors.append(f"{cid}: empty sentence")
        key = normalize(r["sentence"])
        if key in seen_text:
            warnings.append(f"{cid}: same sentence as {seen_text[key]}")
        seen_text.setdefault(key, cid)
    return errors, warnings


def run():
    snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=DATASET)
    published = {p.parent.name for p in DATASET.glob("data/session*/*.parquet")}
    merged = []
    for folder in sorted(RECORDINGS.glob("session*")):
        session = folder.name
        if not load_meta(session):
            continue
        clips = write_session(session)
        total = len(read_prompts(session))
        note = "" if clips == total else f" (partial: {clips} of {total} prompts recorded)"
        print(f"  merged {session}: {clips} clips, {'replaces published copy' if session in published else 'new'}{note}")
        merged.append(session)
    if not merged:
        print("  no local recordings to merge; using the published sessions only")

    rows = load_rows()
    errors, warnings = validate(rows)
    for w in warnings:
        print(f"  warning: {w}")
    if errors:
        raise SystemExit("Dataset validation failed:\n  " + "\n  ".join(errors))

    stats = {}
    for r in rows:
        clips, seconds = stats.get(r["session"], (0, 0.0))
        stats[r["session"]] = (clips + 1, seconds + r["duration"])
    (DATASET / "README.md").write_text(build_readme(stats), encoding="utf8", newline="\n")
    minutes = sum(s for _, s in stats.values()) / 60
    print(f"  dataset: {len(stats)} sessions, {len(rows)} clips, {minutes:.1f} min, validation passed")
    return {"sessions": sorted(stats), "clips": len(rows), "minutes": round(minutes, 1), "merged": merged}
