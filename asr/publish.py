"""Stages: push the merged dataset and the new model to the Hugging Face Hub."""
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from .common import BASE_MODEL, DATASET, DATASET_REPO, MODEL_REPO, MODELS

MODEL_CARD = """---
language:
- en
base_model: {base}
datasets:
- {dataset}
pipeline_tag: automatic-speech-recognition
---

# Lapel-mic Whisper base (single speaker)

`{base}` fine-tuned on [{dataset}](https://huggingface.co/datasets/{dataset}).
Retrained on a regular cycle; every release is tagged with its training date (`vYYMMDD`).

## Current release: {version}

Trained on {train} clips from {sessions} sessions. Word error rate on {held} held-out clips:

| model | WER |
|---|---|
| `{base}` | {wer_base:.1%} |
| previous release | {wer_prev:.1%} |
| **{version}** | **{wer_new:.1%}** |

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="{repo}", revision="{version}")
```
"""


def push_data(merge_info):
    api = HfApi()
    message = f"Add {', '.join(merge_info['merged'])}" if merge_info["merged"] else "Sync dataset"
    api.upload_folder(repo_id=DATASET_REPO, repo_type="dataset", folder_path=DATASET, commit_message=message, ignore_patterns=[".cache/**"])
    print(f"  https://huggingface.co/datasets/{DATASET_REPO}")


def push_model(report):
    version, scores = report["version"], report["held_out_wer"]
    folder = MODELS / version
    card = MODEL_CARD.format(
        base=BASE_MODEL,
        dataset=DATASET_REPO,
        repo=MODEL_REPO,
        version=version,
        train=report["clips"]["train"],
        held=report["clips"]["held_out"],
        sessions=len(report["sessions"]),
        wer_base=scores["whisper-base"],
        wer_prev=scores["published"],
        wer_new=scores["new"],
    )
    (folder / "README.md").write_text(card, encoding="utf8", newline="\n")
    api = HfApi()
    api.upload_folder(repo_id=MODEL_REPO, folder_path=folder, commit_message=f"{version}: held-out WER {scores['new']:.1%}")
    try:  # a second release on the same day moves the tag to the newer commit
        api.delete_tag(MODEL_REPO, tag=version)
    except HfHubHTTPError:
        pass
    api.create_tag(MODEL_REPO, tag=version, tag_message=f"Held-out WER {scores['new']:.1%}")
    print(f"  https://huggingface.co/{MODEL_REPO}/tree/{version}")
