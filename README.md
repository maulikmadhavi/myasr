# my_asr

Weekly maintenance pipeline for a single-speaker lapel-mic Whisper model.

- Dataset: https://huggingface.co/datasets/MaulikMadhavi/lapel-mic-speech-asr-dataset
- Model: https://huggingface.co/MaulikMadhavi/lapel_mic_whisper_base_finetuned

## Weekly cycle

| step | command | what it does |
|---|---|---|
| 1. Record | `pixi run ui` | Browser UI: shows each sentence of the newest `prompts/sessionNNNN.txt`; Record, Stop, play back, Submit. Each take is also transcribed by the deployed model. |
| 2. Release | `pixi run release` | merge -> push data -> train -> gate -> push model -> deploy |
| 3. Feedback | `/asr-feedback` in Claude Code | Analyses the errors and writes the next 200-sentence prompt file. |
| Use | `pixi run dictate` | Hold F9 and speak; the text is typed into the focused window (e.g. Claude Code). |

`pixi run pipeline` runs the same stages as `release` without the two pushes, so nothing leaves this machine.

## Pipeline stages (`asr/pipeline.py`)

1. **merge** (`asr/merge.py`): pulls the published dataset into `dataset/`, adds each folder in `recordings/` as a
   session parquet, validates every clip (unique ids, 16 kHz mono, 0.5-30 s, non-empty text) and regenerates the dataset README.
2. **push data** (`asr/publish.py`, release only): uploads `dataset/` to the dataset repo.
3. **train** (`asr/train.py`): fine-tunes `openai/whisper-base` on all sessions into `models/vYYMMDD`. About 10% of clips,
   fixed by clip id, are held out; the report goes to `reports/train_vYYMMDD.json`.
4. **gate**: the new model must be at least as good as the published one on the held-out clips, otherwise the pipeline
   stops here (`--force` overrides).
5. **push model** (release only): uploads the model, tokenizer and model card, and tags the commit `vYYMMDD`.
6. **deploy**: points `models/current.txt` at the new version, which `dictate` and the UI load.

Settings (repo ids, version format) are at the top of `asr/common.py`.

## Layout

```
asr/          pipeline code
prompts/      sentences to record, one file per session        (git)
reports/      training reports and feedback                    (git)
recordings/   wav files + metadata.jsonl written by the UI     (ignored)
dataset/      local mirror of the Hub dataset                  (ignored)
models/       one folder per version + current.txt             (ignored)
```

## Setup

```
pixi install
pixi run hf auth login     # needs write access to both repos
```
