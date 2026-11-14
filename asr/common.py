"""Configuration and helpers shared by every pipeline stage."""
import hashlib
import io
import json
import re
from datetime import date
from pathlib import Path

import soundfile as sf

DATASET_REPO = "MaulikMadhavi/lapel-mic-speech-asr-dataset"
MODEL_REPO = "MaulikMadhavi/lapel_mic_whisper_base_finetuned"
BASE_MODEL = "openai/whisper-base"
RATE = 16000
VERSION_FORMAT = "%y%m%d"  # model version = "v" + training date, e.g. v261114

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"  # sessionNNNN.txt, one sentence per line (in git)
RECORDINGS = ROOT / "recordings"  # sessionNNNN/*.wav + metadata.jsonl, written by the UI
DATASET = ROOT / "dataset"  # local mirror of the Hub dataset repo, written by merge
MODELS = ROOT / "models"  # one folder per version + current.txt
REPORTS = ROOT / "reports"  # training and feedback reports (in git)
CURRENT = MODELS / "current.txt"  # name of the deployed version


def version():
    return "v" + date.today().strftime(VERSION_FORMAT)


def current_model():
    """Deployed local model if there is one, otherwise the published one."""
    if CURRENT.exists():
        folder = MODELS / CURRENT.read_text(encoding="utf8").strip()
        if (folder / "config.json").exists():
            return str(folder)
    return MODEL_REPO


def session_names():
    return [p.stem for p in sorted(PROMPTS.glob("session*.txt"))]


def read_prompts(session):
    text = (PROMPTS / f"{session}.txt").read_text(encoding="utf-8-sig")
    return [line.strip() for line in text.splitlines() if line.strip()]


def clip_id(session, number):
    """number is the 1-based prompt number."""
    return f"{session}_{number:04d}"


def load_meta(session):
    path = RECORDINGS / session / "metadata.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf8").splitlines())}


def save_meta(session, meta):
    path = RECORDINGS / session / "metadata.jsonl"
    path.write_text("".join(json.dumps(meta[k]) + "\n" for k in sorted(meta)), encoding="utf8")


def is_eval(clip):
    """Fixed ~10% hold-out, keyed on the clip id so it never changes as sessions are added."""
    return int(hashlib.md5(clip.encode()).hexdigest(), 16) % 10 == 0


def normalize(text):
    """Lowercase and drop punctuation so WER only counts word errors."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def wer(ref, hyp):
    import jiwer

    return jiwer.wer(normalize(ref), normalize(hyp))


def dataset_parquets():
    return sorted(DATASET.glob("data/session*/*.parquet"))


def load_rows():
    """Every clip in the local dataset mirror, with a decoded float32 `audio` array."""
    import pyarrow.parquet as pq

    rows = []
    for path in dataset_parquets():
        for r in pq.read_table(path).to_pylist():
            audio, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            rows.append({**r, "audio": audio, "rate": rate})
    return rows


def load_asr(model=None):
    import torch
    from transformers import AutoTokenizer, pipeline

    model = model or current_model()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception:  # the model repo was first published without tokenizer files
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    cuda = torch.cuda.is_available()
    print(f"Loading {model} on {'GPU' if cuda else 'CPU'} ...")
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        device=0 if cuda else -1,
        dtype=torch.float16 if cuda else torch.float32,
        chunk_length_s=30,  # lets recordings longer than 30 s work
        ignore_warning=True,
    )


def transcribe(asr, audio, language="en"):
    out = asr({"raw": audio, "sampling_rate": RATE}, generate_kwargs={"language": language, "task": "transcribe"})
    return out["text"].strip()
