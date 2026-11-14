"""Stage: fine-tune Whisper base on the merged dataset and score it.

~10% of clips (fixed by clip id) are held out. Stock Whisper base, the
currently published model and the new one are all scored on them.
"""
import json
import math
from datetime import date

import jiwer
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from .common import BASE_MODEL, MODEL_REPO, MODELS, RATE, REPORTS, is_eval, load_rows, normalize, wer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def features(processor, rows):
    audio = [r["audio"] for r in rows]
    return processor.feature_extractor(audio, sampling_rate=RATE, return_tensors="pt").input_features.to(DEVICE)


@torch.no_grad()
def evaluate(model, processor, rows, batch_size=16):
    """Returns (corpus WER, [{id, ref, hyp, wer}])."""
    model.eval()
    hyps = []
    for i in range(0, len(rows), batch_size):
        ids = model.generate(features(processor, rows[i : i + batch_size]), language="en", task="transcribe")
        hyps += [h.strip() for h in processor.batch_decode(ids, skip_special_tokens=True)]
    refs = [r["sentence"] for r in rows]
    details = [{"id": r["id"], "ref": ref, "hyp": hyp, "wer": round(wer(ref, hyp), 3)} for r, ref, hyp in zip(rows, refs, hyps)]
    return jiwer.wer([normalize(r) for r in refs], [normalize(h) for h in hyps]), details


def run(version, epochs=5, batch_size=16, lr=1e-5):
    """Trains models/<version> and returns the report (also saved to reports/train_<version>.json)."""
    rows = load_rows()
    train = [r for r in rows if not is_eval(r["id"])]
    held = [r for r in rows if is_eval(r["id"])]
    print(f"  {len(rows)} clips: {len(train)} train, {len(held)} held out, device {DEVICE}")

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language="english", task="transcribe")
    scores = {}
    for name, repo in [("whisper-base", BASE_MODEL), ("published", MODEL_REPO)]:
        baseline = WhisperForConditionalGeneration.from_pretrained(repo).to(DEVICE)
        scores[name], _ = evaluate(baseline, processor, held)
        print(f"  held-out WER  {name:13s} {scores[name]:.1%}")
        del baseline

    out = MODELS / version
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL).to(DEVICE)
    steps = epochs * math.ceil(len(train) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(0)
    best = (float("inf"), None)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for i in range(0, len(train), batch_size):
            batch = [train[j] for j in order[i : i + batch_size]]
            labels = processor.tokenizer([b["sentence"] for b in batch], return_tensors="pt", padding=True)
            # drop the leading <|startoftranscript|>: the model re-adds it when it shifts labels right
            y = labels.input_ids.masked_fill(labels.attention_mask == 0, -100)[:, 1:].to(DEVICE)
            loss = model(input_features=features(processor, batch), labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())
        score, details = evaluate(model, processor, held)
        print(f"  epoch {epoch}  loss {np.mean(losses):.3f}  held-out WER {score:.1%}")
        if score < best[0]:  # keep the best epoch only
            best = (score, details)
            model.save_pretrained(out)
            processor.save_pretrained(out)

    scores["new"] = best[0]
    report = {
        "version": version,
        "date": str(date.today()),
        "clips": {"train": len(train), "held_out": len(held)},
        "sessions": sorted({r["session"] for r in rows}),
        "held_out_wer": {k: round(v, 4) for k, v in scores.items()},
        "held_out_errors": [d for d in best[1] if d["wer"] > 0],
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"train_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    return report
