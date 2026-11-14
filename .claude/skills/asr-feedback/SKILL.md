---
name: asr-feedback
description: Weekly ASR maintenance step. Analyse the lapel-mic Whisper model's latest errors, write a feedback report, and generate the next 200-sentence recording script. Use when the user asks for ASR feedback, new prompts/sentences to record, or the next session.
---

# ASR feedback and next recording script

This repo retrains a single-speaker Whisper base model on a weekly cycle (see README.md):
`pixi run ui` (record) -> `pixi run release` (merge, train, gate, push, deploy) -> `/asr-feedback` (this skill).

## Steps

1. Read the evidence, newest first:
   - `reports/train_v*.json` (newest): `held_out_wer` and `held_out_errors` (`ref` vs `hyp`).
   - `recordings/session*/metadata.jsonl` (newest session): rows where `wer > 0`. These sentences were unseen by the model that transcribed them, so they are the most honest signal.
   - Earlier `reports/feedback_*.md`, to see whether last week's weak spots improved.
2. Read every `prompts/session*.txt` so new sentences do not repeat old ones. The next session number is the highest existing one plus one.
3. Write `reports/feedback_sessionNNNN.md` (NNNN = the new session), short and concrete:
   - WER trend (stock Whisper base, published, new; session WER if there is one).
   - Error patterns grouped by cause, each with the actual `ref -> hyp` pairs: confusable sounds, dropped or inserted function words, word endings (plural, tense), proper names, numbers, compounds and hyphenation, rare or technical words.
   - Separate real recognition errors from scoring artefacts (e.g. "6" vs "six", spelling variants) and say so.
   - What improved or regressed since the previous feedback report.
   - What the new script targets and why.
4. Write `prompts/sessionNNNN.txt`: exactly 200 sentences, one per line, UTF-8, no numbering, no blank lines.
   - About 40% target the error patterns found: put the confused words and sounds in fresh contexts, and include minimal-pair style contrasts (e.g. "cells" / "sells") in different sentences.
   - About 30% are things the speaker would dictate to a coding assistant: software, ML and command-line vocabulary written the way it should be transcribed.
   - About 30% are general coverage: numbers, dates, times, names, places, questions, varied sentence openings.
   - 8 to 18 words each, natural to read aloud in one breath, ordinary punctuation, no abbreviations or symbols that are ambiguous to read.
   - No sentence may duplicate an existing prompt, and avoid near-duplicates within the file.
5. Verify the file: 200 non-empty lines, no line shared with earlier prompt files.
6. Tell the user the three headline findings and the next commands: `pixi run ui`, then `pixi run release`.

Do not run `pixi run release` yourself: it publishes to the user's public Hugging Face repos.
