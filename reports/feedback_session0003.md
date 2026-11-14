# Feedback before session0003 (2026-11-14)

## Scores on the 35 held-out clips

| model | WER |
|---|---|
| stock `openai/whisper-base` | 7.4% |
| published `lapel_mic_whisper_base_finetuned` (May 2025) | 7.2% |
| retrained today on session0001 + session0002 | 5.2% |

The hold-out is only about 360 words, so one word is roughly 0.3 points. The gain is about seven fewer wrong words:
encouraging, not yet conclusive. Session0003 scores from `pixi run record` will be a larger and cleaner test.
The model starts overfitting after epoch 3 (training loss near zero), which is expected with 31 minutes of audio.

## Error patterns in the retrained model

| pattern | reference -> model |
|---|---|
| "sh" / "s" and vowel confusions | shell -> sale, cells -> sells, bagel -> bugle, Blossoms -> Blosoms |
| Word onsets misheard | Can I -> Then I, voice -> world's, Germany -> The money, grocery store -> gloss restore |
| Articles added, dropped or swapped | at platform -> at the platform, a ball -> the ball, after the new -> after new, of the old oak -> of old oak |
| Word endings | funds -> fund, affecting -> affected |
| Proper names | Christian -> Kristen |
| Scoring artefact, not a real error | Here's -> Here is (the hyphen in "Military-grade" is already ignored by the scorer) |

## What session0003 targets

- Lines 1-80: the patterns above in new contexts ("sh"/"s" contrasts, sentence-initial "Can"/"Then", "voice"/"world's",
  singular/plural and -ing/-ed pairs, sentences where the article choice matters, the misheard names and places).
- Lines 81-140: sentences you would dictate to a coding assistant (git, Python, training and evaluation vocabulary).
- Lines 141-200: general coverage (numbers, times, dates, names, places, questions).

No earlier feedback report exists, so there is no week-over-week comparison yet.
