"""Recording UI: read the sentence, record, play back, submit.

    pixi run ui

Audio is captured by this process straight from the microphone (16 kHz mono,
no browser processing), the same way `dictate` captures it. Each take is also
transcribed by the deployed model, so recording a session doubles as a test.
"""
import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf

from .common import RATE, RECORDINGS, clip_id, load_asr, load_meta, read_prompts, save_meta, session_names, transcribe, wer


class Recorder:
    def __init__(self):
        self.stream, self.chunks, self.take = None, [], None  # take: last stopped recording (float32)

    def start(self, device):
        self.stop()
        self.chunks, self.take = [], None
        self.stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=device, callback=lambda data, *_: self.chunks.append(data.copy()))
        self.stream.start()

    def stop(self):
        if self.stream is None:
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.take = np.concatenate(self.chunks)[:, 0] if self.chunks else None


REC = Recorder()
ASR = None


def microphones():
    """(label, index) for each input of the default host API, default microphone first."""
    default = sd.query_devices(kind="input")
    devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0 and d["hostapi"] == default["hostapi"]]
    devices.sort(key=lambda d: d["index"] != default["index"])
    return [(d["name"], d["index"]) for d in devices]


def next_unrecorded(session, start=0):
    total, meta = len(read_prompts(session)), load_meta(session)
    todo = [i for i in range(total) if clip_id(session, i + 1) not in meta]
    return next((i for i in todo if i >= start), todo[0] if todo else total - 1)


def describe(row):
    if not row or "hyp" not in row:
        return ""
    return f"**Model heard:** {row['hyp']}\n\n**Word error rate:** {row['wer']:.0%}"


def view(session, index, status=""):
    """Everything the page shows for one sentence."""
    prompts, meta = read_prompts(session), load_meta(session)
    index = max(0, min(int(index), len(prompts) - 1))
    cid = clip_id(session, index + 1)
    saved = meta.get(cid)
    header = f"**{session}** · sentence {index + 1} of {len(prompts)} · {len(meta)} recorded"
    if saved:
        header += " · already recorded (submitting a new take replaces it)"
    audio = str(RECORDINGS / session / f"{cid}.wav") if saved else None
    return index, header, f"# {prompts[index]}", audio, describe(saved), status


def start(device):
    REC.start(device)
    return "🔴 Recording ... press Stop when you have finished the sentence."


def stop(session, index, score):
    global ASR
    REC.stop()
    audio = REC.take
    if audio is None or len(audio) < RATE * 0.5:
        REC.take = None
        return None, "", "Nothing recorded (too short). Press Record and try again."
    peak = float(np.abs(audio).max())
    level = "clipping, move the mic away or lower the gain" if peak >= 0.99 else "very quiet, check the mic" if peak < 0.05 else "level OK"
    result = ""
    REC.row = {"duration": round(len(audio) / RATE, 3)}
    if score:
        ASR = ASR or load_asr()
        sentence = read_prompts(session)[index]
        hyp = transcribe(ASR, audio)
        REC.row.update(hyp=hyp, wer=round(wer(sentence, hyp), 3))
        result = describe(REC.row)
    status = f"{len(audio) / RATE:.1f} s, peak {peak:.2f} ({level}). Play it back, then Submit or Record again."
    return (RATE, (audio * 32767).astype(np.int16)), result, status


def submit(session, index):
    if REC.take is None:
        return view(session, index, "Record a take first.")
    cid = clip_id(session, index + 1)
    folder = RECORDINGS / session
    folder.mkdir(parents=True, exist_ok=True)
    sf.write(folder / f"{cid}.wav", REC.take, RATE, subtype="PCM_16")
    meta = load_meta(session)
    meta[cid] = {"id": cid, "sentence": read_prompts(session)[index], "session": session, **REC.row}
    save_meta(session, meta)
    REC.take = None
    done = len(meta) == len(read_prompts(session))
    status = f"Saved {cid}." + (" Session complete: run `pixi run pipeline` or `pixi run release`." if done else "")
    return view(session, next_unrecorded(session, index + 1), status)


def move(session, index, step):
    REC.stop()
    REC.take = None
    return view(session, index + step)


def build():
    sessions = session_names()
    if not sessions:
        raise SystemExit("No prompt files in prompts/. Run /asr-feedback in Claude Code to generate one.")
    mics = microphones()
    with gr.Blocks(title="ASR recording") as demo:
        index = gr.State(0)
        with gr.Row():
            session = gr.Dropdown(sessions, value=sessions[-1], label="Session")
            device = gr.Dropdown(mics, value=mics[0][1], label="Microphone")
            score = gr.Checkbox(True, label="Test the current model on each take")
        header = gr.Markdown()
        sentence = gr.Markdown()
        with gr.Row():
            record_btn = gr.Button("● Record", variant="primary")
            stop_btn = gr.Button("■ Stop")
        status = gr.Markdown()
        playback = gr.Audio(label="Playback", interactive=False)
        result = gr.Markdown()
        with gr.Row():
            prev_btn = gr.Button("← Previous")
            submit_btn = gr.Button("✓ Submit and next", variant="primary")
            skip_btn = gr.Button("Skip →")

        page = [index, header, sentence, playback, result, status]
        first = lambda s: view(s, next_unrecorded(s))
        demo.load(first, session, page)
        session.change(first, session, page)
        record_btn.click(start, device, status)
        stop_btn.click(stop, [session, index, score], [playback, result, status])
        submit_btn.click(submit, [session, index], page)
        prev_btn.click(lambda s, i: move(s, i, -1), [session, index], page)
        skip_btn.click(lambda s, i: move(s, i, 1), [session, index], page)
    return demo


if __name__ == "__main__":
    build().launch(inbrowser=True)
