"""Push-to-talk dictation with the deployed model.

Hold the hotkey, speak, release: the transcript is typed into whatever window
has focus (Claude Code's prompt, an editor, a browser, ...).

    pixi run dictate
    pixi run dictate --key f8 --enter

Ctrl+C in this console to quit.
"""
import argparse
import queue
import threading

import keyboard
import numpy as np
import sounddevice as sd

from .common import RATE, load_asr, transcribe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="HF repo id or local folder (default: the deployed version, else the published model)")
    ap.add_argument("--key", default="f9", help="push-to-talk key (default: f9)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--enter", action="store_true", help="press Enter after typing (auto-submit)")
    ap.add_argument("--device", type=int, help="microphone index, see: python -m sounddevice")
    args = ap.parse_args()

    asr = load_asr(args.model)
    chunks, recording, jobs = [], threading.Event(), queue.Queue()

    def on_audio(indata, frames, time, status):
        if recording.is_set():
            chunks.append(indata.copy())

    def press(_):
        if not recording.is_set():  # ignore key auto-repeat while held
            chunks.clear()
            recording.set()
            print("listening ...", end="\r", flush=True)

    def release(_):
        if recording.is_set():
            recording.clear()
            jobs.put(np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32))

    # suppress=True keeps the hotkey itself from reaching the focused app
    keyboard.on_press_key(args.key, press, suppress=True)
    keyboard.on_release_key(args.key, release, suppress=True)

    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=args.device, callback=on_audio):
        print(f"Ready. Hold {args.key.upper()} to dictate. Ctrl+C to quit.")
        while True:
            try:
                audio = jobs.get(timeout=0.2)  # timeout keeps Ctrl+C responsive
            except queue.Empty:
                continue
            if len(audio) < RATE * 0.3:  # accidental tap
                print("too short, ignored")
                continue
            text = transcribe(asr, audio, args.language)
            print(f"> {text}")
            if text:
                keyboard.write(text + " ")
                if args.enter:
                    keyboard.send("enter")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
