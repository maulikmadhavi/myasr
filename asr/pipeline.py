"""Run the release pipeline.

    pixi run pipeline            merge -> train -> gate -> deploy   (nothing leaves this machine)
    pixi run release             merge -> push data -> train -> gate -> push model -> deploy
    pixi run release --force     release even if the new model fails the quality gate

Gate: the new model must be at least as good as the published one on the held-out clips.
"""
import argparse
import time

from . import merge, publish, train
from .common import CURRENT, MODELS, version


def stage(name):
    print(f"\n=== {name} ===", flush=True)
    return time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--push", action="store_true", help="also push the dataset and the model to the Hugging Face Hub")
    ap.add_argument("--force", action="store_true", help="release even if the quality gate fails")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    ver = version()
    start = time.time()

    stage("merge: new recordings + published dataset")
    merge_info = merge.run()

    if args.push:
        stage("push data")
        publish.push_data(merge_info)

    stage(f"train {ver}")
    report = train.run(ver, epochs=args.epochs)

    stage("gate")
    scores = report["held_out_wer"]
    passed = scores["new"] <= scores["published"]
    print(f"  new {scores['new']:.1%} vs published {scores['published']:.1%}: {'PASS' if passed else 'FAIL'}")
    if not passed and not args.force:
        raise SystemExit(f"Gate failed. {ver} is kept in models/ but was not pushed or deployed (--force overrides).")

    if args.push:
        stage(f"push model {ver}")
        publish.push_model(report)

    stage("deploy")
    MODELS.mkdir(exist_ok=True)
    CURRENT.write_text(ver, encoding="utf8")
    print(f"  `pixi run dictate` and the recording UI now use models/{ver}")
    print(f"\nDone in {time.time() - start:.0f} s. Report: reports/train_{ver}.json")


if __name__ == "__main__":
    main()
