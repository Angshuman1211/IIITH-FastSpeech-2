"""Poll a Kaggle kernel and stop when it is no longer running.

    python scripts/watch_kaggle.py --language Telugu --timeout-min 15

Intended to be run in the background. The point is the early signal: every
sanity check in the kernel happens in its first minute, so a run that is still
going after fifteen minutes is genuinely training, and one that is not prints
its status here straight away.
"""

import argparse
import time

from launch_kaggle import api, slug, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["English", "Telugu"])
    ap.add_argument("--every-sec", type=int, default=60)
    ap.add_argument("--timeout-min", type=float, default=15.0)
    args = ap.parse_args()

    a = api()
    kslug = slug(args.language)
    started = time.time()
    while True:
        st = status(a, kslug)
        if st.lower().split(".")[-1] not in ("running", "queued"):
            print(f"FINISHED with status {st} after "
                  f"{(time.time() - started) / 60:.1f} min")
            return
        if time.time() - started > args.timeout_min * 60:
            print(f"still running after {args.timeout_min} min, "
                  f"so it is past every sanity check and training")
            return
        time.sleep(args.every_sec)


if __name__ == "__main__":
    main()
