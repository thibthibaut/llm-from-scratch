# /// script
# requires-python = ">=3.10"
# dependencies = ["clearml"]
# ///
"""Stream a training run's `metrics.csv` to a ClearML dashboard, live.

The trainer is Rust and only writes `<artifact_dir>/metrics.csv`; this script runs beside it,
follows that file, and reports each new row as ClearML scalars (loss and perplexity for train and
valid, plus tokens/s). ClearML also records the machine's GPU/CPU usage while the script runs.

Credentials come from the file given by `--creds` (default: `clearml-creds` at the repo root), in
`clearml.conf` format. It replaces `~/clearml.conf` entirely, so another account configured on the
machine is never used.

    uv run scripts/clearml_bridge.py --artifact-dir artifacts --task gpt2-small-run1

Run it in its own tmux pane next to `train`; Ctrl-C stops it. `train` rewrites `metrics.csv` when
it starts or resumes; the bridge notices and replays the file, so curves stay complete.
"""

import argparse
import math
import os
import signal
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifact-dir", default="artifacts", help="the trainer's --artifact-dir")
    p.add_argument("--task", required=True, help="ClearML task (run) name")
    p.add_argument("--project", default="llm-from-scratch", help="ClearML project name")
    p.add_argument("--creds", default=str(REPO_ROOT / "clearml-creds"), help="clearml.conf-style file")
    p.add_argument("--poll", type=float, default=10.0, help="seconds between checks of the CSV")
    p.add_argument("--once", action="store_true", help="report the CSV as it is now, then exit")
    args = p.parse_args()

    creds = Path(args.creds)
    if not creds.is_file():
        raise SystemExit(f"ClearML credentials file not found: {creds}")
    # Must be set before importing clearml, which reads its config at import time.
    os.environ["CLEARML_CONFIG_FILE"] = str(creds.resolve())
    from clearml import Task

    task = Task.init(
        project_name=args.project,
        task_name=args.task,
        reuse_last_task_id=False,
        auto_connect_frameworks=False,
    )
    config = Path(args.artifact_dir) / "config.json"
    if config.is_file():
        task.connect_configuration(config, name="config.json")
    logger = task.get_logger()

    csv = Path(args.artifact_dir) / "metrics.csv"
    offset = 0
    print(f"Following {csv}; dashboard: {task.get_output_log_web_page()}")
    try:
        while True:
            size = csv.stat().st_size if csv.exists() else 0
            if size < offset:  # train rewrote the file (new run or resume): replay it
                offset = 0
            if size > offset:
                with csv.open() as f:
                    f.seek(offset)
                    chunk = f.read()
                # Only consume complete lines; a row being written stays for the next poll.
                complete = chunk[: chunk.rfind("\n") + 1]
                offset += len(complete.encode())
                for line in complete.splitlines():
                    report(logger, line)
                logger.flush()
            if args.once:
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    # A tmux Ctrl-C reaches python twice (directly and forwarded by `uv run`); don't let the
    # second one abort the close, which marks the task completed and flushes the last scalars.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    task.close()


def report(logger, line: str) -> None:
    cols = line.split(",")
    if len(cols) != 6 or cols[0] == "step":
        return
    step, _epoch, _tokens, split, loss, tps = cols
    step, loss = int(step), float(loss)
    logger.report_scalar("loss", split, loss, iteration=step)
    logger.report_scalar("perplexity", split, math.exp(min(loss, 50.0)), iteration=step)
    if split == "train":
        logger.report_scalar("tokens_per_sec", "train", float(tps), iteration=step)


if __name__ == "__main__":
    main()
