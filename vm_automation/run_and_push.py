"""VM-side replacement for the GitHub Actions workflow -- run from the VM's
own crontab, not GitHub Actions (measured only ~15% real delivery on this
exact repo/cadence; see hormuz-strait-monitor and ea-financial-tracker for
the same migration, both since running at ~100%).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_FILES = ["data/history.csv", "outputs/signal_report.png", "models/baselines.json"]

# main.py prints e.g. "Processed 13 records; flagged 0 anomalies in this run."
SUMMARY_RE = re.compile(r"Processed (\d+) records; flagged (\d+) anomalies")


def run(*args, check=True):
    return subprocess.run(list(args), cwd=str(REPO_DIR), check=check)


def run_pipeline():
    result = subprocess.run(
        [sys.executable, "main.py"], cwd=str(REPO_DIR),
        check=True, capture_output=True, text=True,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)

    match = SUMMARY_RE.search(result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def sync_with_remote():
    # --hard, not --soft, and BEFORE main.py runs: reset --soft only moves
    # HEAD, leaving stale index entries for any file this script doesn't
    # explicitly `git add` (e.g. source files edited from another machine),
    # which then get silently recommitted on the next force-push. Learned
    # this the hard way on hormuz-strait-monitor.
    run("git", "fetch", "origin", "main")
    run("git", "reset", "--hard", "origin/main")


def git_commit_and_push(records, anomalies):
    # freddynyanda@proton.me is Fred's real, verified GitHub email -- the
    # original workflow committed as "DSN Bot <actions@github.com>", an
    # unverified address, so every commit was real but silently uncredited
    # on his contribution graph.
    run("git", "config", "user.name", "nyandajr")
    run("git", "config", "user.email", "freddynyanda@proton.me")
    run("git", "add", *DATA_FILES, check=False)

    diff = run("git", "diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        print("[run_and_push] no changes to commit")
        return

    if records is not None:
        message = f"chore: DSN update — {records} records, {anomalies} anomalies flagged [skip ci]"
    else:
        message = "chore: DSN update [skip ci]"
    run("git", "commit", "-m", message)
    run("git", "push", "--force", "origin", "HEAD:main")


def main():
    sync_with_remote()
    records, anomalies = run_pipeline()
    git_commit_and_push(records, anomalies)
    print("[run_and_push] done")


if __name__ == "__main__":
    main()
