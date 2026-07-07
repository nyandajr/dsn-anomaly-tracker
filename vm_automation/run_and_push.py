"""VM-side replacement for the GitHub Actions workflow -- run from the VM's
own crontab, not GitHub Actions (measured only ~15% real delivery on this
exact repo/cadence; see hormuz-strait-monitor and ea-financial-tracker for
the same migration, both since running at ~100%).
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
DATA_FILES = ["data/history.csv", "outputs/signal_report.png", "models/baselines.json"]


def run(*args, check=True):
    return subprocess.run(list(args), cwd=str(REPO_DIR), check=check)


def sync_with_remote():
    # --hard, not --soft, and BEFORE main.py runs: reset --soft only moves
    # HEAD, leaving stale index entries for any file this script doesn't
    # explicitly `git add` (e.g. source files edited from another machine),
    # which then get silently recommitted on the next force-push. Learned
    # this the hard way on hormuz-strait-monitor.
    run("git", "fetch", "origin", "main")
    run("git", "reset", "--hard", "origin/main")


def git_commit_and_push():
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

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run("git", "commit", "-m", f"chore: DSN update {timestamp} [skip ci]")
    run("git", "push", "--force", "origin", "HEAD:main")


def main():
    sync_with_remote()
    run(sys.executable, "main.py")
    git_commit_and_push()
    print("[run_and_push] done")


if __name__ == "__main__":
    main()
