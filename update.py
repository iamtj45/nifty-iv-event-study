"""Bring the study up to the last completed trading day.

    python update.py                 fetch new days, rebuild, re-backtest
    python update.py --dry-run       say what it would do, change nothing
    python update.py --download-only stop after fetching
    python update.py --since 2026-09-01   re-fetch from a date, to fill a gap

What this exists for. Everything else in the project is a one-shot script
pointed at a date range, which makes the result a snapshot: run build.py once
and the site says October 2026 forever. This is the one command that moves it
forward, and it is deliberately the only place that knows the order the other
scripts have to run in.

    download.py   new bhavcopies, and the Nifty / Bank Nifty / VIX closes
    build.py      per index: daily IV, the event study, placebo, summary
    backtest.py   per index: the straddle rule over events and ordinary days

Two things it will not do.

It will not invent events. data/events.csv is written by hand, because no file
NSE publishes says "the Budget was today". When RBI announces its 2027 meeting
dates, someone has to add them -- and this script says so when it notices
events that are now in the past but were never in the file.

It will not update the deployed site. data/raw/ is gitignored: 86 MB of NSE
files that do not belong in a repository. So the live site cannot refresh
itself. The flow is local update, commit the result CSVs, push, and the host
redeploys. That is a feature rather than a limitation -- it means the public
page only ever serves numbers that were computed here, by code with tests,
before anyone loaded it.

On timing. NSE publishes a day's bhavcopy after the close, around 18:00 IST.
Run this before then and today's file does not exist yet, which is not an
error and is why "up to date" normally means "through yesterday".
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
RAW = DATA / "raw"

# NSE's end-of-day files appear a little after the close. Asking earlier is
# harmless -- the server simply has no file -- but it is worth saying so
# rather than letting it look like a missing trading day.
PUBLISH_HOUR_IST = 18
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

INDEXES = ["NIFTY", "BANKNIFTY"]

# Where each index's summary lives, for the before/after comparison.
SUMMARY = {"NIFTY": DATA / "summary.csv",
           "BANKNIFTY": DATA / "banknifty" / "summary.csv"}


def latest_downloaded():
    """The most recent bhavcopy on disk, by filename.

    Read from data/raw rather than daily_iv.csv on purpose: raw is the input,
    and a day can be downloaded but produce no usable IV -- a holiday-thin
    session, say. Starting from the results would re-download those days
    every single run.
    """
    files = sorted(RAW.glob("*.csv.zip"))
    if not files:
        return None
    return dt.datetime.strptime(files[-1].stem[:8], "%Y%m%d").date()


def first_downloaded():
    files = sorted(RAW.glob("*.csv.zip"))
    if not files:
        return None
    return dt.datetime.strptime(files[0].stem[:8], "%Y%m%d").date()


def read_summaries():
    """Each index's summary table, keyed by index, or None where absent."""
    out = {}
    for name, path in SUMMARY.items():
        out[name] = pd.read_csv(path) if path.exists() else None
    return out


def describe_changes(before, after):
    """Say what moved, in the numbers anyone would actually quote.

    A rebuild that changes nothing should say so plainly. A rebuild that
    shifts an average is the interesting case -- usually because a new event
    landed -- and burying that in a wall of 'done' output is how a surprise
    goes unnoticed.
    """
    lines = []
    for name in INDEXES:
        old, new = before.get(name), after.get(name)
        if new is None:
            continue
        if old is None:
            lines.append(f"  {name}: built for the first time")
            continue

        old_i = old.set_index("type")
        new_i = new.set_index("type")
        for kind in new_i.index:
            if kind not in old_i.index:
                lines.append(f"  {name} {kind}: new event type")
                continue
            o, nw = old_i.loc[kind], new_i.loc[kind]
            if int(o["n"]) != int(nw["n"]):
                lines.append(
                    f"  {name} {kind}: {int(o['n'])} -> {int(nw['n'])} events, "
                    f"average IV change {o['avg_change_pct']:.2f}% -> "
                    f"{nw['avg_change_pct']:.2f}%")
            elif abs(float(o["avg_change_pct"]) - float(nw["avg_change_pct"])) > 0.005:
                lines.append(
                    f"  {name} {kind}: average IV change "
                    f"{o['avg_change_pct']:.2f}% -> {nw['avg_change_pct']:.2f}% "
                    f"(same {int(nw['n'])} events — a revised day)")
    return lines


def check_events(latest):
    """Warn about events that have happened but are not in events.csv.

    Nothing can detect a missing event automatically. What it can detect is
    the opposite: that the file's last entry is behind the data, which almost
    always means dates were announced and never added.
    """
    path = DATA / "events.csv"
    if not path.exists():
        return ["  events.csv is missing"]
    events = pd.read_csv(path, parse_dates=["date"])
    last_event = events["date"].max().date()
    if last_event < latest:
        gap = (latest - last_event).days
        if gap > 120:
            return [f"  events.csv ends {last_event}, {gap} days before the "
                    f"data. New RBI dates or a Budget may be missing — only "
                    f"you can add those."]
    return []


def run(cmd, label):
    """One step, with its output shown as it happens."""
    print(f"\n--- {label} ---", flush=True)
    done = subprocess.run([sys.executable] + cmd, cwd=BASE)
    if done.returncode != 0:
        sys.exit(f"\n{label} failed (exit {done.returncode}). "
                 f"Nothing after this point has run.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, change nothing")
    ap.add_argument("--download-only", action="store_true",
                    help="fetch new days but skip the rebuild")
    ap.add_argument("--since", metavar="YYYY-MM-DD",
                    help="re-fetch from this date instead of the day after "
                         "the newest file, to fill a gap")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild even if no new trading day was found")
    ap.add_argument("--fast", action="store_true",
                    help="solve only the new days instead of re-solving "
                         "every file. Produces byte-identical output; the "
                         "weekly job in CI proves it.")
    ap.add_argument("--full", action="store_true",
                    help="force a complete rebuild from every raw file, "
                         "which is what the weekly check compares against")
    args = ap.parse_args()

    latest = latest_downloaded()
    if latest is None:
        sys.exit("data/raw is empty -- this updates an existing study.\n"
                 "For a first run: python download.py 2022-12-01")

    today = dt.date.today()
    now_ist = dt.datetime.now(IST)
    start = dt.date.fromisoformat(args.since) if args.since else latest + dt.timedelta(days=1)

    print(f"Latest bhavcopy on disk   {latest:%Y-%m-%d} ({latest:%A})")
    print(f"Today                     {today:%Y-%m-%d} ({today:%A})")

    if start > today:
        print("\n  Already current — nothing newer exists yet.")
        if not args.rebuild:
            return
    else:
        days = (today - start).days + 1
        print(f"Will probe                {start:%Y-%m-%d} to {today:%Y-%m-%d} "
              f"({days} calendar day{'s' if days != 1 else ''})")

    # Every calendar day is probed, not just weekdays: NSE opens on a Saturday
    # or Sunday when the Budget falls on one, and this study has six such
    # sessions in it. Letting NSE answer costs seconds and removes a guess.
    if today >= start and now_ist.hour < PUBLISH_HOUR_IST:
        print(f"\n  Note: it is {now_ist:%H:%M} IST. Today's file is usually "
              f"published after {PUBLISH_HOUR_IST}:00 IST, so today will "
              f"probably come back empty. That is not a missing day.")

    if args.dry_run:
        print("\n  --dry-run: stopping here.")
        return

    before = read_summaries()

    if start <= today:
        import download
        print(f"\n--- download: bhavcopies ---", flush=True)
        saved, skipped, missing, absent, special, failed = \
            download.download_bhavcopies(start, today)
        print(f"\n  saved {saved}, already had {skipped}, no file {missing}")
        for d in special:
            print(f"  weekend session found: {d:%Y-%m-%d} {d:%a}")

        if failed:
            # A day NSE could not be reached for is not a holiday. Recording
            # it as one would put a silent hole in the series, so this stops.
            print(f"\n  !! {len(failed)} day(s) could not be reached:")
            for d in failed:
                print(f"     {d:%Y-%m-%d} {d:%a}")
            sys.exit("\n  Stopping rather than rebuilding around a gap. Re-run to retry.")

        if saved == 0 and not args.rebuild:
            print("\n  No new trading days. Nothing to rebuild.")
            return

        # Yahoo's series are rewritten whole rather than appended to. It costs
        # one request per symbol and removes any chance of a splice seam.
        print(f"\n--- download: index closes ---", flush=True)
        download.download_closes(first_downloaded() or start, today)

    if args.download_only:
        print("\n  --download-only: stopping before the rebuild.")
        return

    # --fast re-solves only days missing from daily_iv.csv and reuses
    # closed trades. Everything downstream -- the event study, the placebo
    # test, every average -- is recomputed over the complete series either
    # way, so the two modes differ in work done, not in what they compute.
    fast = ["--fast"] if args.fast and not args.full else []
    for symbol in INDEXES:
        run(["build.py", "--symbol", symbol] + fast, f"build {symbol}")
    for symbol in INDEXES:
        run(["backtest.py", "--symbol", symbol] + fast, f"backtest {symbol}")
    for symbol in INDEXES:
        run(["direction.py", "--symbol", symbol], f"direction {symbol}")

    after = read_summaries()
    new_latest = latest_downloaded()

    print("\n" + "=" * 64)
    print(f"  data now runs to {new_latest:%Y-%m-%d} ({new_latest:%A})")

    changes = describe_changes(before, after)
    if changes:
        print("\n  what moved:")
        for line in changes:
            print(line)
    else:
        print("  no summary figure moved")

    warnings = check_events(new_latest)
    if warnings:
        print("\n  worth checking:")
        for line in warnings:
            print(line)

    print("\n  restart the server to serve the new numbers:")
    print("    python -m uvicorn main:app --reload")


if __name__ == "__main__":
    main()
