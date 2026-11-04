"""Download NSE F&O bhavcopies plus Nifty and India VIX closes.

Runs on your laptop: NSE blocks many cloud servers, and the raw files are large.

    python download.py 2022-12-01 2026-10-30

Bhavcopies land in data/raw/YYYYMMDD.csv.zip, one per trading day. Days already
on disk are skipped, so re-running only fetches what is missing. Spot and VIX
closes are rewritten every run.

Every calendar day is tried, weekends included, because NSE opens a special
session when the Budget falls on a weekend. Days that did not trade answer 404
and cost nothing.

Each file is trimmed to the index contracts this project studies before being
saved, which is about a tenth of NSE's published size. Pass --full to keep the
files exactly as NSE publishes them (roughly 0.9 GB over the full range).
"""

import argparse
import csv
import datetime as dt
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw"

MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
       "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# nsearchives serves the zips without a cookie handshake; only the www host
# needs one, and we never call it.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

# NSE changed the bhavcopy format in mid-2024. The new layout starts in
# Jan 2024, the old one stops after Jul 2024, so the two overlap for seven
# months and either link works in that gap.
NEW_FORMAT_FROM = dt.date(2024, 1, 1)

# What to keep when trimming: index options and index futures for the two
# indices this study covers. Stock contracts are ~85% of each file and unused.
KEEP_SYMBOLS = {"NIFTY", "BANKNIFTY"}
KEEP_TYPES_NEW = {"IDO", "IDF"}        # index option, index future
KEEP_TYPES_OLD = {"OPTIDX", "FUTIDX"}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def new_url(d):
    return ("https://nsearchives.nseindia.com/content/fo/"
            f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def old_url(d):
    mon = MON[d.month - 1]
    return ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
            f"{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip")


def candidates(d):
    """Both links for a day, likeliest format first."""
    if d >= NEW_FORMAT_FROM:
        return [new_url(d), old_url(d)]
    return [old_url(d), new_url(d)]


# "NSE has no file for this day" and "we could not reach NSE" must never be
# confused: the first is a holiday, the second is a gap in our data wearing a
# holiday's clothes. An SSL or connection failure returns this instead of None.
FETCH_FAILED = object()


def fetch(url, attempts=3):
    """Return zip bytes, None if the day is genuinely absent, or FETCH_FAILED.

    FETCH_FAILED means the question never got answered -- the day may well
    have traded, and the caller must not record it as a holiday.
    """
    for attempt in range(attempts):
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                print(f"    {type(exc).__name__}: {exc}", file=sys.stderr)
                return FETCH_FAILED
            time.sleep(2 * (attempt + 1))
            continue

        if r.status_code == 404:
            return None                    # NSE answered: no such file
        if r.status_code == 200:
            if r.content[:2] == b"PK":
                return r.content
            # An empty body is not an answer either way. A holiday replies 200
            # with an HTML notice -- which has content in it -- so nothing at
            # all means a truncated or dropped reply. Returning None here
            # would record a day that may well have traded as a holiday,
            # which is exactly what FETCH_FAILED exists to prevent.
            if not r.content:
                if attempt == attempts - 1:
                    print(f"    empty 200 for {url}", file=sys.stderr)
                    return FETCH_FAILED
                time.sleep(2 * (attempt + 1))
                continue
            # A holiday can answer 200 with an HTML notice instead of a zip.
            return None
        if attempt == attempts - 1:
            print(f"    HTTP {r.status_code} for {url}", file=sys.stderr)
            return FETCH_FAILED            # 5xx, throttling, anything unclear
        time.sleep(2 * (attempt + 1))
    return FETCH_FAILED


def trim(blob):
    """Drop every row this project never reads. Returns a smaller zip.

    Anything unexpected is saved untouched rather than risking data loss: a
    layout we do not recognise, or a filter that matched nothing.

    The old format's header ends in a stray comma; that empty column holds no
    data and is not carried over.
    """
    try:
        zin = zipfile.ZipFile(io.BytesIO(blob))
        name = zin.namelist()[0]
        text = zin.read(name).decode("utf-8", "replace")
    except (zipfile.BadZipFile, IndexError, UnicodeError):
        return blob

    reader = csv.DictReader(io.StringIO(text))
    fields = [f for f in (reader.fieldnames or []) if f]
    if not fields:
        return blob

    if "FinInstrmTp" in fields:
        type_col, sym_col, keep = "FinInstrmTp", "TckrSymb", KEEP_TYPES_NEW
    elif "INSTRUMENT" in fields:
        type_col, sym_col, keep = "INSTRUMENT", "SYMBOL", KEEP_TYPES_OLD
    else:
        return blob

    rows = [r for r in reader
            if r.get(type_col) in keep and r.get(sym_col) in KEEP_SYMBOLS]
    if not rows:
        return blob

    body = io.StringIO()
    writer = csv.DictWriter(body, fieldnames=fields,
                            extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zout:
        zout.writestr(name, body.getvalue())
    return out.getvalue()


def download_bhavcopies(start, end, pause=0.3, full=False):
    RAW.mkdir(parents=True, exist_ok=True)
    saved = skipped = missing = 0
    absent = []
    special = []
    failed = []

    day = start
    while day <= end:
        dest = RAW / f"{day:%Y%m%d}.csv.zip"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            day += dt.timedelta(days=1)
            continue

        # Weekends are tried, not skipped. NSE runs a special session when the
        # Budget falls on a weekend -- Sat 1 Feb 2025 and Sun 1 Feb 2026 both
        # traded a full 9:15-3:30 day -- and those are event days the study
        # cannot afford to miss. A weekend that did not trade simply 404s, so
        # NSE's own answer decides, instead of an assumption about the calendar.
        weekend = day.weekday() > 4

        blob = None
        unreachable = False
        for url in candidates(day):
            got = fetch(url)
            if got is FETCH_FAILED:
                unreachable = True
                continue           # the other format may still answer cleanly
            if got:
                blob = got
                break

        if blob:
            raw_size = len(blob)
            if not full:
                blob = trim(blob)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            saved += 1
            if weekend:
                special.append(day)
            note = "" if full else f"  (from {raw_size:,}B)"
            tag = "   <-- weekend session" if weekend else ""
            print(f"  {day}  {len(blob):>8,}B{note}{tag}", flush=True)
        elif unreachable:
            # Never silently becomes a holiday. Re-running picks it up.
            failed.append(day)
            print(f"  {day}  !! COULD NOT REACH NSE -- re-run to retry",
                  flush=True)
        elif not weekend:
            # A weekday with no file is a holiday worth listing; a quiet
            # weekend is just a weekend, so it is not reported as missing.
            missing += 1
            absent.append(day)
            print(f"  {day}  -- no file (holiday?)", flush=True)

        # No need to be polite about a weekend that answered 404 instantly.
        time.sleep(pause if (blob or not weekend) else 0.05)
        day += dt.timedelta(days=1)

    return saved, skipped, missing, absent, special, failed


def download_closes(start, end):
    """Nifty, Bank Nifty and India VIX daily closes, via yfinance.

    Bank Nifty is here only for the pre-2024 stretch. The newer bhavcopy
    carries each index's own spot in UndrlygPric, recorded at the same instant
    as the option closes, and build.py prefers it. The old layout carries no
    spot at all, so those ~270 days need an outside source.
    """
    import yfinance as yf

    stop = end + dt.timedelta(days=1)        # yfinance end is exclusive
    for symbol, name in (("^NSEI", "spot"), ("^NSEBANK", "bank_spot"),
                         ("^INDIAVIX", "vix")):
        df = yf.download(symbol, start=start.isoformat(), end=stop.isoformat(),
                         progress=False, auto_adjust=False)
        if df.empty:
            print(f"  {name}: no data for {symbol}", file=sys.stderr)
            continue
        if getattr(df.columns, "nlevels", 1) > 1:
            df.columns = df.columns.get_level_values(0)
        out = df[["Close"]].rename(columns={"Close": "close"})
        out.index.name = "date"
        path = DATA / f"{name}.csv"
        out.to_csv(path, date_format="%Y-%m-%d", float_format="%.4f")
        print(f"  {name}: {len(out):,} rows -> {path.name} "
              f"({out.index.min():%Y-%m-%d} to {out.index.max():%Y-%m-%d})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start", help="first date, YYYY-MM-DD")
    ap.add_argument("end", nargs="?", help="last date, YYYY-MM-DD (default: today)")
    ap.add_argument("--skip-closes", action="store_true", help="only fetch bhavcopies")
    ap.add_argument("--skip-bhav", action="store_true", help="only fetch closes")
    ap.add_argument("--full", action="store_true",
                    help="save NSE's files unmodified instead of trimming them")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    if end < start:
        ap.error("end date is before start date")

    DATA.mkdir(parents=True, exist_ok=True)

    if not args.skip_bhav:
        how = "unmodified" if args.full else "trimmed to index contracts"
        print(f"Bhavcopies {start} to {end} ({how})")
        saved, skipped, missing, absent, special, failed = download_bhavcopies(
            start, end, full=args.full)
        print(f"\n  saved {saved}, already had {skipped}, no file {missing}")
        if special:
            print("  weekend sessions found (Budget days):")
            for d in special:
                print(f"    {d:%Y-%m-%d} {d:%a}")
        if absent:
            print("  days with no file (holidays or gaps):")
            for d in absent:
                print(f"    {d:%Y-%m-%d} {d:%a}")
        if failed:
            # Loud and last, so it cannot be mistaken for a holiday list.
            print(f"\n  !! {len(failed)} day(s) could not be reached. These may "
                  f"have traded -- re-run to retry:")
            for d in failed:
                print(f"    {d:%Y-%m-%d} {d:%a}")
            sys.exit(1)

    if not args.skip_closes:
        print(f"\nCloses {start} to {end}")
        download_closes(start, end)


if __name__ == "__main__":
    main()
