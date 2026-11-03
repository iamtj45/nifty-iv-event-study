"""Download the RBI statements and Budget speeches listed in docs/sources.txt.

    python fetch_docs.py

Why this exists: RBI's press-release pages are built by JavaScript and its PDF
links are opaque hashes, so there is no pattern a script can follow. The
Budget archive is no better -- only the current year sits at a predictable
address. Collecting the links is a human job, done once.

So you paste links into docs/sources.txt and this handles the rest: fetching,
checking the file really is a PDF, and naming it so rag.py can read the date
and event type straight off the filename.

Format of docs/sources.txt -- one document per line, blank lines and lines
starting with # ignored:

    2025-10-01 | RBI | https://rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/PR163....PDF
    2025-02-01 | BUDGET | https://www.indiabudget.gov.in/.../budget_speech.pdf

Files land in docs/ as, for example:

    2025-10-01_RBI.pdf
"""

import sys
import time
from pathlib import Path

import requests

DOCS = Path(__file__).parent / "docs"
SOURCES = DOCS / "sources.txt"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def read_sources():
    """Parse docs/sources.txt into (date, kind, url) triples."""
    if not SOURCES.exists():
        sys.exit(f"{SOURCES} does not exist -- see the docstring for the format")

    entries, problems = [], []
    for lineno, raw in enumerate(SOURCES.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            problems.append(f"  line {lineno}: expected 'date | kind | url', got {raw!r}")
            continue
        date, kind, url = parts
        if not url.lower().startswith("http"):
            problems.append(f"  line {lineno}: {url!r} is not a URL")
            continue
        entries.append((date, kind.upper(), url))

    if problems:
        print("Lines I could not read:")
        print("\n".join(problems), file=sys.stderr)
    return entries


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    entries = read_sources()
    if not entries:
        sys.exit("nothing to fetch -- docs/sources.txt has no usable lines")

    session = requests.Session()
    session.headers.update(HEADERS)

    saved = skipped = failed = 0
    for date, kind, url in entries:
        dest = DOCS / f"{date}_{kind}.pdf"
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue

        try:
            r = session.get(url, timeout=60)
        except requests.RequestException as exc:
            print(f"  {dest.name:28} FAILED  {type(exc).__name__}", file=sys.stderr)
            failed += 1
            continue

        if r.status_code != 200:
            print(f"  {dest.name:28} FAILED  HTTP {r.status_code}", file=sys.stderr)
            failed += 1
            continue
        if r.content[:5] != b"%PDF-":
            # A site that has moved a file often answers 200 with an HTML
            # "not found" page. Saving that would poison the search index with
            # navigation text, so it is rejected here rather than later.
            print(f"  {dest.name:28} FAILED  not a PDF (got "
                  f"{r.headers.get('Content-Type', 'unknown')})", file=sys.stderr)
            failed += 1
            continue

        dest.write_bytes(r.content)
        print(f"  {dest.name:28} {len(r.content):>9,}B")
        saved += 1
        time.sleep(0.4)

    print(f"\n  saved {saved}, already had {skipped}, failed {failed}")
    if failed:
        print("  fix the failing links in docs/sources.txt and re-run")
        sys.exit(1)


if __name__ == "__main__":
    main()
