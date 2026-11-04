"""Regenerate the numbers in README.md from the result files.

    python update_readme.py           rewrite the marked blocks
    python update_readme.py --check   fail if any block is out of date

Why this exists. The README carried about forty hand-typed figures, and the
nightly job adds a trading day to the data every evening. One night after it
was written, five of the six coverage numbers were already wrong: 968 trading
days had become 969, the placebo pool 448 had become 450. None of that is
important in itself, and that is exactly the problem -- a README that is
casually wrong about its own day count is not a README anyone trusts about
the 0.04 percentile.

Two halves, and both are needed.

The rewrite keeps the numbers exact. Each block below is generated from the
same CSVs the website serves, so the README cannot disagree with the page.

The `--check` mode is what makes it reliable rather than merely automatic. CI
runs it on every push, so a stale README fails the build instead of being
published quietly. Automation nobody verifies is worse than no automation --
it just moves the staleness somewhere less visible.

What is NOT generated: the findings themselves, the limits, the design notes,
every word of prose. Those are arguments, and an argument that rewrites itself
is not an argument. Only the figures move.

The as-of stamp is the DATA's last date, not today's. The honest question a
reader has is "how current are these numbers", and the day the script happened
to run does not answer it.
"""

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"
README = BASE / "README.md"

INDEX_DIRS = {"NIFTY": DATA, "BANKNIFTY": DATA / "banknifty"}

# Each generated region sits between these, and everything outside is left
# exactly as written. The markers are HTML comments, so they are invisible on
# GitHub.
START = "<!-- AUTO:{name} -->"
END = "<!-- /AUTO:{name} -->"

LABEL = {"RBI": "RBI policy", "BUDGET": "Union Budget",
         "ELECTION": "Election result"}

# Short forms for the fixed-width placebo block. str.title() would render RBI
# as "Rbi", which looks like a typo in a file whose whole point is care.
SHORT = {"RBI": "RBI", "BUDGET": "Budget", "ELECTION": "Election"}

# The README is written with real typography: U+2212 MINUS SIGN for negative
# numbers, not the hyphen an f-string produces. It is a small thing that is
# very visible next to the hand-written prose around it.
MINUS = "−"


def signed(value, digits=2, plus=False):
    """A number with a proper minus sign, matching the prose around it."""
    text = f"{abs(value):.{digits}f}"
    if value < 0:
        return MINUS + text
    return ("+" if plus else "") + text


# The README is hard-wrapped by hand at this width. Generated prose has to
# match, or the file reads as two documents stitched together -- which it then
# is, and the seam is the first thing a careful reader notices.
WRAP = 78


def wrap(text):
    """Re-wrap a generated paragraph to the width the rest of the file uses.

    Markdown joins consecutive lines when it renders, so this changes nothing
    on GitHub. It matters in the source, where a 300-character line sitting
    among 78-character ones is a visible tell that something was pasted.
    """
    import textwrap
    return "\n".join(
        textwrap.fill(para.replace("\n", " "), width=WRAP,
                      break_long_words=False, break_on_hyphens=False)
        for para in text.split("\n\n"))

# Files listed in the "What is here" table, in the order they should appear.
FILES = ["download.py", "iv.py", "build.py", "analysis.py", "backtest.py",
         "direction.py", "update.py", "update_readme.py", "main.py", "rag.py",
         "run_evals.py", "mcp_server.py", "cpp/iv.cpp", "bench.py"]

BLURB = {
    "download.py": "Fetches NSE bhavcopies and index closes",
    "iv.py": "Black-Scholes and the Brent solve for IV",
    "build.py": "Raw files → daily IV, event study, placebo",
    "analysis.py": "Bootstrap intervals, placebo test, expected move",
    "backtest.py": "The straddle rule, events against ordinary days",
    "direction.py": "Sign tests, and volatility against direction",
    "update.py": "Brings everything up to the last trading day",
    "update_readme.py": "Regenerates this file's figures, and checks them in CI",
    "main.py": "FastAPI: the API and the question box",
    "rag.py": "PDF passages and BM25 retrieval",
    "run_evals.py": "The four eval checks",
    "mcp_server.py": "MCP tools so an assistant can query the study",
    "cpp/iv.cpp": "The solver again, in C++, for verification",
    "bench.py": "Accuracy and speed, Python against C++",
}


def load(index):
    """Every result file for one index, or exit saying which is missing."""
    folder = INDEX_DIRS[index]
    out = {}
    for name in ("daily_iv", "results", "summary", "placebo", "curves",
                 "backtest_summary", "direction", "vol_direction"):
        path = folder / f"{name}.csv"
        if not path.exists():
            sys.exit(f"{path} is missing -- run build.py, backtest.py and "
                     f"direction.py first")
        out[name] = pd.read_csv(
            path, parse_dates=["date"] if name == "daily_iv" else None)
    return out


def fmt_date(value):
    """'3 Nov 2026' -- no leading zero, which %-d cannot do on Windows."""
    stamp = pd.Timestamp(value)
    return f"{stamp.day} {stamp:%b %Y}"


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------

def block_coverage(n, b):
    daily, results = n["daily_iv"], n["results"]
    first, last = daily["date"].min(), daily["date"].max()
    counts = results["type"].value_counts()
    mix = " · ".join(
        f"{int(counts[k])} {word}"
        for k, word in (("RBI", "RBI"), ("BUDGET", "Budget"),
                        ("ELECTION", "election"))
        if k in counts)
    built = sum(1 for d in INDEX_DIRS.values() if (d / "summary.csv").exists())
    fence = (f"```\n"
             f"{len(daily)} trading days   {fmt_date(first)} – {fmt_date(last)}\n"
             f"{len(results)} events          {mix}\n"
             f"{built} indices          Nifty 50 and Bank Nifty\n"
             f"```\n\n")
    note = wrap(
        f"*Figures throughout are as of {fmt_date(last)}, the last trading day "
        f"in the data. They are regenerated from the result files by "
        f"`update_readme.py`, and CI fails if they drift — so this page and "
        f"the live site cannot disagree.*")
    return fence + note


def block_findings(n, b):
    s = n["summary"].set_index("type")
    lines = ["| Event | n | IV charged before | Volatility delivered | Overpriced in |",
             "|---|---:|---:|---:|---:|"]
    for kind in ("RBI", "BUDGET", "ELECTION"):
        if kind not in s.index:
            continue
        r = s.loc[kind]
        count = int(r["n"])
        over = f"{int(r['overpriced'])} of {count}"
        # The headline result is bolded; the others are context.
        cell = f"**{over}**" if kind == "RBI" else over
        lines.append(f"| {LABEL[kind]} | {count} | {r['avg_iv_before']:.2f}% "
                     f"| {r['avg_realised_vol']:.2f}% | {cell} |")
    rbi = s.loc["RBI"]
    lines += ["", wrap(
        f"Before RBI meetings the market priced in {rbi['avg_iv_before']:.2f}% "
        f"of volatility and got {rbi['avg_realised_vol']:.2f}%. On the day "
        f"itself implied volatility fell "
        f"**{abs(rbi['avg_change_pct']):.2f}%** on average.")]
    return "\n".join(lines)


def block_placebo(n, b):
    p = n["placebo"].set_index("type")
    s = n["summary"].set_index("type")
    pool = int(p["normal_days"].iloc[0])
    n_rbi = int(p.loc["RBI", "n_events"])
    rows = []
    for kind, note in (("RBI", "none of 10,000 random groups were as extreme"),
                       ("BUDGET", ""),
                       ("ELECTION", "went the other way, as expected")):
        if kind not in p.index:
            continue
        pct = p.loc[kind, "percentile_vs_normal"]
        rows.append(f"{SHORT[kind]:9}{pct:>6.2f} percentile"
                    + (f"      {note}" if note else ""))
    rbi = s.loc["RBI"]
    intro = wrap(f"A placebo test takes {pool} ordinary trading days — at least "
                 f"ten days from any event — draws {n_rbi} of them at random, "
                 f"averages, and repeats 10,000 times.")
    tail = wrap(f"The 95% confidence range on the RBI crush is "
                f"**{signed(rbi['change_ci_low'])}% to "
                f"{signed(rbi['change_ci_high'])}%** — entirely below zero.")
    return (intro + "\n\n```\n" + "\n".join(rows) + "\n```\n\n" + tail)


def block_direction(n, b):
    d = n["direction"].set_index("type")
    lines = ["| | up | down | % up | coin-flip p |",
             "|---|---:|---:|---:|---:|"]
    for kind, label in (("RBI", "RBI days"), ("ORDINARY", "ordinary days")):
        if kind not in d.index:
            continue
        r = d.loc[kind]
        lines.append(f"| {label} | {int(r['up'])} | {int(r['down'])} | "
                     f"{r['pct_up']:.1f}% | {r['sign_test_p']:.3f} |")
    rbi, ordinary = d.loc["RBI"], d.loc["ORDINARY"]
    lines += ["", wrap(
        f"Ordinary days rise {ordinary['pct_up']:.1f}% of the time — the slow "
        f"upward drift of equities. RBI days rise {rbi['pct_up']:.1f}%. "
        f"**Knowing a meeting is tomorrow leaves you with less directional "
        f"information than an ordinary Tuesday.** The decision is in the price "
        f"before it is announced.")]
    return "\n".join(lines)


def block_backtest(n, b):
    bt = n["backtest_summary"].set_index("group")
    lines = ["| Group | Trades | Avg net | Won | Worst loss | Break-even cost |",
             "|---|---:|---:|---:|---:|---:|"]
    order = [("RBI", "RBI"), ("BUDGET", "Budget"), ("ELECTION", "Election"),
             ("NORMAL", "**ordinary days**")]
    for key, label in order:
        if key not in bt.index:
            continue
        r = bt.loc[key]
        gross = signed(r["avg_gross"])
        cell = f"**{gross}**" if key in ("RBI", "NORMAL") else gross
        lines.append(f"| {label} | {int(r['n'])} | "
                     f"{signed(r['avg_net'], plus=True)} | "
                     f"{r['win_rate']:.1f}% | {signed(r['worst'])} | {cell} |")

    rbi, norm = bt.loc["RBI"], bt.loc["NORMAL"]
    worst = bt.loc[bt["worst"].idxmin()]
    total = int(bt["n"].sum())
    lines += ["", wrap(
        '"Break-even cost" is profit before any fees — which is also exactly '
        "the round-trip cost at which the rule stops paying."),
        "", wrap(
        f"**Selling before RBI meetings earned {signed(rbi['avg_gross'])} points "
        f"before costs. Selling on a random Tuesday earned "
        f"{signed(norm['avg_gross'])}.** The event trade is worse than trading "
        f"at random, because event days also *move* more, and the extra "
        f"movement costs the seller more than the collapsing volatility pays."),
        "", wrap(
        # Both halves are worth saying, and the hand-written version had this
        # wrong: it called an ordinary Monday the worst trade of all when the
        # election lost nearly twice as much.
        f"The worst single trade of all {total} was the "
        f"{LABEL.get(worst.name, worst.name).lower()}: "
        f"{fmt_date(worst['worst_day'])}, {signed(worst['worst'], 0)} points. "
        f"The worst ordinary day lost {signed(norm['worst'], 0)} — a deeper "
        f"hole than any RBI meeting produced, on a day when nothing was "
        f"scheduled at all.")]
    return "\n".join(lines)


def block_voldir(n, b):
    v = n["vol_direction"].set_index("measure")
    same = v[v["horizon"] == "same day"]
    ahead = v[v["horizon"] == "next day"]

    def find(frame, needle):
        hit = frame[frame.index.str.contains(needle, case=False)]
        return hit.iloc[0] if len(hit) else None

    same_corr = find(same, "correlation")
    next_corr = find(ahead, "correlation")
    fell_on_rose = find(same, "fell, on days IV rose")
    rose_on_fell = find(same, "rose, on days IV fell")
    spike = find(ahead, "biggest 10%")
    base = find(ahead, "baseline")

    lines = ["| | correlation | days |", "|---|---:|---:|",
             f"| IV change vs **same-day** return | "
             f"**{signed(same_corr['value'], 4)}** | {int(same_corr['n'])} |",
             f"| IV change vs **next-day** return | "
             f"{signed(next_corr['value'], 4, plus=True)} "
             f"| {int(next_corr['n'])} |",
             "", wrap(
             f"On days volatility rose, the market fell "
             f"**{fell_on_rose['value']:.0f}%** of the time. On days it fell, "
             f"the market rose **{rose_on_fell['value']:.0f}%** of the time."),
             "", wrap(
             f"But after the largest 10% of volatility spikes, the next day "
             f"rose {spike['value']:.1f}% — against a {base['value']:.1f}% "
             f"baseline. **Volatility measures today's direction accurately "
             f"and forecasts tomorrow's not at all.** Fear rising and prices "
             f"falling are the same event seen two ways, not a cause and its "
             f"effect.")]
    return "\n".join(lines)


def block_vix(n, b):
    nd, bd = n["daily_iv"], b["daily_iv"]
    both = nd[["atm_iv", "vix"]].dropna()
    corr = both.corr().iloc[0, 1]

    nbt = n["backtest_summary"].set_index("group")
    bbt = b["backtest_summary"].set_index("group")
    bp = b["placebo"].set_index("type")
    bs = b["summary"].set_index("type")
    ns = n["summary"].set_index("type")

    n_crush = ns.loc["RBI", "avg_iv_after"] - ns.loc["RBI", "avg_iv_before"]
    b_crush = bs.loc["RBI", "avg_iv_after"] - bs.loc["RBI", "avg_iv_before"]

    fence = (f"```\n"
             f"our ATM IV       {nd['atm_iv'].mean():.2f}\n"
             f"India VIX        {nd['vix'].mean():.2f}\n"
             f"correlation      {corr:.4f}   over {len(both)} overlapping days\n"
             f"```")
    paras = [
        wrap("The ~1 point gap is expected and, importantly, **steady**. A "
             "broken solver would drift — fine in calm markets, wrong in "
             "volatile ones. A constant offset between two methods measuring "
             "slightly different things is what correctness looks like."),
        wrap(f"**Bank Nifty corroborates independently.** Run through the same "
             f"pipeline it gives {bd['atm_iv'].mean():.2f}% mean IV against "
             f"Nifty's {nd['atm_iv'].mean():.2f}% — higher, as a single-sector "
             f"index should be — and reaches the same conclusions: overpriced "
             f"{int(bs.loc['RBI', 'overpriced'])} of "
             f"{int(bs.loc['RBI', 'n'])}, placebo "
             f"{bp.loc['RBI', 'percentile_vs_normal']:.2f}, and a straddle "
             f"trade that *loses* "
             f"{abs(bbt.loc['RBI', 'avg_gross']):.1f} points before costs."),
        wrap(f"The crush is also the same size in **absolute** volatility "
             f"points in both indices (Nifty {signed(n_crush)} against Bank "
             f"Nifty {signed(b_crush)}), not proportionally larger in the "
             f"rate-sensitive banking index. The event premium is a roughly "
             f"fixed quantity of volatility, not a percentage markup."),
    ]
    return fence + "\n\n" + "\n\n".join(paras)


def block_files(n, b):
    lines = ["| File | Lines | What it does |", "|---|---:|---|"]
    for name in FILES:
        path = BASE / name
        if not path.exists():
            continue
        count = len(path.read_text(encoding="utf-8").splitlines())
        lines.append(f"| `{name}` | {count} | {BLURB[name]} |")
    tests = sorted((BASE / "tests").glob("test_*.py"))
    test_lines = sum(len(p.read_text(encoding="utf-8").splitlines())
                     for p in tests)
    n_tests = count_tests()
    lines.append(f"| `tests/` | {test_lines} | {n_tests} tests on the maths "
                 f"and the statistics |")
    lines += ["", "```bash",
              f"pytest          # {n_tests} tests",
              "python bench.py # C++ against Python: agreement and speed",
              "```"]
    return "\n".join(lines)


def count_tests():
    """How many tests pytest actually collects, rather than a number we typed."""
    try:
        done = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=BASE, capture_output=True, text=True, timeout=180)
        for line in reversed(done.stdout.splitlines()):
            if "test" in line and "collected" in line:
                return int(line.split()[0])
    except Exception:
        pass
    return 0


BLOCKS = {
    "COVERAGE": block_coverage,
    "FINDINGS": block_findings,
    "PLACEBO": block_placebo,
    "DIRECTION": block_direction,
    "BACKTEST": block_backtest,
    "VOLDIR": block_voldir,
    "VIX": block_vix,
    "FILES": block_files,
}

# Blocks --check will rewrite but not fail over.
#
# FILES counts lines of source, so editing any file in the project makes it
# stale. Failing CI for that would mean a one-line change to main.py breaks
# the build over a line-count table -- friction with no safety in it, and the
# fastest way to teach everyone to ignore the check. The figures that could
# actually contradict the live site are all enforced; a line count cannot
# mislead anyone about anything.
ADVISORY = {"FILES"}


# ---------------------------------------------------------------------------

def splice(text, name, body):
    """Replace the content between one pair of markers."""
    open_tag, close_tag = START.format(name=name), END.format(name=name)
    i, j = text.find(open_tag), text.find(close_tag)
    if i < 0 or j < 0:
        sys.exit(f"README.md has no {open_tag} / {close_tag} pair -- add the "
                 f"markers around the block this generates, or remove "
                 f"'{name}' from BLOCKS.")
    if j < i:
        sys.exit(f"{close_tag} appears before {open_tag} in README.md")
    return (text[:i + len(open_tag)] + "\n" + body.rstrip() + "\n"
            + text[j:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="change nothing; exit 1 if any block is out of date")
    args = ap.parse_args()

    if not README.exists():
        sys.exit(f"{README} is missing")

    nifty = load("NIFTY")
    bank = load("BANKNIFTY")

    before = README.read_text(encoding="utf-8")
    after = before
    for name, build in BLOCKS.items():
        after = splice(after, name, build(nifty, bank))

    # Which blocks actually moved, so both modes can be specific.
    stale = [name for name, build in BLOCKS.items()
             if splice(before, name, build(nifty, bank)) != before]

    if args.check:
        enforced = [n for n in stale if n not in ADVISORY]
        for name in stale:
            tag = "advisory" if name in ADVISORY else "STALE   "
            print(f"  {tag}  {name}")
        if not enforced:
            print("README.md figures match the result files")
            return
        print()
        print("README.md is OUT OF DATE relative to the result files.")
        print("Run:  python update_readme.py")
        sys.exit(1)

    if not stale:
        print("README.md is up to date")
        return

    README.write_text(after, encoding="utf-8")
    print(f"README.md rewritten -- {len(stale)} of {len(BLOCKS)} blocks changed"
          f"  ({', '.join(stale)})")


if __name__ == "__main__":
    main()
