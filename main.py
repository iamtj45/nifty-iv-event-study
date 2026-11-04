"""The web app: a small API over the result files, plus the page.

    uvicorn main:app --reload

This module never downloads data, never reads a bhavcopy and never solves an
implied volatility. It reads the CSVs that build.py already produced and hands
them to the page as JSON. Every number on the page was computed offline by code
that has tests; the server only serves it.

Routes
    GET  /                 the page
    GET  /api/results      everything about one event type
    GET  /api/daily        the daily IV series, for the India VIX check
    POST /api/ask          the question box (phase 5; not wired up yet)
    GET  /docs             FastAPI's own API page
"""

import datetime as dt
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# The cost assumption is reported with the backtest rather than restated here,
# so the API and the file that computed it can never disagree.
from backtest import COST_PCT, COST_POINTS

# Load .env if it exists, so an API key can live in a gitignored file rather
# than being retyped into every shell. Real environment variables always win,
# which is what lets Render's dashboard override it in production without the
# file being involved at all. Missing file, or missing package: both fine, and
# the question box reports the absence rather than failing.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

BASE = Path(__file__).parent
DATA = BASE / "data"
STATIC = BASE / "static"

app = FastAPI(
    title="Nifty IV Event Study",
    description="How Nifty implied volatility behaves around the Union Budget, "
                "RBI policy meetings and election results.",
    version="1.0",
)


# ---------------------------------------------------------------------------
# The result files are read once, when the server starts.
#
# They only change when build.py runs, which happens offline. Re-reading a
# 967-row CSV on every request would be wasted work on data that cannot have
# changed since the process booted, and it would put disk latency in front of
# every page load. Loading once costs a few megabytes of memory and makes every
# request a dictionary lookup.
#
# The consequence to remember: after a rebuild, the server must be restarted
# to see new numbers. On Render that happens by itself, because a git push
# redeploys the whole container.
# ---------------------------------------------------------------------------

def _read(name, folder, required=True, **kwargs):
    path = folder / f"{name}.csv"
    if not path.exists():
        if not required:
            return None
        raise SystemExit(f"{path} is missing -- run build.py first")
    return pd.read_csv(path, **kwargs)


# Each index's results live in its own folder, matching build.py's INDEXES.
# NIFTY stayed in data/ so nothing that already worked had to move.
INDEX_DIRS = {
    "NIFTY": DATA,
    "BANKNIFTY": DATA / "banknifty",
}
DEFAULT_INDEX = "NIFTY"


class Store:
    """Everything the API serves for one index, held in memory."""

    def __init__(self, index, folder):
        self.index = index
        self.summary = _read("summary", folder)
        self.placebo = _read("placebo", folder)
        self.results = _read("results", folder,
                             parse_dates=["event_date", "trading_day"])
        self.curves = _read("curves", folder)
        self.daily = _read("daily_iv", folder, parse_dates=["date"])

        # The backtest is optional: the API must still serve an index whose
        # build has run but whose backtest has not, rather than refuse to
        # start. The page hides the panel when this is None.
        self.backtest = _read("backtest_summary", folder, required=False)
        self.direction = _read("direction", folder, required=False)
        self.direction_events = _read("direction_events", folder,
                                      required=False)
        self.vol_direction = _read("vol_direction", folder, required=False)

        self.types = sorted(self.summary["type"].tolist())
        self.latest = self.daily["date"].max().date()
        self.first = self.daily["date"].min().date()
        self.trading_days = len(self.daily)


# >>> WRITE HERE — snippet 1 (one line): build the Store once, at import time.
# One Store per index whose build has actually run. An index listed above but
# never built is skipped rather than crashing the server, so the site comes up
# with whatever is ready.
STORES = {name: Store(name, folder)
          for name, folder in INDEX_DIRS.items()
          if (folder / "summary.csv").exists()}

if not STORES:
    raise SystemExit("no built results found -- run build.py first")

STORE = STORES[DEFAULT_INDEX]


def store_for(index):
    """The Store for one index, or a clear error."""
    key = index.upper()
    if key not in INDEX_DIRS:
        raise HTTPException(
            400, f"unknown index {index!r}; try one of {sorted(INDEX_DIRS)}")
    if key not in STORES:
        raise HTTPException(
            404, f"{key} is configured but not built yet -- "
                 f"run build.py --symbol {key}")
    return STORES[key]


def clean(value):
    """Turn pandas and NumPy values into something JSON can carry.

    NaN is not valid JSON. Left alone it serialises to a bare `NaN` token that
    JSON.parse rejects, so the page fails to load rather than showing a gap.
    Every missing number becomes null instead, and the page decides how to
    draw a hole.
    """
    if value is None:
        return None
    # NaT first, and this order is not cosmetic. isinstance(pd.NaT,
    # datetime.datetime) is True, so the date branch below catches NaT and
    # pd.Timestamp(pd.NaT).date().isoformat() returns the *string* "NaT".
    # That reaches the page as a date, and it slips past CI's NaN grep
    # because the token is spelled differently. A missing date is null.
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, dt.date, dt.datetime)):
        return pd.Timestamp(value).date().isoformat()
    return value


def records(frame):
    """A DataFrame as a list of JSON-safe dicts."""
    return [{k: clean(v) for k, v in row.items()}
            for row in frame.to_dict(orient="records")]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/meta", summary="What this dataset covers")
def meta(index: str = Query(DEFAULT_INDEX, description="NIFTY or BANKNIFTY")):
    """Enough for the page to label itself before any data is drawn.

    The coverage figures are per index. They used to come from the Nifty
    store whatever the reader was looking at, which was invisible only
    because both indices happen to run to the same 969 days and 29 events --
    it would have started lying the moment one of them moved.

    The lists at the bottom are not per index: they describe what the server
    has, which is what the page needs to build its switcher. Called with no
    index at all this answers for Nifty, so Render's health check still works
    as a bare GET.
    """
    store = store_for(index)
    return {
        "index": store.index,
        "first_date": store.first.isoformat(),
        "latest_date": store.latest.isoformat(),
        "trading_days": store.trading_days,
        "event_types": store.types,
        "events": len(store.results),
        # In INDEX_DIRS order, not alphabetical: Nifty is the study and Bank
        # Nifty is the corroboration, and the page shows them in that order.
        "indexes": [k for k in INDEX_DIRS if k in STORES],
        "default_index": DEFAULT_INDEX,
        "has_backtest": sorted(k for k, s in STORES.items()
                               if s.backtest is not None),
        "has_direction": sorted(k for k, s in STORES.items()
                                if s.direction is not None),
    }


@app.get("/api/results", summary="Everything about one event type")
def results(
    type:str=Query("RBI",description="RBI,BUDGET or ELECTION"),
    index:str=Query(DEFAULT_INDEX,description="NIFTY or BANKNIFTY"),
):
    """Summary, placebo verdict, IV curves and the event table, in one call.

    One request rather than four. The page needs all of it before it can draw
    anything, and four round trips would mean four chances to render half a
    screen.
    """
    store = store_for(index)

    event_type = type.upper()
    if event_type not in store.types:
        raise HTTPException(
            404, f"unknown event type {type!r}; try one of {store.types}")

    summary = store.summary[store.summary["type"] == event_type]
    placebo = store.placebo[store.placebo["type"] == event_type]
    curves = store.curves[store.curves["type"] == event_type].sort_values("offset")
    events = (store.results[store.results["type"] == event_type]
              .sort_values("event_date", ascending=False))

    return {
        "type": event_type,
        "index": store.index,
        "latest_date": store.latest.isoformat(),
        "summary": records(summary)[0],
        "placebo": records(placebo)[0] if len(placebo) else None,
        "curve": records(curves),
        "events": records(events),
    }


@app.get("/api/daily", summary="Daily ATM IV against India VIX")
def daily(
    start: str | None = Query(None, description="YYYY-MM-DD"),
    end: str | None = Query(None, description="YYYY-MM-DD"),
    index: str = Query(DEFAULT_INDEX, description="NIFTY or BANKNIFTY"),
):
    """The validation series: our IV next to NSE's own volatility index.

    Defaults to the last two years. The full 967 days is a lot to push down the
    wire for a chart a few hundred pixels wide, and the point of this panel is
    that the two lines move together -- which two years shows as well as four.

    For BANKNIFTY the `vix` column is still India VIX, which is a Nifty
    measure: NSE publishes no Bank Nifty volatility index. The correlation is
    returned anyway as context, but it is not the same validation it is for
    Nifty, and the page labels it accordingly.
    """
    store = store_for(index)
    frame = store.daily

    # `end` is applied first, then the window is measured back from whatever
    # is left. Asking for an end date alone used to skip the two-year default
    # entirely and return the whole series from 2022 -- the one combination
    # that ignored the cap it exists to enforce.
    if end:
        frame = frame[frame["date"] <= pd.Timestamp(end)]
    if start:
        frame = frame[frame["date"] >= pd.Timestamp(start)]
    elif len(frame):
        frame = frame[frame["date"] >= frame["date"].max() - pd.Timedelta(days=730)]

    frame = frame[["date", "atm_iv", "next_iv", "vix", "skew"]]
    both = frame[["atm_iv", "vix"]].dropna()
    correlation = float(both.corr().iloc[0, 1]) if len(both) > 2 else None

    return {
        "index": store.index,
        "vix_is_native": store.index == "NIFTY",
        "points": records(frame),
        "correlation": round(correlation, 4) if correlation is not None else None,
        "n": len(frame),
    }


@app.get("/api/backtest", summary="Would selling the straddle have paid?")
def backtest(index: str = Query(DEFAULT_INDEX, description="NIFTY or BANKNIFTY")):
    """The straddle backtest, one row per group, plus the control group.

    The NORMAL row is the point of the whole endpoint: the same rule applied
    to 448 ordinary days. An event group that does not beat it has no edge,
    whatever its average profit looks like on its own.

    `avg_gross` is profit before any costs, which is also exactly the
    round-trip cost at which the rule breaks even -- the honest way to present
    a result that depends on a cost assumption.
    """
    store = store_for(index)
    if store.backtest is None:
        raise HTTPException(
            404, f"no backtest for {store.index} yet -- "
                 f"run backtest.py --symbol {store.index}")

    rows = records(store.backtest)
    return {
        "index": store.index,
        "groups": [r for r in rows if r["group"] != "NORMAL"],
        "control": next((r for r in rows if r["group"] == "NORMAL"), None),
        "cost_points": COST_POINTS,
        "cost_pct": COST_PCT,
    }


@app.get("/api/direction", summary="Does an event predict which way?")
def direction(index: str = Query(DEFAULT_INDEX, description="NIFTY or BANKNIFTY")):
    """The direction test: is the up/down split lopsided, or is it a coin?

    Every other measure in this study is about how far the index moves, not
    which way -- a straddle seller does not care about direction. This is the
    one endpoint that asks the question people actually arrive with, and the
    answer is mostly no.

    The ORDINARY row is the one to read first. Ordinary days drift slightly
    up, so an event group has to be read against that rather than against
    zero, and on this measure RBI days carry less directional information
    than an ordinary Tuesday does.
    """
    store = store_for(index)
    if store.direction is None:
        raise HTTPException(
            404, f"no direction test for {store.index} yet -- "
                 f"run direction.py --symbol {store.index}")

    rows = records(store.direction)
    return {
        "index": store.index,
        "groups": [r for r in rows if r["type"] != "ORDINARY"],
        "baseline": next((r for r in rows if r["type"] == "ORDINARY"), None),
        "volatility_direction": (records(store.vol_direction)
                                 if store.vol_direction is not None else []),
    }


class Question(BaseModel):
    question: str = Field(..., max_length=500,
                          description="A question about the results")
    # Which index the reader is looking at. The table sent to the model holds
    # both, so without this the model has no way to know that "IV before the
    # Budget" means the one on screen, and sensibly answers for both.
    index: str = Field(DEFAULT_INDEX, max_length=20,
                       description="NIFTY or BANKNIFTY - the panel in view")


# How many questions one visitor may ask per minute. The LLM costs money per
# call, and this page is public: without a cap, one script could spend the
# whole budget in a minute. An in-memory counter is enough -- it resets when
# the server restarts, which is fine for a project site.
ASK_LIMIT_PER_MINUTE = 6
_ask_log: dict[str, list[float]] = {}


def rate_limited(caller):
    now = time.time()
    # Forget callers who have gone quiet. Without this the dict keeps one
    # entry per visitor IP for the life of the process -- harmless at this
    # traffic, unbounded in principle, and the kind of thing that is much
    # easier to fix now than to diagnose later.
    for who in [k for k, seen in _ask_log.items()
                if not seen or now - seen[-1] >= 60]:
        del _ask_log[who]

    recent = [t for t in _ask_log.get(caller, []) if now - t < 60]
    _ask_log[caller] = recent + [now]
    return len(recent) >= ASK_LIMIT_PER_MINUTE


def results_table_text():
    """The events table as plain text, small enough to send with every question.

    The model is given the numbers directly rather than being asked to look
    them up, so it can quote from them without any chance of arithmetic.

    Both indexes are included, each row labelled, because the page now shows
    both and a visitor may ask about either. Leaving Bank Nifty out would mean
    the model refusing questions the site visibly answers.
    """
    lines = ["index | event | type | date | IV before | IV on day | IV change % | "
             "realised vol after | overpriced | index move on the day %"]
    moves = {}
    for name in sorted(STORES):
        de = STORES[name].direction_events
        if de is None:
            continue
        for row in de.itertuples():
            moves[(name, str(row.event_date))] = f"{row.move_on_day:+.2f}"

    for name in sorted(STORES):
        store = STORES[name]
        for row in store.results.sort_values("event_date").itertuples():
            lines.append(
                f"{name} | {row.event} | {row.type} | {row.event_date.date()} | "
                f"{row.iv_before:.2f} | {row.iv_after:.2f} | "
                f"{row.change_pct:.2f} | {row.realised_vol:.2f} | "
                f"{'yes' if row.overpriced else 'no'} | "
                f"{moves.get((name, str(row.event_date.date())), 'n/a')}")

    for name in sorted(STORES):
        bt = STORES[name].backtest
        if bt is None:
            continue
        lines.append("")
        lines.append(f"{name} STRADDLE BACKTEST: "
                     f"group | trades | avg net points | win % | worst loss | "
                     f"avg gross (= break-even cost)")
        for row in bt.itertuples():
            lines.append(
                f"{name} | {row.group} | {row.n} | {row.avg_net:.2f} | "
                f"{row.win_rate:.1f} | {row.worst:.2f} | {row.avg_gross:.2f}")

    for name in sorted(STORES):
        d = STORES[name].direction
        if d is None:
            continue
        lines.append("")
        lines.append(f"{name} DIRECTION TEST (does an event predict which way?):"
                     f" group | events | up | down | % up | average move % | "
                     f"coin-flip p-value | verdict")
        for row in d.itertuples():
            lines.append(
                f"{name} | {row.type} | {row.n} | {row.up} | {row.down} | "
                f"{row.pct_up} | {row.mean_move:+.3f} | {row.sign_test_p} | "
                f"{row.verdict}")

    for name in sorted(STORES):
        v = STORES[name].vol_direction
        if v is None:
            continue
        lines.append("")
        lines.append(f"{name} VOLATILITY AND DIRECTION (does volatility tell "
                     f"you which way the market went?): measure | horizon | "
                     f"value | days | note")
        for row in v.itertuples():
            lines.append(f"{name} | {row.measure} | {row.horizon} | "
                         f"{row.value} | {row.n} | {row.note}")

    return "\n".join(lines)


# ===========================================================================
# WRITE THIS ONE
# ===========================================================================

REFUSAL = "That's not in the results."


def build_prompt(question, table, passages, index=DEFAULT_INDEX):
    """The instructions the model is sent, along with the question.

    This is the most important text in the AI half of the project, and it is
    prose rather than code because that is what the model reads. Everything
    that stops it inventing a confident wrong number lives here.

    A language model's default behaviour is to be helpful, and a helpful model
    asked "what was IV before the March 2025 RBI meeting?" will produce a
    plausible figure whether or not one exists. There was no RBI meeting in
    March 2025. The only thing standing between that question and a fabricated
    answer is this prompt.

    YOUR CODE HERE. Return one string containing, in this order:

      1. A line saying what the model is: something like "You answer questions
         about a study of Nifty implied volatility around Indian market
         events."

      2. The rules, as a short list. These five matter:
           - Use ONLY the results table and the passages below. Never use
             anything you remember about Indian markets.
           - Quote numbers exactly as they appear. Do not calculate anything,
             do not round, do not estimate.
           - Name your source: the event row, or the document and page.
           - If the answer is not in what you were given, reply with exactly:
             {REFUSAL}
           - Never give trading advice, predictions, or opinions about what
             anyone should do.

      3. The heading "RESULTS TABLE:" then `table`.

      4. The heading "DOCUMENT PASSAGES:" then each passage in `passages`,
         labelled so the model can cite it -- for example:
             [doc 2025-10-01_RBI.pdf, page 3]
             <the passage text>
         If `passages` is empty, say so plainly: "none found for this
         question".

      5. The heading "QUESTION:" then `question`.

    Why the refusal line must be word-for-word: run_evals.py checks for that
    exact string. A model that refuses in its own words each time cannot be
    tested automatically, and an AI feature you cannot test is one you cannot
    put your name to.

    Why the question goes last: models weight the end of a prompt most
    heavily, so the thing to act on belongs there, after the material.

    Returns
    -------
    str
    """
    viewing = index.upper() if index.upper() in STORES else DEFAULT_INDEX

    rules = f"""You answer questions about a study of Nifty implied volatility
around Indian market events.

RULES:
- Use ONLY the results table and the document passages below. Never use
  anything you remember about Indian markets.
- Quote numbers exactly as they appear. Do not calculate, round or estimate.
- Name your source: the event row, or the document and page.
- If the answer is not in what you were given, reply with exactly: {REFUSAL}
- Never give trading advice, predictions, or opinions about what to do.
- Write plain sentences. No markdown, no asterisks, no bold, no bullet
  characters, no headings. The page shows your reply as plain text, so any
  formatting marks appear literally on screen.
- Put sources in square brackets at the end of the sentence they support,
  like [results table, NIFTY, Union Budget 2025-26, 2025-02-01] or
  [2025-12-05_RBI.pdf, page 3]. Use ordinary square brackets, not any other
  bracket character.
- Every passage is labelled with the date of the document it came from. If
  the question is about a particular meeting, Budget or date, use ONLY
  passages whose document date matches it. Search always returns its best
  guesses, so passages from other dates will often be present -- they are
  not evidence about the date asked. If no passage carries the right date,
  reply with the refusal rather than describing a different event.
- The reader is looking at the {viewing} panel. The table holds both
  indices, labelled in its first column. Answer for {viewing} unless the
  question names the other one, and say which index your figures are for.
  Do not volunteer the other index's numbers."""

    if passages:
        shown = "\n\n".join(
            f"[doc {p['doc']}, page {p['page']}]\n{p['text']}" for p in passages)
    else:
        shown = "none found for this question"

    return (f"{rules}\n\n"
            f"RESULTS TABLE:\n{table}\n\n"
            f"DOCUMENT PASSAGES:\n{shown}\n\n"
            f"QUESTION:\n{question}")


# The ceiling on what the model may generate.
#
# 400 was far too low, and the reason is not obvious: a reasoning model spends
# tokens thinking before it writes anything, and that thinking counts against
# this number. Measured on "Will implied volatility fall at the next RBI
# meeting?", the reasoning alone came to 113, 181 and over 390 tokens on three
# runs of the same prompt at temperature 0 -- it is not deterministic. The
# visible answer is 26 characters. So at 400 the model occasionally used the
# whole budget thinking and had nothing left to say, returning an empty reply
# with finish_reason "length". Intermittently blank, which is the worst kind
# of broken.
#
# 1200 leaves room for the longest reasoning seen plus a full answer. The cost
# is not the tokens themselves -- billing is on what is actually generated --
# but some providers reserve this figure against a per-minute allowance, so it
# is set with headroom rather than generously.
MAX_OUTPUT_TOKENS = 1200


def call_llm(prompt):
    """Send the prompt to whichever OpenAI-compatible provider is configured.

    Three environment variables, so the project is not married to one vendor:
    LLM_API_KEY, and optionally LLM_BASE_URL and LLM_MODEL. OpenAI, Groq,
    Gemini and most others all speak this protocol.

    Temperature 0 asks for the most predictable answer the model can give.
    Creativity is worth nothing here and makes the evals flaky.

    Returns
    -------
    (answer, error, reason) : the reason names *whose* problem it was, which
        matters because the three cases need different responses. "provider"
        is the vendor refusing or timing out -- wait and retry. "empty" is
        the provider answering fine while the model emits no text, which is
        a fault in this pipeline and must not be filed under the vendor's
        name. "config" is a missing key or package. None on success.
    """
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None, ("The question box needs an LLM key. Set LLM_API_KEY and "
                      "restart the server."), "config"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "The openai package is not installed.", "config"

    # Accept-Encoding is pinned to gzip on purpose. The HTTP stack will
    # otherwise negotiate brotli, and an older brotli package than the HTTP
    # client expects fails while *unpacking the reply* -- the request
    # succeeds, the response arrives, and decompression raises
    # "TypeError: process() takes no keyword arguments". Gzip is universally
    # supported and the payloads here are a few kilobytes, so the compression
    # saved is irrelevant next to the fragility removed.
    # One retry, and a hard timeout. The SDK retries a rate limit by default,
    # which turns a fast, clear "you are out of tokens today" into a long
    # silence -- and on a page with a visitor waiting, a slow failure is worse
    # than a quick one. The caller already reports the error properly.
    client = OpenAI(api_key=key,
                    base_url=os.environ.get("LLM_BASE_URL") or None,
                    default_headers={"Accept-Encoding": "gzip, deflate"},
                    max_retries=1,
                    timeout=30.0)
    try:
        reply = client.chat.completions.create(
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as exc:
        # Report what went wrong, not just its type. "TypeError" alone says
        # nothing; "the model X does not exist" is a fix. The detail is
        # trimmed because it reaches a public page, and the full text goes to
        # the server log where it can be read in full.
        detail = " ".join(str(exc).split())
        print(f"LLM call failed: {type(exc).__name__}: {detail}", file=sys.stderr)
        return None, (f"The language model could not be reached: "
                      f"{type(exc).__name__} - {detail[:200]}"), "provider"

    # A reasoning model can return empty content with no error at all. Two
    # causes, and finish_reason tells them apart:
    #
    #   "length"  it spent the whole MAX_OUTPUT_TOKENS budget thinking and had
    #             nothing left to write. Raising the ceiling fixes it.
    #   "stop"    it finished, but left content blank.
    #
    # Either way a blank answer box reaches the page. Reported as "empty"
    # rather than "provider": the vendor answered perfectly well, and filing
    # this under their name sends whoever reads the log to check their quota
    # instead of this setting -- which is exactly what happened.
    choice = reply.choices[0]
    answer = (choice.message.content or "").strip()
    if not answer:
        print(f"LLM returned empty content "
              f"(finish_reason={choice.finish_reason}, "
              f"max_tokens={MAX_OUTPUT_TOKENS})", file=sys.stderr)
        hint = (" It ran out of room to answer; raise MAX_OUTPUT_TOKENS."
                if choice.finish_reason == "length" else "")
        return None, ("The language model returned an empty answer." + hint
                      + " Please try again."), "empty"

    return answer, None, None


# Characters the model reaches for occasionally whatever the prompt says.
# Asking nicely gets it right most of the time; replacing them afterwards
# gets it right every time. A rule the model follows 90% of the time is a
# rule that fails visibly on a public page once a day.
_TIDY = {
    "【": "[", "】": "]",      # CJK lenticular brackets
    "‘": "'", "’": "'",      # curly quotes
    "“": '"', "”": '"',
    "‑": "-", "–": "-",      # non-breaking and en dashes
    " ": " ", " ": " ",      # narrow and non-breaking spaces
}


def tidy(text):
    """Normalise the handful of characters that render badly or confuse greps."""
    for bad, good in _TIDY.items():
        text = text.replace(bad, good)
    return text


def answer_question(question, index=DEFAULT_INDEX):
    """Retrieve, prompt, call, and return the answer with its sources.

    Shared by the web route and run_evals.py, so the evals test the real
    thing rather than a copy that might drift from it.
    """
    try:
        from rag import searcher
        passages = searcher().search(question)
    except FileNotFoundError:
        passages = []
    except Exception as exc:
        print(f"retrieval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        passages = []

    prompt = build_prompt(question, results_table_text(), passages, index)
    answer, error, reason = call_llm(prompt)
    if error:
        # `reason` travels with the error so run_evals.py can tell a vendor
        # problem (wait and retry) from this pipeline returning nothing
        # (a real failure). The page ignores it and just shows the message.
        return {"answer": error, "sources": [], "available": False,
                "reason": reason}

    return {
        "answer": tidy(answer),
        "sources": [{"doc": p["doc"], "date": p["date"], "page": p["page"],
                     "score": round(p["score"], 2)} for p in passages],
        "available": True,
    }


@app.post("/api/ask", summary="Ask about the results")
def ask(payload: Question, request: Request):
    caller = request.client.host if request.client else "unknown"
    if rate_limited(caller):
        raise HTTPException(
            429, f"Too many questions. Please wait a minute "
                 f"({ASK_LIMIT_PER_MINUTE} per minute).")

    question = payload.question.strip()
    if not question:
        return {"answer": "Please type a question.", "sources": [],
                "available": False, "reason": "no_question"}
    return answer_question(question, payload.index)


@app.get("/", include_in_schema=False)
def page():
    return FileResponse(STATIC / "index.html")


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
