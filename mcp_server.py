"""An MCP server over the study, so an AI assistant can query it directly.

    python mcp_server.py

MCP -- Model Context Protocol -- is a standard way for a language model to
call tools someone else wrote. Claude, or any MCP client, connects to this
process over stdin/stdout and can then ask for the numbers rather than being
handed a prompt and told to read.

Why this is worth having when /api/ask already exists.

The web question box is one fixed pipeline: retrieve five passages, build one
prompt, call one model, return one answer. It works, and the evals measure it,
but the shape of the conversation is decided in advance by build_prompt.

An MCP server inverts that. The model decides what to look up and when, and
can take several steps -- read the RBI summary, notice the crush is small,
pull the backtest to see whether it was tradeable, then search the April 2026
statement for why that meeting moved the index so much. Three tool calls the
web route could never make, because nobody wrote that sequence down.

The important property is the same one the web route has: every number comes
from the CSVs that build.py wrote, and every passage comes from a real PDF.
The model chooses what to ask for. It never computes anything.

Configure it in Claude Desktop (claude_desktop_config.json):

    {
      "mcpServers": {
        "nifty-iv": {
          "command": "python",
          "args": ["F:/projects/nifty/mcp_server.py"]
        }
      }
    }
"""

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# The numbers come from main.py's Stores rather than being re-read here.
#
# Two implementations of "what does summary.csv mean" would eventually
# disagree, and the first anyone would notice is an MCP answer that does not
# match the web page. Importing main gives this server the same objects the
# API serves, so they cannot drift.
#
# Importing main also builds the FastAPI app, which costs nothing when nobody
# serves it.
# ---------------------------------------------------------------------------
from main import DEFAULT_INDEX, STORES, records

mcp = FastMCP(
    "nifty-iv",
    instructions=(
        "Implied volatility around Indian market events: RBI policy meetings, "
        "Union Budgets and the 2024 election result, for the Nifty and Bank "
        "Nifty indices, from NSE end-of-day option data.\n\n"
        "Every number these tools return was computed offline by tested code. "
        "Quote them exactly and never recalculate them. If a tool returns no "
        "data for something, say so rather than filling the gap from memory -- "
        "there was no RBI meeting in March 2025, for instance, and no tool "
        "here will pretend otherwise.\n\n"
        "Read get_study_overview first if you have not already: it explains "
        "what the numbers mean and which conclusions the data supports."
    ),
)


def _store(index: str):
    key = (index or DEFAULT_INDEX).upper()
    if key not in STORES:
        raise ValueError(
            f"unknown index {index!r}; available: {sorted(STORES)}")
    return STORES[key]


def _json(payload: Any) -> str:
    """Tools return text, so everything goes out as readable JSON."""
    return json.dumps(payload, indent=1, default=str)


@mcp.tool()
def get_study_overview() -> str:
    """What this study measured, how, and what it does and does not show.

    Read this before the other tools. It is the context that stops a correct
    number being reported as the wrong conclusion.
    """
    return _json({
        "question": (
            "Does the options market overcharge for protection before known "
            "Indian market events, and how fast does that price collapse once "
            "the news lands?"
        ),
        "method": (
            "NSE end-of-day option files for every trading day. For each day, "
            "the nearest expiry at least 7 calendar days out, then the strike "
            "closest to spot with a traded call AND a traded put. Black-"
            "Scholes solved backwards with Brent's method for implied "
            "volatility. Prices outside the no-arbitrage bounds are rejected."
        ),
        "indexes": sorted(STORES),
        "coverage": {
            k: {"first": str(s.first), "latest": str(s.latest),
                "trading_days": s.trading_days, "events": len(s.results)}
            for k, s in STORES.items()
        },
        "validation": (
            "The Nifty series correlates 0.9157 with India VIX, which NSE "
            "publishes independently using a different method. There is no "
            "Bank Nifty VIX, so that index has no equivalent independent "
            "check -- it is validated only by tracking Nifty at 0.93 while "
            "sitting about 19% higher, which is what a single-sector index "
            "should do."
        ),
        "what_it_shows": [
            "Options were overpriced before 23 of 29 events, measured as "
            "implied volatility before the event against the volatility the "
            "index actually delivered over the following 10 trading days.",
            "The RBI result is strong: a placebo test puts the average crush "
            "at the 0.00 percentile of 10,000 random groups of 23 ordinary "
            "days, and the 95% confidence range is entirely below zero.",
            "The crush does not need a run-up. Before Budgets, volatility "
            "rises about 45% then falls 15.5%. Before RBI meetings it falls "
            "3.2% and still gets crushed 6.5% on the day.",
            "The effect is the same size in absolute volatility points in both "
            "indices (about 3.5 points of overpricing), not proportionally "
            "larger in the rate-sensitive banking index. It is market-wide, "
            "not about sector exposure.",
        ],
        "what_it_does_not_show": [
            "That the effect is tradeable. The straddle backtest found 0.12 "
            "points of gross profit before RBI meetings in Nifty and a loss "
            "in Bank Nifty, both worse than selling on an ordinary day, "
            "because event days also move more. Use get_backtest before "
            "saying anything about profitability.",
            "Anything reliable about Budgets. There are only 5, and the "
            "confidence range on the Budget crush nearly touches zero in "
            "Nifty and crosses it in Bank Nifty.",
            "Anything about a single future event. The averages hold across "
            "23 meetings; both indices fell together on only 12 of them.",
        ],
        "known_limits": [
            "A flat 6.5% interest rate across the whole period.",
            "One at-the-money strike per day, not a volatility surface.",
            "End-of-day closes, so no intraday behaviour at all.",
            "Four days excluded as too thinly traded: one ceremonial session "
            "and three disaster-recovery drills.",
        ],
    })


@mcp.tool()
def get_summary(event_type: str, index: str = DEFAULT_INDEX) -> str:
    """Average result for one event type, with its confidence range and placebo test.

    Args:
        event_type: RBI, BUDGET or ELECTION.
        index: NIFTY or BANKNIFTY.
    """
    store = _store(index)
    kind = event_type.upper()
    if kind not in store.types:
        raise ValueError(f"unknown event type {event_type!r}; "
                         f"available: {store.types}")

    summary = records(store.summary[store.summary["type"] == kind])[0]
    placebo = store.placebo[store.placebo["type"] == kind]
    return _json({
        "index": store.index,
        "event_type": kind,
        "summary": summary,
        "placebo": records(placebo)[0] if len(placebo) else None,
        "reading_the_numbers": {
            "avg_change_pct": "average % change in implied volatility from "
                              "the day before the event to the event-day close",
            "change_ci_low/high": "95% bootstrap range on that average; a "
                                  "range crossing zero means it is not "
                                  "established",
            "overpriced": "events where implied volatility beforehand exceeded "
                          "the volatility actually delivered afterwards",
            "percentile_vs_normal": "where this average ranks among 10,000 "
                                    "random groups of the same size drawn "
                                    "from ordinary trading days; 0.00 means "
                                    "none of them were as extreme",
        },
    })


@mcp.tool()
def get_events(event_type: str = "", index: str = DEFAULT_INDEX,
               limit: int = 40) -> str:
    """The individual events, one row each, newest first.

    Args:
        event_type: RBI, BUDGET or ELECTION. Empty for all of them.
        index: NIFTY or BANKNIFTY.
        limit: maximum rows to return.
    """
    store = _store(index)
    frame = store.results
    if event_type:
        kind = event_type.upper()
        if kind not in store.types:
            raise ValueError(f"unknown event type {event_type!r}; "
                             f"available: {store.types}")
        frame = frame[frame["type"] == kind]

    frame = frame.sort_values("event_date", ascending=False).head(max(1, limit))
    return _json({
        "index": store.index,
        "n": len(frame),
        "events": records(frame),
    })


@mcp.tool()
def get_backtest(index: str = DEFAULT_INDEX) -> str:
    """Would selling the straddle before events have made money? Usually not.

    One fixed rule: sell the at-the-money straddle at the close the day before
    the event, buy it back at the close on the event day. Profit in index
    points after an assumed cost.

    Read the control row before the event rows. It is the same rule applied to
    ordinary days, and an event group that does not beat it has no edge
    whatever its average looks like.

    Args:
        index: NIFTY or BANKNIFTY.
    """
    store = _store(index)
    if store.backtest is None:
        return _json({"index": store.index,
                      "error": "no backtest has been run for this index"})

    rows = records(store.backtest)
    return _json({
        "index": store.index,
        "groups": [r for r in rows if r["group"] != "NORMAL"],
        "control_ordinary_days": next(
            (r for r in rows if r["group"] == "NORMAL"), None),
        "reading_the_numbers": {
            "avg_net": "average profit per trade in index points, after costs",
            "avg_gross": "profit before any costs -- which is also exactly the "
                         "round-trip cost at which the rule breaks even",
            "win_rate": "percentage of trades that profited. Read it with "
                        "`worst`: this is a trade that wins often and loses "
                        "big, so a high win rate alone proves nothing",
            "worst": "largest single loss, the number that decides whether the "
                     "rule is usable at all",
        },
    })


@mcp.tool()
def get_daily_iv(start: str, end: str, index: str = DEFAULT_INDEX) -> str:
    """The daily implied volatility series over a date range.

    Capped at 400 days per call, because a longer range is rarely what a
    question needs and a thousand rows of JSON crowds out everything else.

    Args:
        start: YYYY-MM-DD.
        end: YYYY-MM-DD.
        index: NIFTY or BANKNIFTY.
    """
    import pandas as pd

    store = _store(index)
    frame = store.daily
    frame = frame[(frame["date"] >= pd.Timestamp(start))
                  & (frame["date"] <= pd.Timestamp(end))]
    truncated = len(frame) > 400
    frame = frame.head(400)[["date", "spot", "atm_iv", "next_iv", "vix", "skew"]]

    return _json({
        "index": store.index,
        "n": len(frame),
        "truncated": truncated,
        "vix_is_native": store.index == "NIFTY",
        "points": records(frame),
    })


@mcp.tool()
def search_documents(question: str, k: int = 5) -> str:
    """Search the RBI statements and Budget speeches for relevant passages.

    Keyword search (BM25) over passages cut from the official PDFs. Returns
    the text with its document and page number, so anything quoted can be
    cited and checked.

    Args:
        question: what to search for.
        k: how many passages to return.
    """
    try:
        from rag import searcher
        hits = searcher().search(question, k=max(1, min(k, 10)))
    except FileNotFoundError:
        return _json({"error": "no document index -- run `python rag.py` after "
                               "adding PDFs to docs/", "passages": []})

    return _json({
        "question": question,
        "n": len(hits),
        "passages": [{"doc": h["doc"], "date": h["date"], "page": h["page"],
                      "score": round(h["score"], 2), "text": h["text"]}
                     for h in hits],
    })


if __name__ == "__main__":
    mcp.run()
