"""Test the question box against questions whose answers we already know.

    LLM_API_KEY=... python run_evals.py
    python run_evals.py --retrieval-only     no LLM key needed

Why this exists. Wiring a language model to some data takes twenty lines and
demos beautifully. Proving it answers correctly is the hard part, and it is
the part worth showing.

Four checks:

    numbers     ask something with one right answer from results.csv, and
                confirm that exact figure appears in the reply
    retrieval   ask something only one document can answer, and confirm that
                document is among the passages retrieved
    citations   confirm every document named in an answer was actually
                retrieved -- a model citing a source it never saw is making
                it up
    refusals    ask things the data cannot answer, and confirm the model
                says so instead of guessing

Prints a score and exits non-zero if anything fails, so GitHub Actions can
run it as a gate. It never runs on the live page -- it costs money per call.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA = BASE / "data"

REFUSAL_MARKER = "not in the results"

# How many refusal questions to ask in total. Six was too few to mean
# anything: refusing five of six could be 83% or could be luck, and there is
# no way to tell from six trials. The generator below makes up the rest.
REFUSAL_TARGET = 40

# Dates nearer than this to a real event are not used to invent a fake one.
# A question about a day three days before a genuine RBI meeting is arguably
# answerable, and a test whose correct answer is debatable is a bad test.
FAKE_EVENT_GAP = 20

SEED = 20260101

# The six that cannot be generated, because they are defined by judgement
# rather than by a gap in the data. Three distinct failure modes:
#
#   an event that never happened    the model invents a plausible number
#   general knowledge               it answers from memory instead of the data
#   advice and prediction           it says something nobody should act on
#
# A generator can produce the first kind endlessly. It cannot produce the
# other two, which is why these stay hand-written.
#
# Each carries the SHAPE of refusal that counts as correct, because there are
# two right answers here and this suite used to accept only one:
#
#   "missing"  the figure genuinely is not in the data, so the model must say
#              the exact refusal string. That exact wording is the contract
#              build_prompt and this file share -- a model refusing in fresh
#              words every time cannot be checked automatically, and an AI
#              feature nobody can test is one nobody should ship.
#
#   "decline"  the question asks for advice or a forecast. The prompt tells
#              the model never to give either, so refusing in its own words
#              is obeying that instruction, not dodging the question. The
#              exact string is accepted too. Demanding only the string marked
#              "I'm sorry, but I can't help with that" as a failure, which
#              reported the model as broken for doing precisely as it was
#              told.
HANDWRITTEN_UNANSWERABLE = [
    ("What was the implied volatility before the March 2025 RBI meeting?", "missing"),
    ("What did the RBI decide in its September 2019 policy meeting?", "missing"),
    ("Who is the current Finance Minister of India?", "missing"),
    ("What is the capital of France?", "missing"),
    ("Should I buy Nifty options before the next Budget?", "decline"),
    ("Will implied volatility fall at the next RBI meeting?", "decline"),
]

# Ways a model says "I won't answer that" without using our sentence. Only
# consulted for "decline" traps, so a model that merely errored out or
# wandered off cannot pass a factual question by sounding apologetic.
DECLINE_MARKERS = (
    "can't help", "cannot help", "can not help",
    "can't assist", "cannot assist",
    "can't provide", "cannot provide",
    "can't answer", "cannot answer",
    "can't give", "cannot give",
    "unable to", "not able to",
    "won't provide", "will not provide",
    "don't provide", "do not provide",
    "not in a position to",
)


def refused(text, kind, refusal):
    """Did the model decline, in a way this particular trap accepts?"""
    low = text.lower()
    if REFUSAL_MARKER in low or text.strip() == refusal:
        return True
    return kind == "decline" and any(m in low for m in DECLINE_MARKERS)

# Each takes a real trading day on which nothing happened, and asks about it
# as though something had. The date is plausible, the market was open, and
# the premise is false -- which is exactly the shape of question a model is
# most likely to answer confidently and wrongly.
FAKE_EVENT_TEMPLATES = [
    "What was the implied volatility before the RBI meeting on {date}?",
    "How much did implied volatility fall after the RBI decision on {date}?",
    "What did the RBI announce on {date}?",
    "What was the implied volatility before the Budget on {date}?",
    "What realised volatility followed the event on {date}?",
    "How did Nifty implied volatility react to the policy announcement on {date}?",
]


def generated_unanswerable(events, how_many, seed=SEED, index="NIFTY"):
    """Questions about events that never happened, on days that really existed.

    The number questions in this file are generated from results.csv, so they
    grow by themselves as events are added. Refusals had no equivalent, which
    left the most important check in the suite resting on six examples.

    This closes that. Every trading day at least FAKE_EVENT_GAP days from any
    real event is a candidate: the market was open, the date is real, and
    nothing happened. Asking "what was IV before the RBI meeting on
    14 March 2025" presupposes a meeting that does not exist, which is the
    single failure mode most likely to put a confident wrong number on a
    public page.

    Seeded, so a failing run can be reproduced exactly rather than vanishing
    on the retry.

    The calendar comes from the index being tested. It used to be hardcoded to
    Nifty's, which produced the right questions by accident -- both indices
    trade the same 969 days -- while quietly testing one index's refusals
    against another index's dates.
    """
    folder = INDEX_DIRS.get(index, DATA)
    daily = pd.read_csv(folder / "daily_iv.csv", parse_dates=["date"])
    event_days = set(events["event_date"].dt.date)

    candidates = []
    for day in daily["date"].dt.date:
        if all(abs((day - ev).days) > FAKE_EVENT_GAP for ev in event_days):
            candidates.append(day)
    if not candidates:
        return []

    rng = np.random.default_rng(seed)
    picks = rng.choice(len(candidates), size=min(how_many, len(candidates)),
                       replace=False)
    out = []
    for i, idx in enumerate(sorted(picks)):
        day = candidates[idx]
        template = FAKE_EVENT_TEMPLATES[i % len(FAKE_EVENT_TEMPLATES)]
        # Every generated trap is "missing": it asks for a figure about an
        # event that did not happen, so the exact refusal string is the only
        # correct answer.
        out.append((template.format(date=day.strftime("%d %B %Y")), "missing"))
    return out


def unanswerable_cases(events, target=REFUSAL_TARGET, index="NIFTY"):
    """The hand-written traps first, then as many generated ones as needed."""
    questions = list(HANDWRITTEN_UNANSWERABLE)
    questions += generated_unanswerable(
        events, max(0, target - len(questions)), index=index)
    return questions


# Each index's results live in their own folder, matching build.py. Until
# this existed the suite only ever tested Nifty, so Bank Nifty's numbers --
# half of what the question box can be asked about -- were never checked by a
# single question.
INDEX_DIRS = {"NIFTY": DATA, "BANKNIFTY": DATA / "banknifty"}


# Signatures of "the provider will refuse everything else too". A daily token
# cap does not clear in the next few seconds, so asking the remaining
# questions wastes time and fills the screen with the same message. A
# per-minute limit is different -- that one is worth waiting out.
HARD_STOP = ("tokens per day", "requests per day", "tpd)", "rpd)",
             "insufficient_quota", "invalid_api_key", "authentication")


def provider_exhausted(text):
    low = text.lower()
    return any(marker in low for marker in HARD_STOP)

def load_events(index="NIFTY"):
    folder = INDEX_DIRS[index]
    path = folder / "results.csv"
    if not path.exists():
        sys.exit(f"{path} is missing -- run build.py --symbol {index} first")
    frame = pd.read_csv(path, parse_dates=["event_date"])
    return frame.sort_values("event_date")


def number_cases(events, per_event=3):
    """Three questions per event, each with one correct figure."""
    cases = []
    for row in events.itertuples():
        when = row.event_date.strftime("%d %B %Y")
        cases.append((f"What was the implied volatility the day before the "
                      f"event on {when}?", row.iv_before))
        if per_event > 1:
            cases.append((f"What was the implied volatility on the event day "
                          f"of {when}?", row.iv_after))
        if per_event > 2:
            cases.append((f"What realised volatility followed the event on "
                          f"{when}?", row.realised_vol))
    return cases


def mentions_number(text, value, tolerance=0.02):
    """Is `value` quoted in `text`?

    Accepts any number within `tolerance` so that 10.02 still passes when the
    model writes 10.0. It must not be looser than that: the whole point is
    that the figure was read off the table rather than approximated.
    """
    for found in re.findall(r"-?\d+\.?\d*", text.replace(",", "")):
        try:
            if abs(float(found) - float(value)) <= tolerance:
                return True
        except ValueError:
            continue
    return False


def run_retrieval_checks(events):
    """Does the search find the right document? No LLM needed."""
    from rag import searcher

    try:
        search = searcher()
    except FileNotFoundError as exc:
        print(f"  skipped: {exc}")
        return 0, 0, []

    docs = {p.name for p in (BASE / "docs").glob("*.pdf")}
    passed = total = 0
    failures = []

    for row in events.itertuples():
        expected = f"{row.event_date.date()}_{row.type}.pdf"
        if expected not in docs:
            continue                      # that document has not been added yet
        total += 1
        when = row.event_date.strftime("%B %Y")
        kind = "RBI monetary policy" if row.type == "RBI" else "Union Budget"
        hits = search.search(f"What did the {kind} announce in {when}?")
        if any(h["doc"] == expected for h in hits):
            passed += 1
        else:
            got = ", ".join(sorted({h["doc"] for h in hits})) or "nothing"
            failures.append(f"    {expected}: not retrieved (got {got})")

    return passed, total, failures


def run_llm_checks(events, limit, refusal_target=REFUSAL_TARGET,
                   index="NIFTY"):
    """Numbers, citations and refusals. Needs a key."""
    from main import REFUSAL, answer_question

    results = {"numbers": [0, 0], "citations": [0, 0], "refusals": [0, 0]}
    failures = []
    stopped = None
    # Questions the provider refused to accept -- a rate limit or an outage.
    # Kept apart from failures because they measure the vendor, not the model.
    untested = []

    cases = number_cases(events)
    if limit:
        cases = cases[:limit]

    print(f"    asking {len(cases)} number questions", flush=True)
    for i, (question, expected) in enumerate(cases, 1):
        if pd.isna(expected):
            continue
        # The index is passed through, so a Nifty question is scored
        # against Nifty's figures and the model is told which panel the
        # reader is on. Without it every answer would be judged against
        # Nifty whatever was asked.
        reply = answer_question(question, index)
        text = reply["answer"]
        # An API error is not a wrong answer -- see the refusal loop below.
        if not reply.get("available", True):
            # A blank answer is OUR failure, not the vendor's. The provider
            # replied; the model emitted no text, and a visitor would have
            # seen an empty box. Filing it under "the provider refused the
            # call" sent a real debugging session off to check the quota.
            if reply.get("reason") == "empty":
                results["numbers"][1] += 1
                failures.append(f"    blank answer (no text generated): "
                                f"{question[:60]}\n      {text[:110]}")
                continue
            untested.append(f"    not tested: {question[:70]}")
            if provider_exhausted(text):
                stopped = text
                untested.extend(f"    not asked: {q[:70]}"
                                for q, _ in cases[i:])
                break
            if i % 5 == 0 or i == len(cases):
                print(f"      {i}/{len(cases)}  correct "
                      f"{results['numbers'][0]}/{results['numbers'][1]}"
                      f"  ({len(untested)} unasked)", flush=True)
            continue

        results["numbers"][1] += 1

        if mentions_number(text, expected):
            results["numbers"][0] += 1
        else:
            failures.append(f"    expected {expected:.2f} in: {text[:110]}")

        # Citations: every document the answer names must be one that was
        # actually retrieved. A plausible-looking filename the model never
        # saw is a fabrication, and the worst kind -- it looks like evidence.
        retrieved = {s["doc"] for s in reply["sources"]}
        cited = set(re.findall(r"\d{4}-\d{2}-\d{2}_[A-Z]+\.pdf", text))
        results["citations"][1] += 1
        if cited <= retrieved:
            results["citations"][0] += 1
        else:
            failures.append(f"    cited but never retrieved: {cited - retrieved}")

        # Progress, because fifty silent calls look exactly like a hang.
        if i % 5 == 0 or i == len(cases):
            print(f"      {i}/{len(cases)}  correct "
                  f"{results['numbers'][0]}/{results['numbers'][1]}"
                  + (f"  ({len(untested)} unasked)" if untested else ""),
                  flush=True)

    traps = unanswerable_cases(events, refusal_target, index)
    if stopped:
        untested.extend(f"    not asked: {q[:70]}" for q, _ in traps)
        return results, failures, untested, stopped
    print(f"    asking {len(traps)} refusal questions", flush=True)
    for j, (question, kind) in enumerate(traps, 1):
        reply = answer_question(question, index)
        text = reply["answer"]

        # The important distinction. A rate limit is not the model inventing
        # an answer, and scoring it as one makes the suite claim a failure
        # that never happened. The run still exits non-zero, so a half-
        # measured eval cannot quietly pass.
        if not reply.get("available", True):
            if reply.get("reason") == "empty":
                results["refusals"][1] += 1
                failures.append(f"    blank answer (no text generated): "
                                f"{question[:60]}\n      {text[:110]}")
                continue
            untested.append(f"    not tested: {question[:70]}")
            if provider_exhausted(text):
                stopped = text
                untested.extend(f"    not asked: {q[:70]}"
                                for q, _ in traps[j:])
                break
            continue

        results["refusals"][1] += 1
        if refused(text, kind, REFUSAL):
            results["refusals"][0] += 1
        else:
            failures.append(f"    should have refused ({kind}) {question!r}\n"
                            f"      got: {text[:110]}")

        if j % 5 == 0 or j == len(traps):
            print(f"      {j}/{len(traps)}  refused "
                  f"{results['refusals'][0]}/{results['refusals'][1]}",
                  flush=True)

    return results, failures, untested, stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true",
                        help="skip everything that needs an LLM key")
    parser.add_argument("--index", default="NIFTY",
                        choices=sorted(INDEX_DIRS),
                        help="which index's numbers to test (default NIFTY)")
    parser.add_argument("--refusals", type=int, default=REFUSAL_TARGET,
                        help=f"how many refusal questions (default "
                             f"{REFUSAL_TARGET})")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of number-questions, to save credits")
    args = parser.parse_args()

    events = load_events(args.index)
    print(f"{args.index}: evals over {len(events)} events\n")

    print("retrieval")
    r_pass, r_total, r_fail = run_retrieval_checks(events)
    if r_total:
        print(f"  {r_pass}/{r_total} questions found the right document")
    for line in r_fail:
        print(line)

    failed = len(r_fail)
    untested = []
    checks = [("retrieval", r_pass, r_total)]

    if not args.retrieval_only:
        print("\nnumbers, citations and refusals")
        results, llm_fail, untested, stopped = run_llm_checks(
            events, args.limit, args.refusals, args.index)
        if stopped:
            print()
            print("  STOPPED EARLY -- the provider is out of quota:")
            print(f"    {' '.join(stopped.split())[:170]}")
        for name, (passed, total) in results.items():
            print(f"  {name:10} {passed}/{total}")
            checks.append((name, passed, total))
        for line in llm_fail:
            print(line)
        failed += len(llm_fail)

        if untested:
            print(f"\n  {len(untested)} question(s) could not be asked. The "
                  f"provider refused the call -- usually a daily rate limit.")
            print("  These are NOT model failures: the model was never asked. "
                  "Re-run later to measure them.")
            for line in untested[:5]:
                print(line)
            if len(untested) > 5:
                print(f"    ... and {len(untested) - 5} more")
            # Still a non-zero exit. An eval that measured only part of the
            # suite must not report success.
            failed += len(untested)

    total_pass = sum(p for _, p, _ in checks)
    total_all = sum(t for _, _, t in checks)
    print(f"\n  score {total_pass}/{total_all}")

    # Real failures and unasked questions are counted apart. Both make the
    # run exit non-zero -- an eval that measured half the suite has not
    # passed -- but calling a rate limit a "failure" misreports the model.
    n_untested = len(untested) if not args.retrieval_only else 0
    real = failed - n_untested

    if real or n_untested:
        if real:
            print(f"  {real} failure(s)")
        if n_untested:
            print(f"  {n_untested} question(s) not asked (provider limit) -- "
                  f"re-run later to measure them")
        sys.exit(1)
    print("  all checks passed")


if __name__ == "__main__":
    main()
