
# Nifty IV Event Study
Live link: https://nifty-iv-event-study.onrender.com/


**What the Indian options market expects before major events, against what actually happens.**

Four years of NSE end-of-day option data, solved for implied volatility, lined
up around 29 scheduled events — RBI policy meetings, Union Budgets and the 2024
election result — and tested three ways: is the effect real, does it predict
direction, and could you have traded it.

<!-- AUTO:COVERAGE -->
```
977 trading days   1 Dec 2022 – 13 Nov 2026
29 events          23 RBI · 5 Budget · 1 election
2 indices          Nifty 50 and Bank Nifty
```

*Figures throughout are as of 13 Nov 2026, the last trading day in the data.
They are regenerated from the result files by `update_readme.py`, and CI fails
if they drift — so this page and the live site cannot disagree.*
<!-- /AUTO:COVERAGE -->

---

## The findings

### 1. Options are systematically overpriced before events

<!-- AUTO:FINDINGS -->
| Event | n | IV charged before | Volatility delivered | Overpriced in |
|---|---:|---:|---:|---:|
| RBI policy | 23 | 13.35% | 9.87% | **19 of 23** |
| Union Budget | 5 | 17.98% | 13.34% | 3 of 5 |
| Election result | 1 | 26.92% | 17.66% | 1 of 1 |

Before RBI meetings the market priced in 13.35% of volatility and got 9.87%.
On the day itself implied volatility fell **5.62%** on average.
<!-- /AUTO:FINDINGS -->

### 2. It is not chance

<!-- AUTO:PLACEBO -->
A placebo test takes 458 ordinary trading days — at least ten days from any
event — draws 23 of them at random, averages, and repeats 10,000 times.

```
RBI        0.02 percentile      none of 10,000 random groups were as extreme
Budget     0.00 percentile
Election  98.82 percentile      went the other way, as expected
```

The 95% confidence range on the RBI crush is **−8.88% to −2.70%** — entirely
below zero.
<!-- /AUTO:PLACEBO -->

### 3. Direction is not predictable

<!-- AUTO:DIRECTION -->
| | up | down | % up | coin-flip p |
|---|---:|---:|---:|---:|
| RBI days | 11 | 12 | 47.8% | 1.000 |
| ordinary days | 252 | 205 | 55.1% | 0.031 |

Ordinary days rise 55.1% of the time — the slow upward drift of equities. RBI
days rise 47.8%. **Knowing a meeting is tomorrow leaves you with less
directional information than an ordinary Tuesday.** The decision is in the
price before it is announced.
<!-- /AUTO:DIRECTION -->

### 4. And you could not have traded it

One rule, fixed in advance: sell the at-the-money straddle at the close the day
before, buy it back at the close on the event day. Profit in index points.

<!-- AUTO:BACKTEST -->
| Group | Trades | Avg net | Won | Worst loss | Break-even cost |
|---|---:|---:|---:|---:|---:|
| RBI | 23 | −3.02 | 56.5% | −181.93 | **0.12** |
| Budget | 5 | +85.63 | 80.0% | −114.50 | 89.10 |
| Election | 1 | −722.36 | 0.0% | −722.36 | −716.80 |
| **ordinary days** | 458 | +1.81 | 62.4% | −429.00 | **4.89** |

"Break-even cost" is profit before any fees — which is also exactly the
round-trip cost at which the rule stops paying.

**Selling before RBI meetings earned 0.12 points before costs. Selling on a
random Tuesday earned 4.89.** The event trade is worse than trading at random,
because event days also *move* more, and the extra movement costs the seller
more than the collapsing volatility pays.

The worst single trade of all 487 was the election result: 4 Jun 2024, −722
points. The worst ordinary day lost −429 — a deeper hole than any RBI meeting
produced, on a day when nothing was scheduled at all.
<!-- /AUTO:BACKTEST -->

### 5. What volatility *does* tell you

<!-- AUTO:VOLDIR -->
| | correlation | days |
|---|---:|---:|
| IV change vs **same-day** return | **−0.5006** | 976 |
| IV change vs **next-day** return | +0.0318 | 975 |

On days volatility rose, the market fell **62%** of the time. On days it fell,
the market rose **66%** of the time.

But after the largest 10% of volatility spikes, the next day rose 54.1% —
against a 53.0% baseline. **Volatility measures today's direction accurately
and forecasts tomorrow's not at all.** Fear rising and prices falling are the
same event seen two ways, not a cause and its effect.
<!-- /AUTO:VOLDIR -->

---

## Is it right?

The strongest evidence is external. **India VIX** is published by NSE using a
different method — many strikes over a fixed 30 days, where this uses one
near-dated strike.

<!-- AUTO:VIX -->
```
our ATM IV       12.77
India VIX        13.83
correlation      0.9159   over 968 overlapping days
```

The ~1 point gap is expected and, importantly, **steady**. A broken solver
would drift — fine in calm markets, wrong in volatile ones. A constant offset
between two methods measuring slightly different things is what correctness
looks like.

**Bank Nifty corroborates independently.** Run through the same pipeline it
gives 15.04% mean IV against Nifty's 12.77% — higher, as a single-sector index
should be — and reaches the same conclusions: overpriced 21 of 23, placebo
0.07, and a straddle trade that *loses* 19.1 points before costs.

The crush is also the same size in **absolute** volatility points in both
indices (Nifty −0.87 against Bank Nifty −0.81), not proportionally larger in
the rate-sensitive banking index. The event premium is a roughly fixed
quantity of volatility, not a percentage markup.
<!-- /AUTO:VIX -->

**A C++ port of the solver** agrees with the Python to 3.3e-08, and rejects
exactly the same 164 unsolvable contracts out of 1,026 in a real session.
Agreeing on which questions have *no* answer is a stronger statement than
agreeing on the answers.

---

## Honest limits

- **End-of-day closes only.** No intraday behaviour whatsoever.
- **A flat 6.5% interest rate** across the whole period.
- **One at-the-money strike per day**, not a volatility surface.
- **Five Budgets.** Any Budget result is a hypothesis for the next five. A
  two-sided sign test on five events cannot go below p = 0.063 however the
  data falls.
- **Three event types are tested**, so one looking unusual at the 5% level is
  roughly what chance produces anyway.
- **Four sessions excluded** as too thinly traded: one ceremonial session and
  three disaster-recovery drills, named in `build.py`.
- **The averages say nothing about one future event.** Both indices fell
  together on only 12 of the 23 RBI meetings.
- **The phenomenon is not new.** Implied volatility falling after scheduled
  events is well documented. The contribution here is the measurement on
  Indian data, validated externally and shown to be untradeable.

### Two findings were corrected downward during the work

- **"Overpriced before 22 of 23 RBI meetings" became 11 of 23** when a horizon
  mismatch was fixed — the original compared a to-expiry expected move against
  a one-day actual move. The headline figure above (19 of 23) uses realised
  volatility, which was always measured correctly.
- **The placebo test compared an average of 23 events against single ordinary
  days.** Averages vary far less than single days, so the reference was about
  five times too wide and the effect was being *hidden*. Corrected, the RBI
  result moved from the 40th percentile to the 0.04th.

Both corrections are documented in the code.

---

## Running it

```bash
pip install -r requirements.txt

python download.py 2022-12-01        # ~971 NSE files, 86 MB after trimming
python build.py                      # daily IV, event study, placebo
python build.py --symbol BANKNIFTY
python backtest.py                   # the straddle rule, both indices
python direction.py                  # the direction tests
python -m uvicorn main:app --reload  # http://127.0.0.1:8000
```

Keeping it current:

```bash
python update.py        # fetches new days, rebuilds, reports what moved
```

`update.py` is the only script that knows the order the others run in.

### The question box

A retrieval-augmented question box over the results and the official RBI and
Budget PDFs. Set a key in `.env` (gitignored):

```
LLM_API_KEY=...
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Any OpenAI-compatible provider works — the vendor is three environment
variables, not a code dependency.

**The model never calculates.** It is handed the finished tables and told to
quote from them, refuse anything absent, and never give advice. Every figure
on the page was computed offline by code with tests.

```bash
python run_evals.py --index NIFTY     --limit 9 --refusals 6
python run_evals.py --index BANKNIFTY --limit 9 --refusals 6
```

Four checks: **numbers** (is the exact figure quoted?), **citations** (was every
document named actually retrieved?), **refusals** (unanswerable questions,
including real dates on which no event happened), and **retrieval**.

> **Eval status:** `openai/gpt-oss-120b`, both indices, 4 Nov 2026 —
> numbers **15/15**, citations **15/15**, refusals **12/12**, retrieval
> **28/28** per index. No failures.
>
> Retrieval covers all 28 documents, which is what makes it worth quoting:
> picking the right meeting out of 23 near-identically worded RBI
> resolutions is a real test, where 1/1 against a single PDF was not. The
> other three checks were measured while one document was indexed.
>
> Three questions went unasked, against the free tier's 8,000 tokens/minute;
> they are reported separately from failures, because a rate limit measures
> the vendor rather than the model, and scoring it as a wrong answer would
> claim a failure that never happened.
>
> Eval results belong to the model that produced them, not to this repository.
> `gpt-oss-20b` failed the advice question deterministically, returning empty
> content three times out of three.

Two scoring bugs found by running this suite, both in the harness rather than
the model, and both worth stating because they are the kind that flatter or
libel a model silently:

- **The advice refusal was marked wrong.** The prompt forbids trading advice
  *and* demands an exact refusal string for absent data. Asked "should I buy
  Nifty options?", the model declined in its own words — obeying the first
  rule — and the scorer, checking only for the string, recorded a failure.
  Each trap now carries the *shape* of refusal that counts as correct, and an
  apologetic non-answer still fails a factual question.
- **An empty reply was blamed on the provider.** `max_tokens` was 400, and a
  reasoning model's thinking counts against it — measured at 113, 181 and
  over 390 tokens for the same prompt at temperature 0. When it overran, the
  answer came back blank and the harness reported "the provider refused the
  call", sending a debugging session to the wrong place entirely. A blank is
  now counted as this pipeline's failure, which it is.

---

## What is here

<!-- AUTO:FILES -->
| File | Lines | What it does |
|---|---:|---|
| `download.py` | 312 | Fetches NSE bhavcopies and index closes |
| `iv.py` | 149 | Black-Scholes and the Brent solve for IV |
| `build.py` | 701 | Raw files → daily IV, event study, placebo |
| `analysis.py` | 263 | Bootstrap intervals, placebo test, expected move |
| `backtest.py` | 538 | The straddle rule, events against ordinary days |
| `direction.py` | 363 | Sign tests, and volatility against direction |
| `update.py` | 291 | Brings everything up to the last trading day |
| `update_readme.py` | 467 | Regenerates this file's figures, and checks them in CI |
| `main.py` | 738 | FastAPI: the API and the question box |
| `rag.py` | 296 | PDF passages and BM25 retrieval |
| `run_evals.py` | 471 | The four eval checks |
| `cpp/iv.cpp` | 224 | The solver again, in C++, for verification |
| `bench.py` | 348 | Accuracy and speed, Python against C++ |
| `tests/` | 442 | 80 tests on the maths and the statistics |

```bash
pytest          # 80 tests
python bench.py # C++ against Python: agreement and speed
```
<!-- /AUTO:FILES -->

---

## Design decisions worth knowing

**Everything is computed offline.** The browser downloads finished numbers and
draws them; it never calculates one. A page that computes its own figures has
a second place for a number to be wrong, with no test covering it.

**Prices outside the no-arbitrage bounds are rejected** before the solver sees
them. A stale close below intrinsic value produces a confident wrong
volatility rather than an error — this is the most common reason a real NSE
close is unusable.

**Both a traded call and a traded put are required** at the chosen strike.
Skipping a day is better than recording a number nobody can check.

**Brent's method, not Newton's.** Newton follows the slope, and for options
far from the money that slope collapses to nearly nothing. Brent keeps the
root bracketed and cannot run away.

**Every calendar day is probed, not just weekdays.** NSE opens on a Saturday
or Sunday when the Budget falls on one. Six such sessions are in this data,
and skipping weekends would have lost two Budget days.

**A failed download is never recorded as a holiday.** Two December 2023 days
were reported as holidays before this was fixed; both had files, and the
connection had simply dropped.
