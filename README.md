# Nifty IV Event Study

**What the Indian options market expects before major events, against what actually happens.**

Four years of NSE end-of-day option data, solved for implied volatility, lined
up around 29 scheduled events — RBI policy meetings, Union Budgets and the 2024
election result — and tested three ways: is the effect real, does it predict
direction, and could you have traded it.

```
968 trading days   1 Dec 2022 – 2 Nov 2026
29 events          23 RBI · 5 Budget · 1 election
2 indices          Nifty 50 and Bank Nifty
```

---

## The findings

### 1. Options are systematically overpriced before events

| Event | n | IV charged before | Volatility delivered | Overpriced in |
|---|---:|---:|---:|---:|
| RBI policy | 23 | 13.35% | 9.87% | **19 of 23** |
| Union Budget | 5 | 17.99% | 13.34% | 3 of 5 |
| Election result | 1 | 26.92% | 17.66% | 1 of 1 |

Before RBI meetings the market priced in 13.35% of volatility and got 9.87%.
On the day itself implied volatility fell **5.62%** on average.

### 2. It is not chance

A placebo test takes 448 ordinary trading days — at least ten days from any
event — draws 23 of them at random, averages, and repeats 10,000 times.

```
RBI       0.04 percentile      none of 10,000 random groups were as extreme
Budget    0.02 percentile
Election 98.88 percentile      went the other way, as expected
```

The 95% confidence range on the RBI crush is **−8.88% to −2.70%** — entirely
below zero.

### 3. Direction is not predictable

| | up | down | % up | coin-flip p |
|---|---:|---:|---:|---:|
| RBI days | 11 | 12 | 47.8% | 1.000 |
| ordinary days | 245 | 203 | 54.7% | 0.053 |

Ordinary days rise 54.7% of the time — the slow upward drift of equities. RBI
days rise 47.8%. **Knowing a meeting is tomorrow leaves you with less
directional information than an ordinary Tuesday.** The decision is in the
price before it is announced.

### 4. And you could not have traded it

One rule, fixed in advance: sell the at-the-money straddle at the close the day
before, buy it back at the close on the event day. Profit in index points.

| Group | Trades | Avg net | Won | Worst loss | Break-even cost |
|---|---:|---:|---:|---:|---:|
| RBI | 23 | −3.02 | 56.5% | −181.93 | **0.12** |
| Budget | 5 | +85.63 | 80.0% | −114.50 | 89.10 |
| Election | 1 | −722.36 | 0% | −722.36 | −716.80 |
| **ordinary days** | 449 | +2.52 | 62.4% | −429.00 | **5.58** |

"Break-even cost" is profit before any fees — which is also exactly the
round-trip cost at which the rule stops paying.

**Selling before RBI meetings earned 0.12 points before costs. Selling on a
random Tuesday earned 5.58.** The event trade is worse than trading at random,
because event days also *move* more (0.675% against 0.547%), and the extra
movement costs the seller more than the collapsing volatility pays.

The single worst trade in all 477 was not an event at all: an ordinary Monday,
12 May 2025, −429 points.

### 5. What volatility *does* tell you

| | correlation | days |
|---|---:|---:|
| IV change vs **same-day** return | **−0.5125** | 967 |
| IV change vs **next-day** return | +0.0328 | 966 |

On days volatility rose, the market fell **62%** of the time. On days it fell,
the market rose **66%** of the time.

But after the largest 10% of volatility spikes, the next day rose 54.6% — against
a 52.8% baseline. **Volatility measures today's direction accurately and
forecasts tomorrow's not at all.** Fear rising and prices falling are the same
event seen two ways, not a cause and its effect.

---

## Is it right?

The strongest evidence is external. **India VIX** is published by NSE using a
different method — many strikes over a fixed 30 days, where this uses one
near-dated strike.

```
our ATM IV       12.74
India VIX        13.81
correlation      0.9157   over 960 overlapping days
```

The ~1 point gap is expected and, importantly, **steady**. A broken solver
would drift — fine in calm markets, wrong in volatile ones. A constant offset
between two methods measuring slightly different things is what correctness
looks like.

**Bank Nifty corroborates independently.** Run through the same pipeline it
gives 15.03% mean IV against Nifty's 12.74% — higher, as a single-sector index
should be — and reaches the same conclusions: overpriced 21 of 23, placebo
0.05, and a straddle trade that *loses* 19.1 points before costs.

The crush is also the same size in **absolute** volatility points in both
indices (−0.87 vs −0.81), not proportionally larger in the rate-sensitive
banking index. The event premium is a roughly fixed quantity of volatility,
not a percentage markup.

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
python run_evals.py --index NIFTY --limit 35 --refusals 15
```

Four checks: **numbers** (is the exact figure quoted?), **citations** (was every
document named actually retrieved?), **refusals** (40 unanswerable questions,
including dates where no event happened), and **retrieval**.

> **Eval status:** numbers 10/10 and citations 10/10 on a partial run with
> `gpt-oss-120b`; refusals 4/4 of those reached before the free-tier daily
> token limit stopped the run. The full suite has not yet been run end to end.
> Eval results belong to the model that produced them — `gpt-oss-20b` failed
> deterministically on the advice question, returning empty content.

---

## What is here

| File | Lines | What it does |
|---|---:|---|
| `download.py` | 299 | Fetches NSE bhavcopies and index closes |
| `iv.py` | 149 | Black-Scholes and the Brent solve for IV |
| `build.py` | 597 | Raw files → daily IV, event study, placebo |
| `analysis.py` | 263 | Bootstrap intervals, placebo test, expected move |
| `backtest.py` | 472 | The straddle rule, events against ordinary days |
| `direction.py` | 355 | Sign tests, and volatility against direction |
| `update.py` | 268 | Brings everything up to the last trading day |
| `main.py` | 667 | FastAPI: the API and the question box |
| `rag.py` | 296 | PDF passages and BM25 retrieval |
| `run_evals.py` | 398 | The four eval checks |
| `mcp_server.py` | 319 | MCP tools so an assistant can query the study |
| `cpp/iv.cpp` | 218 | The solver again, in C++, for verification |
| `bench.py` | 321 | Accuracy and speed, Python against C++ |
| `tests/` | 442 | 80 tests on the maths and the statistics |

```bash
pytest          # 80 tests
python bench.py # C++ against Python: agreement and speed
```

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
