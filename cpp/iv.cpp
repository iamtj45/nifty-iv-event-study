// The implied-volatility solver again, in C++.
//
// Build (Windows, MinGW-w64):
//     g++ -O2 -shared -static-libgcc -static-libstdc++ -o libiv.dll iv.cpp
// Build (Linux/macOS):
//     g++ -O2 -shared -fPIC -o libiv.so iv.cpp
//
// Why this file exists. Nothing in the study needs it: build.py already runs
// in a few minutes and iv.py is correct. It is here to answer two questions a
// trading desk would ask, and to answer them with measurements rather than
// claims.
//
//   1. Is the maths right? Two independent implementations agreeing to twelve
//      decimal places is a much stronger statement than one implementation
//      passing its own tests. A transcription error in either shows up
//      immediately; a shared misunderstanding of Black-Scholes would not, but
//      nothing catches that except the put-call parity and bounds checks that
//      both versions already do.
//
//   2. How much faster is compiled code, really? bench.py measures it instead
//      of assuming. The answer is interesting partly because it is smaller
//      than people expect once the comparison is honest.
//
// Three decisions worth knowing about:
//
//   erfc, not a hand-rolled normal CDF. iv.py deliberately uses math.erfc so
//   that this file can use std::erfc and get bit-comparable results. Writing
//   a polynomial approximation here -- the usual textbook approach -- would
//   introduce differences of 1e-7 and make it impossible to tell a porting
//   bug from an approximation artefact.
//
//   Brent's method implemented by hand. There is no SciPy here, so the
//   algorithm is written out: Brent's 1973 zeroin, the same method
//   scipy.optimize.brentq uses, with the same tolerance of 1e-8. Both
//   converge on the same mathematical root, so the two agree to far better
//   than the tolerance.
//
//   A batch entry point. The single-value function is the readable one, but
//   calling it once per option from Python means paying ctypes marshalling
//   overhead on every call -- which is comparable to the arithmetic itself and
//   would make the speed test meaningless. iv_implied_vol_batch crosses the
//   boundary once for an entire array. Measuring the batch version is the
//   honest comparison, and bench.py reports both so the overhead is visible.

#include <cmath>
#include <limits>

#if defined(_WIN32)
#define IV_EXPORT extern "C" __declspec(dllexport)
#else
#define IV_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

// Mirrors iv.py exactly.
constexpr double VOL_MIN = 1e-4;
constexpr double VOL_MAX = 5.0;
constexpr double XTOL = 1e-8;
constexpr int MAXITER = 100;

constexpr int KIND_CALL = 0;
constexpr int KIND_PUT = 1;

const double NAN_VALUE = std::numeric_limits<double>::quiet_NaN();

// 0.5 * erfc(-x / sqrt(2)) -- the same expression as iv.py, for the same
// reason: std::erfc and math.erfc are both the platform's erfc.
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double bs_price(double S, double K, double T, double r, double sigma, int kind) {
    if (sigma <= 0.0 || T <= 0.0) return NAN_VALUE;

    const double sqrt_t = std::sqrt(T);
    const double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T)
                      / (sigma * sqrt_t);
    const double d2 = d1 - sigma * sqrt_t;
    const double pv_strike = K * std::exp(-r * T);

    if (kind == KIND_CALL) return S * norm_cdf(d1) - pv_strike * norm_cdf(d2);
    if (kind == KIND_PUT) return pv_strike * norm_cdf(-d2) - S * norm_cdf(-d1);
    return NAN_VALUE;
}

// The no-arbitrage range. A price outside it cannot have come from a real
// trade, and feeding one to the solver produces a confident wrong answer
// rather than an error -- which is why both implementations check first.
void price_bounds(double S, double K, double T, double r, int kind,
                  double* low, double* high) {
    const double pv_strike = K * std::exp(-r * T);
    if (kind == KIND_CALL) {
        *low = std::fmax(0.0, S - pv_strike);
        *high = S;
    } else {
        *low = std::fmax(0.0, pv_strike - S);
        *high = pv_strike;
    }
}

struct Target {
    double price, S, K, T, r;
    int kind;

    // How far bs_price overshoots the market price. The root of this is the
    // implied volatility.
    double excess(double sigma) const {
        return bs_price(S, K, T, r, sigma, kind) - price;
    }
};

// Brent's method (zeroin, Brent 1973). Keeps the root bracketed at all times,
// so unlike Newton it cannot step somewhere absurd when vega collapses --
// which is the whole reason for choosing it in iv.py.
//
// It combines three strategies: inverse quadratic interpolation when the three
// most recent points allow it, the secant method when they do not, and
// bisection whenever the interpolated step looks unreliable. The bisection
// fallback is what guarantees convergence.
double brent(const Target& t, double a, double b, double fa, double fb) {
    if (fa == 0.0) return a;
    if (fb == 0.0) return b;

    double c = a, fc = fa, d = 0.0, e = 0.0;

    for (int i = 0; i < MAXITER; ++i) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c = a; fc = fa; d = b - a; e = d;
        }
        if (std::fabs(fc) < std::fabs(fb)) {
            a = b;  b = c;  c = a;
            fa = fb; fb = fc; fc = fa;
        }

        const double tol = 2.0 * std::numeric_limits<double>::epsilon()
                           * std::fabs(b) + 0.5 * XTOL;
        const double m = 0.5 * (c - b);
        if (std::fabs(m) <= tol || fb == 0.0) return b;

        if (std::fabs(e) >= tol && std::fabs(fa) > std::fabs(fb)) {
            double p, q;
            const double s = fb / fa;
            if (a == c) {                       // secant
                p = 2.0 * m * s;
                q = 1.0 - s;
            } else {                            // inverse quadratic
                const double qq = fa / fc, rr = fb / fc;
                p = s * (2.0 * m * qq * (qq - rr) - (b - a) * (rr - 1.0));
                q = (qq - 1.0) * (rr - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q; else p = -p;

            if (2.0 * p < std::fmin(3.0 * m * q - std::fabs(tol * q),
                                    std::fabs(e * q))) {
                e = d;                          // accept the interpolation
                d = p / q;
            } else {
                d = m; e = m;                   // fall back to bisection
            }
        } else {
            d = m; e = m;
        }

        a = b; fa = fb;
        b += (std::fabs(d) > tol) ? d : (m > 0.0 ? tol : -tol);
        fb = t.excess(b);
    }
    return NAN_VALUE;                           // did not converge in MAXITER
}

}  // namespace


IV_EXPORT double iv_bs_price(double S, double K, double T, double r,
                             double sigma, int kind) {
    return bs_price(S, K, T, r, sigma, kind);
}


// The same sequence of guards as iv.py, in the same order, so a price that
// returns NaN there returns NaN here. Any disagreement is a bug in one of
// them, and bench.py counts disagreements rather than ignoring them.
IV_EXPORT double iv_implied_vol(double price, double S, double K, double T,
                                double r, int kind) {
    if (!(price > 0.0) || S <= 0.0 || K <= 0.0 || T <= 0.0) return NAN_VALUE;
    if (kind != KIND_CALL && kind != KIND_PUT) return NAN_VALUE;

    double low, high;
    price_bounds(S, K, T, r, kind, &low, &high);
    if (price < low || price > high) return NAN_VALUE;

    const Target t{price, S, K, T, r, kind};
    const double at_min = t.excess(VOL_MIN);
    const double at_max = t.excess(VOL_MAX);
    if (std::isnan(at_min) || std::isnan(at_max) || at_min * at_max > 0.0) {
        return NAN_VALUE;
    }
    return brent(t, VOL_MIN, VOL_MAX, at_min, at_max);
}


// One crossing of the Python/C boundary for a whole array. See the note at the
// top: per-call ctypes overhead is the same order as the arithmetic, so this
// is the version worth timing.
IV_EXPORT void iv_implied_vol_batch(const double* price, const double* S,
                                    const double* K, const double* T,
                                    const double* r, const int* kind,
                                    double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = iv_implied_vol(price[i], S[i], K[i], T[i], r[i], kind[i]);
    }
}


IV_EXPORT const char* iv_version() {
    return "nifty-iv 1.0 (" __DATE__ ")";
}
