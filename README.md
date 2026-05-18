# Options Pricer — Black-Scholes & Monte Carlo

A Python implementation of European options pricing using two methods: the **Black-Scholes closed-form solution** and **Monte Carlo simulation**. Includes calculation of all five major Greeks.

Built as part of an 

---

## What this project covers

| Topic | Description |
|---|---|
| Black-Scholes model | Analytical pricing of European calls and puts |
| Monte Carlo simulation | 100,000-path numerical pricing with confidence intervals |
| The Greeks | Delta, Gamma, Vega, Theta, Rho — all computed and visualised |
| Visualisations | Payoff diagram, MC distribution, Greeks vs spot price |

---

## Theory

### Black-Scholes

Assumes stock price follows Geometric Brownian Motion:

```
dS = μS dt + σS dW
```

The closed-form price of a European call is:

```
C = S·N(d1) - K·e^(-rT)·N(d2)

d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d2 = d1 - σ·√T
```

### Monte Carlo

Simulates N terminal stock prices under the risk-neutral measure:

```
S_T = S · exp((r - σ²/2)·T + σ·√T·Z),  Z ~ N(0,1)
```

Averages discounted payoffs across all simulated paths.

---

## Usage

```bash
pip install numpy scipy matplotlib pandas
python options_pricer.py
```

### Example output

```
EUROPEAN OPTIONS PRICER
Parameters:
  Stock price  S = $100
  Strike price K = $105
  Time to expiry = 0.5 years (6 months)
  Risk-free rate = 5.0%
  Volatility     = 20.0%

BLACK-SCHOLES RESULT
  Option price = $6.0441

MONTE CARLO RESULT (100,000 simulations)
  Option price = $6.0387
  95% CI       = [$5.9856, $6.0919]
  Difference from B-S: $0.0054
```

### Change the parameters

Edit the parameters block in `options_pricer.py`:

```python
S     = 100    # Current stock price
K     = 105    # Strike price
T     = 0.5    # Time to expiry in years
r     = 0.05   # Risk-free rate
sigma = 0.20   # Annual volatility
otype = 'call' # 'call' or 'put'
```

---

## Output charts

The script generates three PNG files:

- `payoff_diagram.png` — Option price vs stock price, showing time value
- `mc_distribution.png` — Distribution of simulated terminal prices and payoffs
- `greeks.png` — All 5 Greeks plotted against the stock price

---

## Key concepts demonstrated

**Delta** — How much the option price moves per $1 change in the stock. A delta of 0.45 means the option gains $0.45 for every $1 the stock rises.

**Gamma** — How fast delta changes. High gamma = option is very sensitive near the strike.

**Vega** — Sensitivity to volatility. Buying options is implicitly a bet on volatility increasing.

**Theta** — Time decay. Options lose value every day as expiry approaches.

**Rho** — Sensitivity to interest rates. Less important for short-dated options.

---

## Next steps / extensions

- [ ] Add implied volatility solver (given market price, solve for σ)
- [ ] Price American options using Binomial Tree model
- [ ] Add volatility smile / surface visualisation
- [ ] Build a simple delta-hedging simulation

---

## Author





## Author

Syed Mohammad Zaheen
MSc Quantitative Finance
LinkedIn: [iamzaheen](https://linkedin.com/in/iamzaheen)
GitHub: [iamzaheen](https://github.com/iamzaheen)
