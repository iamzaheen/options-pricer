"""
Options Pricer — Quant Finance Portfolio Project
=================================================
Prices European options using two methods:
  1. Black-Scholes (closed-form analytical solution)
  2. Monte Carlo simulation (numerical method)

Also calculates the Greeks: Delta, Gamma, Vega, Theta, Rho

Author: Syed Mohammad Zaheen
MSc Quantitative Finance, University of Kiel
GitHub: github.com/iamzaheen
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. BLACK-SCHOLES PRICER
# ─────────────────────────────────────────────
# The Black-Scholes model assumes the stock price follows
# Geometric Brownian Motion (GBM):
#   dS = μS dt + σS dW
#
# For a European call option, the closed-form price is:
#   C = S * N(d1) - K * e^(-rT) * N(d2)
#
# where:
#   d1 = [ln(S/K) + (r + σ²/2) * T] / (σ * √T)
#   d2 = d1 - σ * √T
#   N(.) = cumulative normal distribution

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Price a European option using Black-Scholes formula.

    Parameters
    ----------
    S           : float — current stock price (e.g. 100)
    K           : float — strike price (e.g. 105)
    T           : float — time to expiry in years (e.g. 0.5 = 6 months)
    r           : float — risk-free rate as decimal (e.g. 0.05 = 5%)
    sigma       : float — volatility as decimal (e.g. 0.20 = 20%)
    option_type : str   — 'call' or 'put'

    Returns
    -------
    price : float — option price
    d1    : float — d1 parameter
    d2    : float — d2 parameter
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price, d1, d2


# ─────────────────────────────────────────────
# 2. THE GREEKS
# ─────────────────────────────────────────────
# Greeks measure how sensitive the option price is to changes
# in each input parameter. Essential for risk management.

def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute all 5 major Greeks for a European option.

    Returns a dict with: delta, gamma, vega, theta, rho
    """
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'vega':  vega,
        'theta': theta,
        'rho':   rho
    }


# ─────────────────────────────────────────────
# 3. MONTE CARLO PRICER
# ─────────────────────────────────────────────
# Under risk-neutral measure, the stock price at expiry is:
#   S_T = S * exp((r - σ²/2)*T + σ*√T*Z)
# where Z ~ N(0,1)

def monte_carlo(S, K, T, r, sigma, option_type='call', n_simulations=100_000):
    """
    Price a European option using Monte Carlo simulation.

    Parameters
    ----------
    n_simulations : int — number of simulated paths (default: 100,000)

    Returns
    -------
    price     : float — estimated option price
    std_error : float — standard error of the estimate
    """
    np.random.seed(42)
    Z   = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    price     = float(np.mean(discounted_payoffs))
    std_error = float(np.std(discounted_payoffs) / np.sqrt(n_simulations))

    return price, std_error


# ─────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_payoff_diagram(S, K, T, r, sigma, option_type='call'):
    """Plot option price vs stock price — the classic hockey stick diagram."""
    stock_prices = np.linspace(S * 0.5, S * 1.5, 200)
    bs_prices    = [black_scholes(s, K, T, r, sigma, option_type)[0] for s in stock_prices]
    intrinsic    = [max(s - K, 0) if option_type == 'call' else max(K - s, 0)
                    for s in stock_prices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_prices, bs_prices, label='Black-Scholes price',
            color='steelblue', lw=2)
    ax.plot(stock_prices, intrinsic, label='Intrinsic value (at expiry)',
            color='gray', lw=1.5, linestyle='--')
    ax.axvline(K, color='red',   linestyle=':', alpha=0.7, label=f'Strike K={K}')
    ax.axvline(S, color='green', linestyle=':', alpha=0.7, label=f'Current price S={S}')
    ax.fill_between(stock_prices, intrinsic, bs_prices,
                    alpha=0.1, color='steelblue', label='Time value')
    ax.set_xlabel('Stock Price ($)', fontsize=12)
    ax.set_ylabel('Option Price ($)', fontsize=12)
    ax.set_title(f'European {option_type.capitalize()} Option — Payoff Diagram', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('payoff_diagram.png', dpi=150)
    plt.show()
    print("Saved: payoff_diagram.png")


def plot_mc_distribution(S, K, T, r, sigma, option_type='call', n_simulations=100_000):
    """Plot distribution of simulated terminal stock prices and payoffs."""
    np.random.seed(42)
    Z   = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(S_T, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(K, color='red', lw=2, label=f'Strike K={K}')
    ax1.axvline(np.mean(S_T), color='orange', lw=2,
                label=f'Mean S_T = {np.mean(S_T):.1f}')
    ax1.set_title('Simulated Terminal Stock Prices', fontsize=13)
    ax1.set_xlabel('Stock Price at Expiry ($)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(payoffs[payoffs > 0], bins=60, color='green', edgecolor='white', alpha=0.8)
    pct_itm = 100 * np.mean(payoffs > 0)
    ax2.set_title(f'Option Payoffs (in-the-money: {pct_itm:.1f}%)', fontsize=13)
    ax2.set_xlabel('Payoff ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Monte Carlo Simulation — {n_simulations:,} paths', fontsize=14)
    plt.tight_layout()
    plt.savefig('mc_distribution.png', dpi=150)
    plt.show()
    print("Saved: mc_distribution.png")


def plot_greeks_vs_spot(S, K, T, r, sigma, option_type='call'):
    """Plot all 5 Greeks as the stock price changes."""
    spot_range  = np.linspace(S * 0.5, S * 1.5, 200)
    greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
    greek_data  = {g: [] for g in greek_names}

    for s in spot_range:
        g = greeks(s, K, T, r, sigma, option_type)
        for name in greek_names:
            greek_data[name].append(g[name])

    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (name, color) in enumerate(zip(greek_names, colors)):
        axes[i].plot(spot_range, greek_data[name], color=color, lw=2)
        axes[i].axvline(K, color='gray', linestyle='--', alpha=0.5, label='Strike')
        axes[i].axvline(S, color='black', linestyle=':', alpha=0.5, label='Spot')
        axes[i].set_title(name.capitalize(), fontsize=13)
        axes[i].set_xlabel('Stock Price ($)')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=8)

    axes[5].axis('off')
    fig.suptitle(f'Option Greeks vs Stock Price — {option_type.capitalize()}', fontsize=15)
    plt.tight_layout()
    plt.savefig('greeks.png', dpi=150)
    plt.show()
    print("Saved: greeks.png")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':

    S     = 100
    K     = 105
    T     = 0.5
    r     = 0.05
    sigma = 0.20
    otype = 'call'

    print("=" * 55)
    print("  EUROPEAN OPTIONS PRICER")
    print("=" * 55)
    print(f"\n  Parameters:")
    print(f"    Stock price    S = {S}")
    print(f"    Strike price   K = {K}")
    print(f"    Time to expiry   = {T} years ({T*12:.0f} months)")
    print(f"    Risk-free rate   = {r*100:.1f}%")
    print(f"    Volatility       = {sigma*100:.1f}%")
    print(f"    Option type      = {otype.upper()}")

    bs_price, d1, d2 = black_scholes(S, K, T, r, sigma, otype)
    print(f"\n  BLACK-SCHOLES RESULT")
    print(f"  d1 = {d1:.4f}  |  d2 = {d2:.4f}")
    print(f"  Option price = {bs_price:.4f}")

    g = greeks(S, K, T, r, sigma, otype)
    print(f"\n  GREEKS")
    print(f"  Delta (d) = {g['delta']:.4f}  -> option moves {g['delta']:.2f} per $1 stock move")
    print(f"  Gamma (G) = {g['gamma']:.4f}  -> delta changes by {g['gamma']:.4f} per $1 move")
    print(f"  Vega  (v) = {g['vega']:.4f}  -> price changes {g['vega']:.4f} per 1% vol change")
    print(f"  Theta (T) = {g['theta']:.4f}  -> option loses {abs(g['theta']):.4f} per day")
    print(f"  Rho   (r) = {g['rho']:.4f}  -> price changes {g['rho']:.4f} per 1% rate change")

    mc_price, mc_error = monte_carlo(S, K, T, r, sigma, otype)
    print(f"\n  MONTE CARLO RESULT (100,000 simulations)")
    print(f"  Option price = {mc_price:.4f}")
    print(f"  Std error    = {mc_error:.4f}")
    print(f"  95% CI       = [{mc_price - 1.96*mc_error:.4f}, {mc_price + 1.96*mc_error:.4f}]")
    print(f"  Difference from BS = {abs(bs_price - mc_price):.4f}")

    print("\n  Generating charts...")
    plot_payoff_diagram(S, K, T, r, sigma, otype)
    plot_mc_distribution(S, K, T, r, sigma, otype)
    plot_greeks_vs_spot(S, K, T, r, sigma, otype)
    print("\n  All done. Check the 3 saved PNG files.")
