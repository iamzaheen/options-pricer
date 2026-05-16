import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

def greeks(S, K, T, r, sigma, option_type='call'):
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def monte_carlo(S, K, T, r, sigma, option_type='call', n_simulations=100_000):
    np.random.seed(42)
    Z = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)
    discounted = np.exp(-r * T) * payoffs
    return np.mean(discounted), np.std(discounted) / np.sqrt(n_simulations)

if __name__ == '__main__':
    S, K, T, r, sigma, otype = 100, 105, 0.5, 0.05, 0.20, 'call'
    print("=" * 55)
    print("  EUROPEAN OPTIONS PRICER")
    print("=" * 55)
    bs_price, d1, d2 = black_scholes(S, K, T, r, sigma, otype)
    print(f"\n  Black-Scholes price = \${bs_price:.4f}")
    print(f"  d1 = {d1:.4f}, d2 = {d2:.4f}")
    g = greeks(S, K, T, r, sigma, otype)
    print(f"\n  Greeks:")
    for name, val in g.items():
        print(f"    {name.capitalize():6} = {val:.4f}")
    mc_price, mc_err = monte_carlo(S, K, T, r, sigma, otype)
    print(f"\n  Monte Carlo price = \${mc_price:.4f}")
    print(f"  Std error         = \${mc_err:.4f}")
    print(f"  Difference from BS= \${abs(bs_price - mc_price):.4f}")
    print("\n  Done!")
