import numpy as np
from scipy.stats import norm

S0 = 100          # Initial stock price
K = 100          # Strike price
sigma = 0.2       # Volatility (20%)
r = 0.015         # Risk-free rate (1.5%)
T = 1             # Time to maturity (1 year)
N = 10000         # Number of Monte Carlo simulations
M = 252           # Number of time steps (business days)

# Q1
# Simulate terminal stock prices using Geometric Brownian Motion
np.random.seed(1)
Z = np.random.normal(0, 1, N)
ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

payoffs = np.maximum(K - ST, 0)

option_price = np.exp(-r * T) * np.mean(payoffs)

payoff_std = np.std(payoffs)
confidence_interval = [
    option_price - 1.96 * payoff_std / np.sqrt(N),
    option_price + 1.96 * payoff_std / np.sqrt(N)
]


print(f"Monte Carlo price of the put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")


#Q2
ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
ST2 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Z))

payoffs1 = np.maximum(K - ST1, 0)
payoffs2 = np.maximum(K - ST2, 0)

payoffs = (payoffs1 + payoffs2) / 2

option_price = np.exp(-r * T) * np.mean(payoffs)

# Use the Black-Scholes formula to find the exact control option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Theoretical price using the Black-Scholes formula
option_price_bs_theoretical = black_scholes_put(S0, K, T, r, sigma)

# Print the theoretical price for comparison
print(f"Theoretical Black-Scholes price of the put option (calculated): {option_price_bs_theoretical:.4f}")

sigv = np.std(payoffs)
confidence_interval = [
    option_price - 1.96 * sigv / np.sqrt(N),
    option_price + 1.96 * sigv / np.sqrt(N)
]

print(f"Monte Carlo price of the put option using antithetic variables: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")


#Q3

B = 64.55# Barrier level in British Pounds with exchange rate USDGBP = 0.81

dt = T / M
Z = np.random.normal(0, 1, (N, M))
S_paths = np.zeros((N, M + 1))
S_paths[:, 0] = S0


# Simulate the stock price paths
for t in range(1, M + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

barrier_breached = np.any(S_paths <= B, axis=1)

ST = S_paths[:, -1]  # Terminal stock prices
payoffs = np.where(barrier_breached, np.maximum(K - ST, 0), 0)

option_price = np.exp(-r * T) * np.mean(payoffs)

sigv = np.std(payoffs)
confidence_interval = [
    option_price - 1.96 * sigv / np.sqrt(N),
    option_price + 1.96 * sigv / np.sqrt(N)
]

print(f"Monte Carlo price of the knock-in put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")



#Q4

barrier_breached = np.any(S_paths <= B, axis=1)

# Calculate the knock-out option payoff
ST = S_paths[:, -1]
knock_out_payoffs = np.where(barrier_breached, 0, np.maximum(K - ST, 0))

control_payoffs = np.maximum(K - ST, 0)

C_knock_out = np.exp(-r * T) * np.mean(knock_out_payoffs)
C_control = np.exp(-r * T) * np.mean(control_payoffs)
# Use the Black-Scholes formula to find the exact control option price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

C_true_control = black_scholes_put(S0, K, T, r, sigma)

covariance = np.cov(knock_out_payoffs, control_payoffs)[0, 1]
variance_control = np.var(control_payoffs)
beta = -covariance / variance_control

# Apply the control variate adjustment
C_final = C_knock_out + beta * (C_control - C_true_control)

# Calculate the confidence interval
adjusted_payoffs = knock_out_payoffs + beta * (control_payoffs - C_true_control)
sigv = np.std(adjusted_payoffs)
confidence_interval = [
    C_final - 1.96 * sigv / np.sqrt(N),
    C_final + 1.96 * sigv / np.sqrt(N)
]

print(f"Monte Carlo price of the knock-out put option (with control variate): {C_final:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")



#Q5

# Simulate stock price paths using geometric Brownian motion
for t in range(1, M + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

max_prices = np.max(S_paths, axis=1)
ST = S_paths[:, -1]
payoffs = np.maximum(max_prices - ST, 0)


option_price = np.exp(-r * T) * np.mean(payoffs)

sigv = np.std(payoffs)
confidence_interval = [
    option_price - 1.96 * sigv / np.sqrt(N),
    option_price + 1.96 * sigv / np.sqrt(N)
]

print(f"Monte Carlo price of the lookback put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")


#6

import numpy as np

# Parameters
S0 = 100
sigma = 0.2
r = 0.015
T = 1
K = 100
N_initial = 10000
M = 252

# Generate stock price paths and calculate payoffs (Standard MC)
np.random.seed(0)
dt = T / M
Z = np.random.normal(0, 1, (N_initial, M))
S_paths = np.zeros((N_initial, M + 1))
S_paths[:, 0] = S0

for t in range(1, M + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

max_prices = np.max(S_paths, axis=1)
ST = S_paths[:, -1]
payoffs_standard = np.maximum(max_prices - ST, 0)
discounted_payoffs_standard = np.exp(-r * T) * payoffs_standard

# Estimate standard deviation
sigma_standard = np.std(discounted_payoffs_standard)

width_target = 0.01
critical_value = 1.96
N_required_standard = (2 * critical_value * sigma_standard / width_target) ** 2

np.random.seed(0)
Z = np.random.normal(0, 1, (N_initial, M))

S_paths1 = np.zeros((N_initial, M + 1))
S_paths2 = np.zeros((N_initial, M + 1))
S_paths1[:, 0], S_paths2[:, 0] = S0, S0

for t in range(1, M + 1):
    S_paths1[:, t] = S_paths1[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    S_paths2[:, t] = S_paths2[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt) * Z[:, t - 1])

max_prices1, max_prices2 = np.max(S_paths1, axis=1), np.max(S_paths2, axis=1)
ST1, ST2 = S_paths1[:, -1], S_paths2[:, -1]

payoffs1 = np.maximum(max_prices1 - ST1, 0)
payoffs2 = np.maximum(max_prices2 - ST2, 0)
payoffs_antithetic = (payoffs1 + payoffs2) / 2
discounted_payoffs_antithetic = np.exp(-r * T) * payoffs_antithetic

sigma_antithetic = np.std(discounted_payoffs_antithetic)

N_required_antithetic = (2 * critical_value * sigma_antithetic / width_target) ** 2


print(f"Estimated standard deviation (Standard MC): {sigma_standard:.4f}")
print(f"Estimated standard deviation (Antithetic MC): {sigma_antithetic:.4f}")
print(f"Required N for Standard MC to achieve width 0.01: {int(N_required_standard)}")
print(f"Required N for Antithetic MC to achieve width 0.01: {int(N_required_antithetic)}")
