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

# Calculate payoffs and discounted option price
payoffs = np.maximum(K - ST, 0)
discounted_payoffs = np.exp(-r * T) * payoffs
option_price = np.mean(discounted_payoffs)

# Calculate confidence interval for the discounted option price
discounted_std = np.std(discounted_payoffs)  # Standard deviation of discounted payoffs
confidence_interval = [
    option_price - 1.96 * discounted_std / np.sqrt(N),
    option_price + 1.96 * discounted_std / np.sqrt(N)
]

print(f"Monte Carlo price of the put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")



# Q2
# Generate antithetic stock price paths
np.random.seed(1)
Z = np.random.normal(0, 1, N)
ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
ST2 = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Z))

# Calculate payoffs and use antithetic averaging
payoffs1 = np.maximum(K - ST1, 0)
payoffs2 = np.maximum(K - ST2, 0)
payoffs = (payoffs1 + payoffs2) / 2
discounted_payoffs = np.exp(-r * T) * payoffs

# Calculate the option price and confidence interval
option_price = np.mean(discounted_payoffs)
discounted_std = np.std(discounted_payoffs)
confidence_interval = [
    option_price - 1.96 * discounted_std / np.sqrt(N),
    option_price + 1.96 * discounted_std / np.sqrt(N)
]

# Black-Scholes theoretical price
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

option_price_bs_theoretical = black_scholes_put(S0, K, T, r, sigma)


print(f"Theoretical Black-Scholes price of the put option: {option_price_bs_theoretical:.4f}")
print(f"Monte Carlo price of the put option using antithetic variables: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")




# --- Q3: Knock-in Option ---

# Parameters (as defined earlier)
B = 64.55  # Barrier level
dt = T / M
Z = np.random.normal(0, 1, (N, M))

# Stock price paths initialization
S_paths = np.zeros((N, M + 1))
S_paths[:, 0] = S0

# Simulate stock price paths
for t in range(1, M + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

barrier_breached = np.any(S_paths <= B, axis=1)
ST = S_paths[:, -1]
payoffs = np.where(barrier_breached, np.maximum(K - ST, 0), 0)
discounted_payoffs = np.exp(-r * T) * payoffs

# Calculate the option price and confidence interval using discounted payoffs
option_price = np.mean(discounted_payoffs)
discounted_std = np.std(discounted_payoffs)
confidence_interval = [
    option_price - 1.96 * discounted_std / np.sqrt(N),
    option_price + 1.96 * discounted_std / np.sqrt(N)
]

print(f"Monte Carlo price of the knock-in put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")

# Q4

# Knock-out payoff
knock_out_payoffs = np.where(barrier_breached, 0, np.maximum(K - ST, 0))
discounted_knock_out_payoffs = np.exp(-r * T) * knock_out_payoffs

# Control variate payoffs
control_payoffs = np.maximum(K - ST, 0)
discounted_control_payoffs = np.exp(-r * T) * control_payoffs

# Theoretical control option price
C_true_control = black_scholes_put(S0, K, T, r, sigma)

# Control variate adjustment
covariance = np.cov(discounted_knock_out_payoffs, discounted_control_payoffs)[0, 1]
variance_control = np.var(discounted_control_payoffs)
beta = -covariance / variance_control

# Adjusted price and confidence interval
C_final = np.mean(discounted_knock_out_payoffs) + beta * (np.mean(discounted_control_payoffs) - C_true_control)
adjusted_payoffs = discounted_knock_out_payoffs + beta * (discounted_control_payoffs - C_true_control)
adjusted_std = np.std(adjusted_payoffs)
confidence_interval = [
    C_final - 1.96 * adjusted_std / np.sqrt(N),
    C_final + 1.96 * adjusted_std / np.sqrt(N)
]

print(f"Monte Carlo price of the knock-out put option (with control variate): {C_final:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")



#Q5

# Simulate stock price paths using geometric Brownian motion
for t in range(1, M + 1):
    S_paths[:, t] = S_paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

# Lookback option is path-dependent
max_prices = np.max(S_paths, axis=1)
ST = S_paths[:, -1]
payoffs = np.maximum(max_prices - ST, 0)

# Discount payoffs before calculating the option price
discounted_payoffs = np.exp(-r * T) * payoffs
option_price = np.mean(discounted_payoffs)

# Correctly calculate confidence interval based on discounted payoffs
sigv = np.std(discounted_payoffs)
confidence_interval = [
    option_price - 1.96 * sigv / np.sqrt(N),
    option_price + 1.96 * sigv / np.sqrt(N)
]

print(f"Monte Carlo price of the lookback put option: {option_price:.4f}")
print(f"95% confidence interval: ({confidence_interval[0]:.4f}, {confidence_interval[1]:.4f})")
print(f"Width of the confidence interval: {confidence_interval[1] - confidence_interval[0]:.4f}")


#Q6

import numpy as np

# Parameters
S0 = 100          # Initial stock price
sigma = 0.2       # Volatility (20%)
r = 0.015         # Risk-free rate (1.5%)
T = 1             # Time to maturity (1 year)
K = 100           # Strike price is based on max price during the path
N_initial = 10000 # Initial simulations to estimate standard deviation
M = 252           # Number of time steps (daily observations for 1 year)

# Generate random paths and their antithetic counterparts
np.random.seed(1)
dt = T / M
Z = np.random.normal(0, 1, (N_initial, M))
# Parameters remain the same
S_paths_1 = np.zeros((N_initial, M + 1))
S_paths_2 = np.zeros((N_initial, M + 1))
S_paths_1[:, 0] = S0
S_paths_2[:, 0] = S0

# Simulate paths using GBM
for t in range(1, M + 1):
    S_paths_1[:, t] = S_paths_1[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])
    S_paths_2[:, t] = S_paths_2[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * (-Z[:, t - 1]))

# Payoffs for Lookback put option
max_price_1 = np.max(S_paths_1, axis=1)
max_price_2 = np.max(S_paths_2, axis=1)
payoffs_1 = np.maximum(max_price_1 - S_paths_1[:, -1], 0)
payoffs_2 = np.maximum(max_price_2 - S_paths_2[:, -1], 0)

# Discounted payoffs
discounted_payoffs_1 = np.exp(-r * T) * payoffs_1
discounted_payoffs_2 = np.exp(-r * T) * payoffs_2

# Calculate covariance and variance with discounted payoffs
covariance = np.cov(discounted_payoffs_1, discounted_payoffs_2)[0, 1]
variance_standard = np.var(discounted_payoffs_1)
variance_antithetic = (variance_standard + variance_standard + 2 * covariance) / 4
correlation = covariance / (np.std(discounted_payoffs_1) * np.std(discounted_payoffs_2))

# Standard deviations
std_standard = np.sqrt(variance_standard)
std_antithetic = np.sqrt(variance_antithetic)

# Calculate number of simulations required
desired_width = 0.01
critical_value = 1.96

N_standard = (2 * critical_value * std_standard / desired_width) ** 2
N_antithetic = (2 * critical_value * std_antithetic / desired_width) ** 2

# Display results
print(f"Estimated standard deviation (standard MC): {std_standard:.4f}")
print(f"Estimated standard deviation (antithetic variables): {std_antithetic:.4f}")
print(f"Covariance between paths: {covariance:.4f}")
print(f"Correlation between paths: {correlation:.4f}")
print(f"Required N for standard MC: {int(np.ceil(N_standard))}")
print(f"Required N for antithetic variables: {int(np.ceil(N_antithetic))}")
