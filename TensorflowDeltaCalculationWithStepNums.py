import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100.0   # Current asset price
K = 100.0    # Strike price
T = 1.0      # Time to maturity (years)
r = 0.05     # Risk-free interest rate
sigma = 0.2  # Volatility
num_simulations = 100000  # Number of simulations
num_steps = 2  # Number of time steps

# Simulate asset price paths
def simulate_asset_prices(S0, T, r, sigma, num_simulations, num_steps):
    dt = T / num_steps
    S0 = tf.convert_to_tensor(S0, dtype=tf.float32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    Z = tf.random.normal(shape=[num_simulations, num_steps], dtype=tf.float32)
    S = tf.TensorArray(tf.float32, size=num_steps+1)
    S = S.write(0, tf.fill([num_simulations], S0))

    for t in range(1, num_steps + 1):
        S_t = S.read(t-1) * tf.exp((r - 0.5 * sigma**2) * dt + sigma * tf.sqrt(dt) * Z[:, t-1])
        S = S.write(t, S_t)

    return S.stack()[-1, :]

# Calculate European call option payoff
def call_option_payoff(S, K):
    K = tf.convert_to_tensor(K, dtype=tf.float32)
    return tf.maximum(S - K, 0.0)

# Monte Carlo simulation to calculate option price
def monte_carlo_call_option_price(S0, K, T, r, sigma, num_simulations, num_steps):
    ST = simulate_asset_prices(S0, T, r, sigma, num_simulations, num_steps)
    payoffs = call_option_payoff(ST, K)
    call_price = tf.exp(-r * T) * tf.reduce_mean(payoffs)
    return call_price, ST

# Calculate the baseline call option price
call_price_base, simulated_prices = monte_carlo_call_option_price(S0, K, T, r, sigma, num_simulations, num_steps)

# Calculate Delta using automatic differentiation
S = tf.Variable(S0, dtype=tf.float32)
with tf.GradientTape() as tape:
    ST = simulate_asset_prices(S, T, r, sigma, num_simulations, num_steps)
    payoffs = call_option_payoff(ST, K)
    call_price = tf.exp(-r * T) * tf.reduce_mean(payoffs)

delta = tape.gradient(call_price, S)

# Output the option price and Delta
print(f"The Monte Carlo estimated call option price is: {call_price_base.numpy()}")
print(f"The Monte Carlo estimated Delta of the call option is: {delta.numpy()}")

# Plot the distribution of simulated end prices
plt.figure(figsize=(10, 6))
plt.hist(simulated_prices.numpy(), bins=100, density=True, alpha=0.6, color='b')
plt.title('Simulated End Prices Distribution')
plt.xlabel('Asset Price at T')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
