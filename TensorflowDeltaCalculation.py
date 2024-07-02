import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 参数
S0 = 100.0   # 当前资产价格
K = 110.0    # 行权价
T = 1.0      # 到期时间（年）
r = 0.00     # 无风险利率
sigma = 0.3  # 波动率
num_simulations = 1000000  # 模拟次数

# 模拟资产价格路径
def simulate_asset_prices(S0, T, r, sigma, num_simulations):
    S0 = tf.convert_to_tensor(S0, dtype=tf.float32)
    T = tf.convert_to_tensor(T, dtype=tf.float32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
    Z = tf.random.normal(shape=[num_simulations], dtype=tf.float32)
    ST = S0 * tf.exp((r - 0.5 * sigma**2) * T + sigma * tf.sqrt(T) * Z)
    return ST

# 计算欧式看涨期权价格
def call_option_payoff(S, K):
    K = tf.convert_to_tensor(K, dtype=tf.float32)
    return tf.maximum(S - K, 0.0)

# 使用蒙特卡洛模拟计算期权价格
def monte_carlo_call_option_price(S0, K, T, r, sigma, num_simulations):
    ST = simulate_asset_prices(S0, T, r, sigma, num_simulations)
    payoffs = call_option_payoff(ST, K)
    call_price = tf.exp(-r * T) * tf.reduce_mean(payoffs)
    return call_price, ST

# 基准期权价格
call_price_base, simulated_prices = monte_carlo_call_option_price(S0, K, T, r, sigma, num_simulations)

# 使用自动微分计算Delta
S = tf.Variable(S0, dtype=tf.float32)
with tf.GradientTape() as tape:
    ST = simulate_asset_prices(S, T, r, sigma, num_simulations)
    payoffs = call_option_payoff(ST, K)
    call_price = tf.exp(-r * T) * tf.reduce_mean(payoffs)

delta = tape.gradient(call_price, S)

# 输出期权价格
print(f"The Monte Carlo estimated call option price is: {call_price_base.numpy()}")
print(f"The Monte Carlo estimated Delta of the call option is: {delta.numpy()}")

# 绘制价格走势
plt.figure(figsize=(10, 6))
plt.hist(simulated_prices.numpy(), bins=100, density=True, alpha=0.6, color='b')
plt.title('Simulated End Prices Distribution')
plt.xlabel('Asset Price at T')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
