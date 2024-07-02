import tensorflow as tf

# Black-Scholes公式定价
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (tf.math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * tf.math.sqrt(T))
    d2 = d1 - sigma * tf.math.sqrt(T)
    call_price = S * tf.math.erfc(-d1 / tf.math.sqrt(2.0)) / 2.0 - K * tf.math.exp(-r * T) * tf.math.erfc(-d2 / tf.math.sqrt(2.0)) / 2.0
    return call_price

# 定义期权参数
S = tf.Variable(100.0)  # 标的资产价格
K = 100.0               # 行权价
T = 1.0                 # 到期时间（年）
r = 0.05                # 无风险利率
sigma = 0.2             # 波动率

# 使用自动微分计算Delta
with tf.GradientTape() as tape:
    call_price = black_scholes_call_price(S, K, T, r, sigma)

delta = tape.gradient(call_price, S)

print(f"The Delta of the call option is: {delta.numpy()}")
