import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 定义均值向量和协方差矩阵
mean = [0, 0, 0, 0]
cov = [
    [1, 0.9, 0.5, 0.5],
    [0.9, 1, 0.5, 0.5],
    [0.5, 0.5, 1, 0.9],
    [0.5, 0.5, 0.9, 1]
]

# 生成4个正态分布的序列
data = np.random.multivariate_normal(mean, cov, 1000)

# 将生成的序列分配到A1, A2, B1, B2
A1, A2, B1, B2 = data.T

# 可视化生成的序列
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(A1, label='A1')
plt.plot(A2, label='A2')
plt.title('Sequences for A1, A2')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(B1, label='B1')
plt.plot(B2, label='B2')
plt.title('Sequences for B1, B2')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(A1, label='A1')
plt.plot(B1, label='B1')
plt.plot(A2, label='A2')
plt.plot(B2, label='B2')
plt.title('Mixed Sequences')
plt.legend()

plt.tight_layout()
plt.show()

# 输出相关性矩阵来验证
print(np.corrcoef(data.T))

