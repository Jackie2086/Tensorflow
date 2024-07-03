# Output the option price and Delta
print(f"The Monte Carlo estimated call option price is: {call_price_base.numpy()}")
for i in range(m):
    print(f"The Monte Carlo estimated Delta of the call option for A{i+1} is: {delta[i].numpy()}")
for j in range(n):
    print(f"The Monte Carlo estimated Delta of the call option for B{j+1} is: {delta[m + j].numpy()}")

# Plot
fig, axs = plt.subplots(2, m, figsize=(12, 10))

# Plot simulated end prices for A
for i in range(m):
    axs[0, i].hist(simulated_prices_A[i].numpy(), bins=100, density=True, alpha=0.6, color='b')
    axs[0, i].set_title(f"Simulated End Prices Distribution for A{i+1}")
    axs[0, i].set_xlabel(f"Asset Price at T")
    axs[0, i].set_ylabel("Frequency")

# Plot simulated end prices for B
fig, axs = plt.subplots(2, n, figsize=(12, 10))
for j in range(n):
    axs[1, j].hist(simulated_prices_B[j].numpy(), bins=100, density=True, alpha=0.6, color='r')
    axs[1, j].set_title(f"Simulated End Prices Distribution for B{j+1}")
    axs[1, j].set_xlabel(f"Asset Price at T")
    axs[1, j].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
