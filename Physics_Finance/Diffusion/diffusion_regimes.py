import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as student_t

np.random.seed(42)
S0 = 100
T = 1
N = 252
dt = T/N
mu = 0.05
sigma = 0.2
n_paths = 10000

# --- Gaussian (Black-Scholes) ---
Z = np.random.normal(0, 1, (n_paths, N))
returns_bs = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
prices_bs = S0 * np.exp(np.cumsum(returns_bs, axis=1))

# --- Student-t (fat tails) ---
df = 4  # degrees of freedom — lower = fatter tails
Z_t = student_t.rvs(df, size=(n_paths, N))
Z_t_scaled = Z_t / np.sqrt(df/(df-2))  # normalise to unit variance
returns_t = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_t_scaled
prices_t = S0 * np.exp(np.cumsum(returns_t, axis=1))

# --- Jump diffusion (Merton) ---
lam = 0.5    # jump intensity — expected jumps per year
mu_j = -0.05  # mean jump size
sig_j = 0.1   # jump size volatility
poisson_jumps = np.random.poisson(lam*dt, (n_paths, N))
jump_sizes = np.random.normal(mu_j, sig_j, (n_paths, N))
returns_merton = (mu - 0.5*sigma**2 - lam*mu_j)*dt + \
                  sigma*np.sqrt(dt)*np.random.normal(0,1,(n_paths,N)) + \
                  poisson_jumps * jump_sizes
prices_merton = S0 * np.exp(np.cumsum(returns_merton, axis=1))

# --- Final returns ---
final_ret_bs = np.log(prices_bs[:,-1]/S0)
final_ret_t = np.log(prices_t[:,-1]/S0)
final_ret_merton = np.log(prices_merton[:,-1]/S0)

# --- Shannon entropy ---
def shannon_entropy(data, bins=100):
    counts, _ = np.histogram(data, bins=bins, density=True)
    counts = counts[counts > 0]
    counts = counts / counts.sum()
    return -np.sum(counts * np.log(counts))

print(f"Gaussian entropy:      {shannon_entropy(final_ret_bs):.4f}")
print(f"Student-t entropy:     {shannon_entropy(final_ret_t):.4f}")
print(f"Jump diffusion entropy:{shannon_entropy(final_ret_merton):.4f}")

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Return distributions
x = np.linspace(-1.5, 1.5, 200)
axes[0].hist(final_ret_bs, bins=100, density=True, alpha=0.4, label='Gaussian (BS)', color='blue')
axes[0].hist(final_ret_t, bins=100, density=True, alpha=0.4, label='Student-t', color='red')
axes[0].hist(final_ret_merton, bins=100, density=True, alpha=0.4, label='Jump Diffusion', color='green')
axes[0].set_title('Return Distributions: Gaussian vs Fat-Tailed vs Jump')
axes[0].set_xlabel('Log Return')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].set_xlim(-1.5, 1.5)

# QQ plot against Gaussian
for ret, label, color in [(final_ret_bs, 'Gaussian', 'blue'),
                           (final_ret_t, 'Student-t', 'red'),
                           (final_ret_merton, 'Jump', 'green')]:
    (osm, osr), (slope, intercept, r) = stats.probplot(ret, dist='norm')
    axes[1].plot(osm, osr, color=color, alpha=0.7, label=label)

axes[1].plot([-4,4], [-4,4], 'k--', label='Perfect Gaussian')
axes[1].set_title('QQ Plot vs Gaussian — Fat Tails Visible')
axes[1].set_xlabel('Theoretical Quantiles')
axes[1].set_ylabel('Sample Quantiles')
axes[1].legend()

plt.tight_layout()
plt.savefig('week1_diffusion_regimes.png', dpi=150)
plt.show()