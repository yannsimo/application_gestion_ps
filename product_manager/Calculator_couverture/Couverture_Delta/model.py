import numpy as np
from numba import jit, prange
from numba.typed import List

def get_singleton_market_data():
    from ...views import SingletonMarketData
    return SingletonMarketData


@jit(nopython=True, fastmath=True)
def simulate_single_path(S0, r, sigma, T, dt, Z):
    steps = int(T / dt)
    S = np.zeros(steps + 1, dtype=np.float64)
    S[0] = S0
    sqrt_dt = np.sqrt(dt)
    exp_term = (r - 0.5 * sigma ** 2) * dt

    for i in range(1, steps + 1):
        S[i] = S[i - 1] * np.exp(exp_term + sigma * sqrt_dt * Z[i - 1])

    return S


@jit(nopython=True, parallel=True, fastmath=True)
def simulate_multiple_paths(S0, r, sigma, T, dt, num_simulations, Z):
    steps = int(T / dt)
    paths = np.zeros((num_simulations, steps + 1), dtype=np.float64)

    for sim in prange(num_simulations):
        paths[sim] = simulate_single_path(S0, r, sigma, T, dt, Z[:, sim])  # Utilisation des chocs corrélés

    return paths


class BlackScholesSimulation:
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def simulate_multiple_indices(S0_list, r_list, sigma_list, T, dt, num_simulations, correlated_Z):
        num_indices = len(S0_list)
        steps = int(T / dt)

        # Correction : Initialiser `results` correctement
        results = np.zeros((num_indices, num_simulations, steps + 1), dtype=np.float64)

        for i in prange(num_indices):
            results[i] = simulate_multiple_paths(S0_list[i], r_list[i], sigma_list[i], T, dt, num_simulations, correlated_Z[i])
        return results
