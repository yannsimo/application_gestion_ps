# testr.py - Version optimisée avec Numba
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from numba import njit, prange

from ..Product.parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates


# Fonctions numba optimisées - indépendantes de la classe
@njit(parallel=True)
def simulate_gbm_daily_paths_numba(initial_prices, volatilities, cholesky_matrix, time_steps, num_paths, seed):
    """
    Simulate price paths with daily time steps using Numba acceleration.
    """
    np.random.seed(seed)
    num_assets = len(initial_prices)

    # Initialize paths array
    paths = np.zeros((num_assets, num_paths, time_steps))

    # Set initial prices
    for i in range(num_assets):
        paths[i, :, 0] = initial_prices[i]

    # Daily time step (assuming 252 trading days per year)
    dt = 1.0 / 252.0

    # Simulate paths
    for t in range(1, time_steps):
        # Generate standard normal random numbers
        Z = np.random.standard_normal((num_assets, num_paths))

        # Apply Cholesky decomposition for correlation
        correlated_Z = np.zeros((num_assets, num_paths))
        for i in range(num_assets):
            for j in range(i + 1):
                correlated_Z[i, :] += cholesky_matrix[i, j] * Z[j, :]

        # Apply GBM formula with correlation
        for i in range(num_assets):
            drift = (0.0 - 0.5 * volatilities[i] ** 2) * dt
            diffusion = volatilities[i] * np.sqrt(dt) * correlated_Z[i, :]
            paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

    return paths


@njit(parallel=True)
def simulate_gbm_observation_paths_numba(initial_prices, volatilities, cholesky_matrix, time_increments, num_paths,
                                         seed):
    """
    Simulate price paths with jumps directly to observation dates using Numba acceleration.
    """
    np.random.seed(seed)
    num_assets = len(initial_prices)
    num_dates = len(time_increments) + 1

    # Initialize paths array
    paths = np.zeros((num_assets, num_paths, num_dates))

    # Set initial prices
    for i in range(num_assets):
        paths[i, :, 0] = initial_prices[i]

    # Simulate paths between each observation date
    for t in range(1, num_dates):
        # Get time increment in years
        dt = time_increments[t - 1]

        # Generate standard normal random numbers
        Z = np.random.standard_normal((num_assets, num_paths))

        # Apply Cholesky decomposition for correlation
        correlated_Z = np.zeros((num_assets, num_paths))
        for i in range(num_assets):
            for j in range(i + 1):
                correlated_Z[i, :] += cholesky_matrix[i, j] * Z[j, :]

        # Apply GBM formula with correlation
        for i in range(num_assets):
            drift = (0.0 - 0.5 * volatilities[i] ** 2) * dt
            diffusion = volatilities[i] * np.sqrt(dt) * correlated_Z[i, :]
            paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

    return paths


@njit(parallel=True)
def calculate_payoff_numba(paths, indices_count, minimum_guarantee, floor, cap,
                           participation_rate, dividend_multiplier):
    """
    Calculate product payoff from simulated paths using Numba acceleration.
    """
    num_paths = paths.shape[1]
    num_dates = paths.shape[2]

    # Container for results
    final_payoffs = np.zeros(num_paths)
    dividends = np.zeros((num_paths, num_dates - 2))  # Exclude T0 and Tc

    # Process each path
    for p in prange(num_paths):
        # For tracking excluded indices (only for dividend calculation)
        dividend_excluded_indices = np.zeros(indices_count, dtype=np.int8)
        threshold_reached = False

        # Process each observation date (skip first and last)
        for t in range(1, num_dates - 1):
            # Calculate performance for all indices
            perfs = np.zeros(indices_count)
            active_indices = np.zeros(indices_count, dtype=np.int8)
            active_count = 0

            for i in range(indices_count):
                perfs[i] = paths[i, p, t] / paths[i, p, t - 1] - 1.0

                # Only count if not excluded for dividend
                if dividend_excluded_indices[i] == 0:
                    active_indices[i] = 1
                    active_count += 1

            # Check if performance triggers minimum guarantee
            basket_perf = np.sum(perfs) / indices_count
            if basket_perf >= minimum_guarantee:
                threshold_reached = True

            # Find best performing index for dividend
            if active_count > 0:
                best_perf = -1.0
                best_idx = -1

                for i in range(indices_count):
                    if active_indices[i] == 1 and perfs[i] > best_perf:
                        best_perf = perfs[i]
                        best_idx = i

                # Calculate dividend
                dividends[p, t - 1] = max(0.0, dividend_multiplier * best_perf)

                # Exclude for future dividend calculations
                if best_idx >= 0:
                    dividend_excluded_indices[best_idx] = 1
            else:
                # No active indices left, so no dividend
                dividends[p, t - 1] = 0.0

        # Calculate final performance at maturity using ALL INDICES
        final_perfs = np.zeros(indices_count)
        for i in range(indices_count):
            final_perfs[i] = paths[i, p, -1] / paths[i, p, 0] - 1.0

        basket_final_perf = np.sum(final_perfs) / indices_count

        # Apply floors and caps
        if basket_final_perf < 0:
            basket_final_perf = max(basket_final_perf, floor)
        else:
            basket_final_perf = min(basket_final_perf, cap)

        # Apply minimum guarantee if threshold reached
        if threshold_reached:
            basket_final_perf = max(basket_final_perf, minimum_guarantee)

        # Apply participation rate
        final_payoffs[p] = 1000.0 * (1.0 + participation_rate * basket_final_perf)

    return final_payoffs, dividends


class SimpleSimulator:
    """Simple Monte Carlo simulator for Product 11 with Numba optimization"""

    def __init__(self):
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        self.start_date = self.product_parameter.current_date
        self.end_date = self.product_parameter.final_date
        # Product parameters for Product 11
        self.risk_free_rate = 0.02  # 2% risk-free rate for calculations
        # Calculate observation dates - simplify by dividing total period into 5 parts
        total_days = (self.end_date - self.start_date).days

    def _get_initial_prices(self):
        """Get initial prices for all underlying assets, converted to the pricing currency."""
        current_date = self.market_data.current_date

        return np.array([
            self.market_data.get_price(code, current_date) *
            self.market_data.get_index_exchange_rate(code, current_date)
            for code in self.product_parameter.underlying_indices
        ])

    def _get_volatilities(self):
        """Get volatilities for all underlying assets."""
        return np.array([
            self.product_parameter.volatilities[code]
            for code in self.product_parameter.underlying_indices
        ])

    def simulate_paths(self, num_paths=1000, seed=None, daily_steps=True):
        """
        Simulate price paths using Geometric Brownian Motion (GBM) with correlations.
        This version uses Numba optimization for significant performance improvement.

        Args:
            num_paths (int): Number of simulation paths to generate
            seed (int, optional): Random seed for reproducibility
            daily_steps (bool): If True, simulate with daily time steps, otherwise use observation dates only

        Returns:
            numpy.ndarray: Simulated price paths
        """
        if seed is None:
            seed = np.random.randint(0, 10000)

        # Setup initial parameters
        initial_prices = self._get_initial_prices()
        volatilities = self._get_volatilities()
        cholesky_matrix = self.product_parameter.cholesky_matrix
        num_assets = len(self.product_parameter.underlying_indices)

        if daily_steps:
            # Calculate days between start and end dates
            total_days = (self.end_date - self.start_date).days
            all_dates = [self.start_date + timedelta(days=d) for d in range(total_days + 1)]

            # Use optimized Numba function for daily paths
            paths = simulate_gbm_daily_paths_numba(
                initial_prices,
                volatilities,
                cholesky_matrix,
                len(all_dates),
                num_paths,
                seed
            )

            # For evaluation purposes, extract values at observation dates
            obs_indices = []
            for obs_date in self.product_parameter.observation_dates:
                idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
                obs_indices.append(idx)

            # Extract paths at observation dates (this part is not JIT-compiled)
            obs_paths = np.zeros((num_assets, num_paths, len(self.product_parameter.observation_dates)))
            for i, idx in enumerate(obs_indices):
                obs_paths[:, :, i] = paths[:, :, idx]

            return paths

        else:
            # Calculate time increments between observation dates in years
            observation_dates = self.product_parameter.observation_dates
            time_increments = np.zeros(len(observation_dates) - 1)

            for t in range(1, len(observation_dates)):
                time_increments[t - 1] = (observation_dates[t] - observation_dates[t - 1]).days / 365.0

            # Use optimized Numba function for observation date paths
            paths = simulate_gbm_observation_paths_numba(
                initial_prices,
                volatilities,
                cholesky_matrix,
                time_increments,
                num_paths,
                seed
            )

            return paths

    def calculate_payoff(self, paths):
        """
        Calculate Product 11 payoff from simulated paths
        This version uses Numba optimization for significant performance improvement.

        Args:
            paths: Simulated price paths

        Returns:
            dict: Dictionary with payoff results
        """
        num_indices = len(self.product_parameter.underlying_indices)

        # Use optimized Numba function
        final_payoffs, dividends = calculate_payoff_numba(
            paths,
            num_indices,
            self.product_parameter.minimum_guarantee,
            self.product_parameter.floor,
            self.product_parameter.cap,
            self.product_parameter.participation_rate,
            self.product_parameter.dividend_multiplier
        )

        # Calculate average results
        avg_payoff = np.mean(final_payoffs)
        avg_dividends = np.mean(dividends, axis=0)

        return {
            'final_payoff': avg_payoff,
            'dividends': avg_dividends,
            'total_payoff': avg_payoff + np.sum(avg_dividends)
        }

    def _extract_observation_paths(self, daily_paths, all_dates):
        """
        Extract paths at observation dates from daily paths.

        Args:
            daily_paths: The simulated daily price paths
            all_dates: List of all daily dates

        Returns:
            numpy.ndarray: Paths at observation dates only
        """
        observation_dates = self.product_parameter.observation_dates
        num_assets = daily_paths.shape[0]
        num_paths = daily_paths.shape[1]

        # Find indices of observation dates in the all_dates array
        obs_indices = []
        for obs_date in observation_dates:
            idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
            obs_indices.append(idx)

        # Extract paths at observation dates
        obs_paths = np.zeros((num_assets, num_paths, len(observation_dates)))
        for i, idx in enumerate(obs_indices):
            obs_paths[:, :, i] = daily_paths[:, :, idx]

        return obs_paths