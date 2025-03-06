# simulation_test.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from ..Product.parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates


class SimpleSimulator:
    """Simple Monte Carlo simulator for Product 11"""

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

        Args:
            num_paths (int): Number of simulation paths to generate
            seed (int, optional): Random seed for reproducibility
            daily_steps (bool): If True, simulate with daily time steps, otherwise use observation dates only

        Returns:
            numpy.ndarray: Simulated price paths
        """
        if seed is not None:
            np.random.seed(seed)

        # Setup initial parameters
        initial_prices = self._get_initial_prices()
        volatilities = self._get_volatilities()
        cholesky_matrix = self.product_parameter.cholesky_matrix
        num_assets = len(self.product_parameter.underlying_indices)

        if daily_steps:
            return self._simulate_daily_paths(num_assets, num_paths, initial_prices, volatilities, cholesky_matrix)
        else:
            return self._simulate_observation_date_paths(
                num_assets, num_paths, initial_prices, volatilities, cholesky_matrix)


    def _simulate_daily_paths(self, num_assets, num_paths, initial_prices, volatilities, cholesky_matrix):
        """
        Simulate price paths with daily time steps.

        Returns:
            numpy.ndarray: Price paths with shape (num_assets, num_paths, num_days)
        """
        # Generate daily dates between start and end
        total_days = (self.end_date - self.start_date).days
        all_dates = [self.start_date + timedelta(days=d) for d in range(total_days + 1)]

        # Initialize paths array
        paths = np.zeros((num_assets, num_paths, len(all_dates)))

        # Set initial prices
        for i in range(num_assets):
            paths[i, :, 0] = initial_prices[i]

        # Simulate daily paths
        dt = 1 / 252  # Daily time step (assuming 252 trading days per year)

        for t in range(1, len(all_dates)):
            # Generate correlated random shocks
            random_shocks = np.random.standard_normal((num_assets, num_paths))
            correlated_shocks = cholesky_matrix @ random_shocks

            # Apply GBM formula with correlation
            for i in range(num_assets):
                drift = (0.0 - 0.5 * volatilities[i] ** 2) * dt  # Assume 0 drift for simplicity
                diffusion = volatilities[i] * np.sqrt(dt) * correlated_shocks[i, :]
                paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

        # Extract paths at observation dates for evaluation (if needed)
        obs_paths = self._extract_observation_paths(paths, all_dates)

        return paths

    def _simulate_observation_date_paths(self, num_assets, num_paths, initial_prices, volatilities, cholesky_matrix):
        """
        Simulate price paths with jumps directly to observation dates.

        Returns:
            numpy.ndarray: Price paths with shape (num_assets, num_paths, num_observation_dates)
        """
        observation_dates = self.product_parameter.observation_dates
        num_dates = len(observation_dates)

        # Initialize paths container
        paths = np.zeros((num_assets, num_paths, num_dates))

        # Set initial prices
        for i in range(num_assets):
            paths[i, :, 0] = initial_prices[i]

        # Simulate paths between each observation date
        for t in range(1, num_dates):
            # Calculate time increment in years
            dt = (observation_dates[t] - observation_dates[t - 1]).days / 365.0

            # Generate correlated random shocks
            random_shocks = np.random.standard_normal((num_assets, num_paths))
            correlated_shocks = cholesky_matrix @ random_shocks

            # Apply GBM formula with correlation
            for i in range(num_assets):
                drift = (0.0 - 0.5 * volatilities[i] ** 2) * dt  # Assume 0 drift for simplicity
                diffusion = volatilities[i] * np.sqrt(dt) * correlated_shocks[i, :]
                paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

        return paths

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

    def calculate_payoff(self, paths):
        """Calculate Product 11 payoff from simulated paths"""
        num_indices = len(self.product_parameter.underlying_indices)
        num_paths = paths.shape[1]
        num_dates = paths.shape[2]

        # Container for results
        final_payoffs = np.zeros(num_paths)
        dividends = np.zeros((num_paths, num_dates - 2))  # Exclude T0 and Tc

        # Process each path
        for p in range(num_paths):
            # For tracking excluded indices (only for dividend calculation)
            dividend_excluded_indices = []
            threshold_reached = False

            # Process each observation date (skip first and last)
            for t in range(1, num_dates - 1):
                # Calculate annual performance for each index
                perfs = []
                all_perfs = []  # For all indices (used for threshold check)

                # Calculate performance for all indices
                for i in range(num_indices):
                    perf = paths[i, p, t] / paths[i, p, t - 1] - 1.0
                    all_perfs.append(perf)

                    # Only add to perfs if not excluded for dividend
                    if i not in dividend_excluded_indices:
                        perfs.append((i, perf))

                # Check if any performance triggers minimum guarantee
                basket_perf = sum(all_perfs) / len(all_perfs)
                if basket_perf >= self.product_parameter.minimum_guarantee:
                    threshold_reached = True

                # Find best performing index for dividend
                if perfs:
                    best_idx, best_perf = max(perfs, key=lambda x: x[1])

                    # Calculate dividend
                    dividends[p, t - 1] = max(0, self.product_parameter.dividend_multiplier * best_perf)

                    # Add to excluded indices for future dividend calculations
                    dividend_excluded_indices.append(best_idx)
                else:
                    # No active indices left, so no dividend
                    dividends[p, t - 1] = 0.0

            # Calculate final performance at maturity using ALL INDICES
            # (even those excluded from dividend calculations)
            final_perfs = [(i, paths[i, p, -1] / paths[i, p, 0] - 1.0) for i in range(num_indices)]
            basket_final_perf = sum(perf for _, perf in final_perfs) / len(final_perfs)

            # Apply floors and caps
            if basket_final_perf < 0:
                basket_final_perf = max(basket_final_perf, self.product_parameter.floor)
            else:
                basket_final_perf = min(basket_final_perf, self.product_parameter.cap)

            # Apply minimum guarantee if threshold reached
            if threshold_reached:
                basket_final_perf = max(basket_final_perf, self.product_parameter.minimum_guarantee)

            # Apply participation rate
            final_payoffs[p] = 1000.0 * (1.0 + self.product_parameter.participation_rate * basket_final_perf)

        # Calculate average results
        avg_payoff = np.mean(final_payoffs)
        avg_dividends = np.mean(dividends, axis=0)

        return {
            'final_payoff': avg_payoff,
            'dividends': avg_dividends,
            'total_payoff': avg_payoff + np.sum(avg_dividends)
        }
