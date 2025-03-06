# simulation_test.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from pythonProject.structured_product.product_manager.Product.parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import matplotlib.dates as mdates

class SimpleSimulator:
    """Simple Monte Carlo simulator for Product 11"""

    def __init__(self, start_date, end_date):
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        self.start_date = self.product_parameter.current_date
        self.end_date = self.product_parameter.final_date
        # Product parameters for Product 11
        self.risk_free_rate = 0.02  # 2% risk-free rate for calculations
        # Calculate observation dates - simplify by dividing total period into 5 parts
        total_days = (end_date - start_date).days


    def simulate_paths(self, num_paths=1000, seed=None, daily_steps=True):
        """Simulate price paths using GBM with correlations"""
        if seed is not None:
            np.random.seed(seed)
        current_date = self.market_data.current_date
        # Get initial prices
        S0 = np.array([self.market_data.get_price(code, current_date) * self.market_data.get_index_exchange_rate(code,current_date) for code in self.product_parameter.underlying_indices])

        # Get volatilities
        vols = np.array([self.product_parameter.volatilities[code] for code in self.product_parameter.underlying_indices])


        chol_matrix = self.product_parameter.cholesky_matrix

        n = len(self.product_parameter.underlying_indices)
        # Get Cholesky decomposition of correlation matrix
        if daily_steps:
            # Calculate days between start and end dates
            total_days = (self.end_date - self.start_date).days
            all_dates = [self.start_date + timedelta(days=d) for d in range(total_days + 1)]

            # Initialize paths container
            paths = np.zeros((n, num_paths, len(all_dates)))
            # Set initial prices
            for i in range(n):
                paths[i, :, 0] = S0[i]

            # Simulate daily paths
            for t in range(1, len(all_dates)):
                dt = 1 / 252  # Daily time step (assuming 252 trading days per year)

                # Generate correlated random shocks
                Z = np.random.standard_normal((n, num_paths))
                correlated_Z = chol_matrix @ Z

                # Apply GBM formula with correlation
                for i in range(n):
                    drift = (0.0 - 0.5 * vols[i] ** 2) * dt  # Assume 0 drift for simplicity
                    diffusion = vols[i] * np.sqrt(dt) * correlated_Z[i, :]
                    paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

            # For evaluation purposes, also extract values at observation dates
            obs_indices = []
            for obs_date in self.product_parameter.observation_dates:
                idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
                obs_indices.append(idx)

            # Extract paths at observation dates for evaluation
            obs_paths = np.zeros((n, num_paths, len(self.product_parameter.observation_dates)))
            for i, idx in enumerate(obs_indices):
                obs_paths[:, :, i] = paths[:, :, idx]

            # Return full daily paths for visualization
            return paths
        else:
            # Number of observation dates
            num_dates = len(self.product_parameter.observation_dates)

            # Initialize paths container
            paths = np.zeros((n, num_paths, num_dates))

            # Set initial prices
            for i in range(n):
                paths[i, :, 0] = S0[i]

            # Simulate paths between each observation date
            for t in range(1, num_dates):
                dt = (self.product_parameter.observation_dates[t] - self.product_parameter.observation_dates[t - 1]).days / 365.0

                # Generate correlated random shocks
                Z = np.random.standard_normal((n, num_paths))
                correlated_Z = chol_matrix @ Z

                # Apply GBM formula with correlation
                for i in range(n):
                    drift = (0.0 - 0.5 * vols[i] ** 2) * dt  # Assume 0 drift for simplicity
                    diffusion = vols[i] * np.sqrt(dt) * correlated_Z[i, :]
                    paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)
            print(paths)
            return paths

    def calculate_payoff(self, paths):
        """Calculate Product 11 payoff from simulated paths"""
        num_indices = len(self.indices)
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
                if basket_perf >= self.min_guarantee:
                    threshold_reached = True

                # Find best performing index for dividend
                if perfs:
                    best_idx, best_perf = max(perfs, key=lambda x: x[1])

                    # Calculate dividend
                    dividends[p, t - 1] = max(0, self.dividend_multiplier * best_perf)

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
                basket_final_perf = max(basket_final_perf, self.floor)
            else:
                basket_final_perf = min(basket_final_perf, self.cap)

            # Apply minimum guarantee if threshold reached
            if threshold_reached:
                basket_final_perf = max(basket_final_perf, self.min_guarantee)

            # Apply participation rate
            final_payoffs[p] = 1000.0 * (1.0 + self.participation_rate * basket_final_perf)

        # Calculate average results
        avg_payoff = np.mean(final_payoffs)
        avg_dividends = np.mean(dividends, axis=0)

        return {
            'final_payoff': avg_payoff,
            'dividends': avg_dividends,
            'total_payoff': avg_payoff + np.sum(avg_dividends)
        }

    def calculate_deltas(self, bump_percent=0.01):
        """Calculate deltas by bumping each index price"""
        # Run base case simulation
        np.random.seed(42)  # For reproducibility

        # For delta calculations, we can use observation-date paths
        daily_paths = self.simulate_paths(num_paths=1000, daily_steps=True)

        # Extract observation-date paths for calculation
        obs_indices = []
        total_days = (self.end_date - self.start_date).days
        all_dates = [self.start_date + timedelta(days=d) for d in range(total_days + 1)]

        for obs_date in self.observation_dates:
            idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
            obs_indices.append(idx)

        # Create observation-date paths for payoff calculation
        n = len(self.indices)
        num_paths = daily_paths.shape[1]
        base_paths = np.zeros((n, num_paths, len(self.observation_dates)))

        for i, idx in enumerate(obs_indices):
            base_paths[:, :, i] = daily_paths[:, :, idx]

        # Calculate base case payoff
        try:
            base_result = self.calculate_payoff(base_paths)
            base_total_payoff = base_result['total_payoff']
        except Exception as e:
            print(f"Error in base payoff calculation: {e}")
            # Provide a fallback delta calculation
            return {idx: 0.0 for idx in self.indices}

        # Calculate deltas for each index
        deltas = {}
        for i, idx in enumerate(self.indices):
            try:
                # Bump initial price of this index
                bump_paths = base_paths.copy()
                bump_paths[i, :, 0] *= (1 + bump_percent)

                # Recalculate paths after bump
                # Only need to recalculate the specific index that was bumped
                S0_bumped = bump_paths[i, :, 0]  # Save bumped initial values

                # Get volatility for this index
                vol = self.market_data.get_volatility(idx, self.start_date)

                # Recalculate future values based on bumped initial value
                for t in range(1, bump_paths.shape[2]):
                    dt = (self.observation_dates[t] - self.observation_dates[t - 1]).days / 365.0

                    # Generate random shocks
                    Z = np.random.standard_normal(num_paths)

                    # Apply GBM formula
                    drift = (0.0 - 0.5 * vol ** 2) * dt
                    diffusion = vol * np.sqrt(dt) * Z

                    # Update paths
                    if t == 1:
                        # First step uses bumped initial value
                        bump_paths[i, :, t] = S0_bumped * np.exp(drift + diffusion)
                    else:
                        # Subsequent steps use previous values
                        bump_paths[i, :, t] = bump_paths[i, :, t - 1] * np.exp(drift + diffusion)

                # Calculate payoff with bumped index
                bump_result = self.calculate_payoff(bump_paths)
                bump_total_payoff = bump_result['total_payoff']

                # Calculate delta
                price_change = self.market_data.get_price(idx, self.start_date) * bump_percent
                if price_change != 0:
                    payoff_change = bump_total_payoff - base_total_payoff
                    deltas[idx] = payoff_change / price_change
                else:
                    deltas[idx] = 0.0

            except Exception as e:
                print(f"Error calculating delta for {idx}: {e}")
                deltas[idx] = 0.0

        return deltas

    def calculate_portfolio(self, initial_investment=1000.0):
        """Calculate portfolio composition using delta-hedging"""
        try:
            # Calculate deltas for each index
            deltas = self.calculate_deltas()

            # Determine the amount to invest in each index and in risk-free asset
            total_investment = 0
            portfolio = {}

            for idx in self.indices:
                # Calculate amount needed for delta hedging
                delta = deltas[idx]
                idx_price = self.market_data.get_price(idx, self.start_date)
                if idx_price is None or idx_price <= 0:
                    print(f"Warning: Invalid price for {idx}. Using default price of 100.")
                    idx_price = 100.0

                investment_amount = delta * idx_price

                # Store in portfolio
                portfolio[idx] = {
                    'delta': delta,
                    'price': idx_price,
                    'investment': investment_amount,
                    'quantity': investment_amount / idx_price
                }

                total_investment += abs(investment_amount)

            # Calculate how much to invest in risk-free asset
            risk_free_investment = initial_investment - total_investment

            # Add risk-free investment to portfolio
            portfolio['risk_free'] = {
                'rate': self.risk_free_rate,
                'investment': risk_free_investment
            }

            return portfolio

        except Exception as e:
            print(f"Error calculating portfolio: {e}")
            # Return a default portfolio
            default_portfolio = {idx: {'delta': 0.0, 'price': 100.0, 'investment': 0.0, 'quantity': 0.0}
                                 for idx in self.indices}
            default_portfolio['risk_free'] = {'rate': self.risk_free_rate, 'investment': initial_investment}
            return default_portfolio

    def visualize_paths(self, paths=None, num_to_show=3, daily_paths=True):
        """
        Visualize historical and simulated paths for each index

        Args:
            paths: Simulated price paths (optional, will simulate if None)
            num_to_show: Number of simulated paths to show per index
            daily_paths: Whether to use daily simulation points (True) or just observation dates (False)
        """
        # If no paths provided, simulate them
        if paths is None:
            paths = self.simulate_paths(num_paths=num_to_show + 1, daily_steps=daily_paths)

        # Create a figure with subplots for each index
        fig, axes = plt.subplots(len(self.indices), 1, figsize=(12, 4 * len(self.indices)))
        if len(self.indices) == 1:
            axes = [axes]  # Make sure axes is a list for single index case

        # Format dates
        date_fmt = DateFormatter('%Y-%m')

        # Get historical dates and prices (1 year before start)
        hist_start = self.start_date - timedelta(days=365)
        historical_dates = []
        current_date = hist_start
        while current_date <= self.start_date:
            if current_date in self.market_data.dates:
                historical_dates.append(current_date)
            current_date += timedelta(days=1)

        # Create simulation dates - one for each day
        total_days = (self.end_date - self.start_date).days
        sim_dates = [self.start_date + timedelta(days=d) for d in range(total_days + 1)]

        # Re-simulate with daily data if needed
        if daily_paths and paths.shape[2] <= len(self.observation_dates):
            print("Simulating daily paths for visualization...")
            daily_sim = self.simulate_paths(num_paths=num_to_show + 1, daily_steps=True)

            # Extract all indices at all dates
            n_indices = len(self.indices)
            daily_paths = daily_sim
        else:
            daily_paths = paths

        # Plot each index
        for i, idx in enumerate(self.indices):
            ax = axes[i]

            # Get historical prices
            historical_prices = [self.market_data.get_price(idx, date) for date in historical_dates]

            # Plot historical data
            ax.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical')

            # Mark the start date
            ax.axvline(x=self.start_date, color='r', linestyle='--', label='Start Date (T0)')

            # Plot observation dates with markers
            for obs_date in self.observation_dates:
                ax.axvline(x=obs_date, color='gray', linestyle=':', alpha=0.5)

            # Create plotable dates
            if daily_paths.shape[2] <= len(self.observation_dates):
                # Using sparse dates (observation dates only)
                plot_dates = self.observation_dates
            else:
                # Using daily dates
                plot_dates = sim_dates

            # Plot simulated paths
            for p in range(min(num_to_show, daily_paths.shape[1])):
                path_prices = daily_paths[i, p, :]
                if len(path_prices) == len(plot_dates):
                    ax.plot(plot_dates, path_prices, 'g-', alpha=0.5)
                else:
                    print(f"Warning: Path length ({len(path_prices)}) doesn't match plot dates ({len(plot_dates)})")

            # Plot average simulated path
            avg_path = np.mean(daily_paths[i, :, :], axis=0)
            ax.plot(plot_dates, avg_path, 'r-', linewidth=2, label='Average Simulation')

            # Set title and labels
            ax.set_title(f'{idx} Price Path', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)

            # Format x-axis
            ax.xaxis.set_major_formatter(date_fmt)
            ax.xaxis.set_major_locator(MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            # Add legend
            ax.legend()

            # Add grid
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('index_simulations.png')
        plt.show()


def validate_simulation(simulator):
    """
    Run multiple validation checks on the Product 11 simulation
    to verify its correctness.
    """
    print("\n=== VALIDATION TESTS ===\n")

    # 1. Check convergence with increasing number of paths
    convergence_test(simulator)

    # 2. Test with predefined scenarios
    predefined_scenarios_test(simulator)

    # 3. Print detailed breakdown of a sample path
    sample_path_analysis(simulator)

    # 4. Verify delta hedging effectiveness
    verify_delta_hedging(simulator)

    print("\n=== VALIDATION COMPLETE ===\n")


def convergence_test(simulator, path_counts=[100, 500, 1000, 5000, 10000]):
    """Test if results converge as the number of paths increases."""
    print("Running convergence test...")
    results = []

    for num_paths in path_counts:
        # Set seed for reproducibility
        np.random.seed(42)

        # Run simulation with increasing path counts
        paths = simulator.simulate_paths(num_paths=num_paths, daily_steps=True)

        # Extract observation dates for payoff calculation
        n = len(simulator.indices)
        obs_indices = []
        total_days = (simulator.end_date - simulator.start_date).days
        all_dates = [simulator.start_date + timedelta(days=d) for d in range(total_days + 1)]

        for obs_date in simulator.observation_dates:
            idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
            obs_indices.append(idx)

        # Extract paths at observation dates
        obs_paths = np.zeros((n, num_paths, len(simulator.observation_dates)))
        for i, idx in enumerate(obs_indices):
            obs_paths[:, :, i] = paths[:, :, idx]

        result = simulator.calculate_payoff(obs_paths)
        results.append({
            'num_paths': num_paths,
            'final_payoff': result['final_payoff'],
            'total_payoff': result['total_payoff'],
            'dividends': result['dividends']
        })

    # Print results
    print("\nConvergence Results:")
    print(f"{'Paths':>10} | {'Final Payoff':>15} | {'Total Payoff':>15} | {'Std Dev %':>10}")
    print("-" * 60)

    payoffs = [r['total_payoff'] for r in results]
    for i, r in enumerate(results):
        std_dev_pct = 0
        if i > 0:
            # Calculate standard deviation as percentage of mean for last two runs
            std_dev_pct = abs(payoffs[i] - payoffs[i - 1]) / payoffs[i] * 100

        print(f"{r['num_paths']:>10} | {r['final_payoff']:>15.2f} | {r['total_payoff']:>15.2f} | {std_dev_pct:>10.2f}")

    # Check if results converge (last two runs should be within 1%)
    if abs(payoffs[-1] - payoffs[-2]) / payoffs[-1] < 0.01:
        print("\n✓ Results converged (difference < 1%)")
    else:
        print("\n✗ Results did not converge (difference >= 1%)")


def predefined_scenarios_test(simulator):
    """Test with predefined scenarios with known outcomes."""
    print("\nTesting with predefined scenarios...")

    # Create test scenarios
    scenarios = [
        {
            'name': "All Up 30%",
            'description': "All indices rise by 30% by the end, 5% each year",
            'annual_returns': [1.05, 1.05, 1.05, 1.05, 1.05],  # Compound to ~27%
            'expected_final_payoff_range': (1000 * (1 + 0.4 * 0.3), 1000 * (1 + 0.4 * 0.3)),
            'min_dividend': 50 * 0.05  # Minimum dividend from 5% growth
        },
        {
            'name': "All Down 20%",
            'description': "All indices fall by 20%, 5% each year",
            'annual_returns': [0.95, 0.95, 0.95, 0.95, 0.95],  # Compound to ~-23%
            'expected_final_payoff_range': (1000 * (1 + 0.4 * -0.15), 1000 * (1 + 0.4 * -0.15)),  # Floor at -15%
            'max_dividend': 0  # No dividend for negative performance
        },
        {
            'name': "Trigger 20% Guarantee",
            'description': "One observation with 25% increase triggers guarantee",
            'annual_returns': [1.25, 0.9, 0.9, 0.9, 0.9],  # One year with 25% increase
            'expected_final_payoff_range': (1000 * (1 + 0.4 * 0.2), 1000 * (1 + 0.4 * 0.2)),  # Guarantee activated
            'min_dividend': 50 * 0.25  # First dividend based on 25% increase
        }
    ]

    # Test each scenario
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")

        # Create artificial paths matching scenario
        n = len(simulator.indices)
        num_paths = 100
        num_dates = len(simulator.observation_dates)

        paths = np.zeros((n, num_paths, num_dates))

        # Set initial values (T0)
        for i in range(n):
            idx = simulator.indices[i]
            price = simulator.market_data.get_price(idx, simulator.start_date)
            paths[i, :, 0] = price

        # Generate specific price paths based on scenario
        for t in range(1, num_dates):
            for i in range(n):
                paths[i, :, t] = paths[i, :, t - 1] * scenario['annual_returns'][t - 1]

        # Calculate payoff for scenario
        result = simulator.calculate_payoff(paths)

        # Check if results match expectations
        expected_min, expected_max = scenario['expected_final_payoff_range']
        within_range = expected_min <= result['final_payoff'] <= expected_max

        print(f"Expected final payoff: {expected_min:.2f} to {expected_max:.2f}")
        print(f"Actual final payoff: {result['final_payoff']:.2f}")
        print(f"Total payoff: {result['total_payoff']:.2f}")
        print(f"Dividends: {[f'€{d:.2f}' for d in result['dividends']]}")

        if within_range:
            print(f"✓ Final payoff within expected range")
        else:
            print(f"✗ Final payoff outside expected range")

        # Additional checks based on scenario
        if 'min_dividend' in scenario and scenario['min_dividend'] > 0:
            max_dividend = max(result['dividends'])
            if max_dividend >= scenario['min_dividend']:
                print(f"✓ Maximum dividend (€{max_dividend:.2f}) >= expected minimum (€{scenario['min_dividend']:.2f})")
            else:
                print(f"✗ Maximum dividend (€{max_dividend:.2f}) < expected minimum (€{scenario['min_dividend']:.2f})")

        if 'max_dividend' in scenario and scenario['max_dividend'] == 0:
            if sum(result['dividends']) == 0:
                print(f"✓ No dividends paid as expected")
            else:
                print(f"✗ Dividends paid when none expected")


def sample_path_analysis(simulator):
    """Analyze a single sample path in detail to verify calculations."""
    print("\nAnalyzing a sample path...")

    # Generate a single path with a fixed seed
    np.random.seed(123)

    # Create a simple deterministic path for easier verification
    n = len(simulator.indices)
    num_dates = len(simulator.observation_dates)

    # Create a single path manually
    path = np.zeros((n, 1, num_dates))

    # Set initial prices at T0
    for i in range(n):
        idx = simulator.indices[i]
        price = simulator.market_data.get_price(idx, simulator.start_date)
        path[i, 0, 0] = price

    # Create a scenario with different performance for each index
    annual_returns = [
        [1.08, 1.03, -0.02, 0.05, 0.07],  # Index 0
        [0.12, -0.05, 0.08, 0.04, 0.06],  # Index 1
        [-0.03, 0.09, 0.11, -0.01, 0.08],  # Index 2
        [0.05, 0.07, 0.03, 0.10, 0.04],  # Index 3
        [0.10, 0.04, 0.06, 0.02, 0.09]  # Index 4
    ]

    # Apply returns
    for t in range(1, num_dates):
        for i in range(n):
            # Calculate price based on annual return
            path[i, 0, t] = path[i, 0, 0] * (1 + annual_returns[i][t - 1])

    # Print path details
    print("\nPrice Path:")
    print(f"{'Date':>12} | ", end="")
    for i in range(n):
        print(f"{simulator.indices[i]:>10} | ", end="")
    print()
    print("-" * (12 + (n * 13)))

    for t in range(num_dates):
        date_str = simulator.observation_dates[t].strftime('%Y-%m-%d')
        print(f"{date_str:>12} | ", end="")
        for i in range(n):
            print(f"{path[i, 0, t]:>10.2f} | ", end="")
        print()

    # Calculate and print annual performances
    print("\nAnnual Index Performances:")
    print(f"{'Period':>12} | ", end="")
    for i in range(n):
        print(f"{simulator.indices[i]:>10} | ", end="")
    print("Basket Perf | Best Index")
    print("-" * (12 + (n * 13) + 25))

    dividend_excluded = []

    for t in range(1, num_dates):
        period = f"{t}"
        print(f"{period:>12} | ", end="")

        performances = []
        for i in range(n):
            perf = (path[i, 0, t] / path[i, 0, t - 1]) - 1
            performances.append(perf)
            print(f"{perf * 100:>10.2f}% | ", end="")

        # Calculate basket performance
        basket_perf = sum(performances) / len(performances)
        print(f"{basket_perf * 100:>10.2f}% | ", end="")

        # Find best performer for dividend (excluding already selected)
        dividend_eligible = [(i, perf) for i, perf in enumerate(performances) if i not in dividend_excluded]
        if dividend_eligible:
            best_idx, best_perf = max(dividend_eligible, key=lambda x: x[1])
            print(f"{simulator.indices[best_idx]} ({best_perf * 100:.2f}%)")
            dividend_excluded.append(best_idx)
        else:
            print("None left")

    # Calculate final product performance
    print("\nFinal Product Performance Calculation:")

    # Calculate total performance from T0 to Tc
    total_performances = [(path[i, 0, -1] / path[i, 0, 0]) - 1 for i in range(n)]
    avg_total_perf = sum(total_performances) / len(total_performances)

    print(f"Total index performances: {[f'{p * 100:.2f}%' for p in total_performances]}")
    print(f"Average basket performance: {avg_total_perf * 100:.2f}%")

    # Apply floor/cap
    if avg_total_perf < 0:
        capped_perf = max(avg_total_perf, simulator.floor)
        print(f"After floor: {capped_perf * 100:.2f}%")
    else:
        capped_perf = min(avg_total_perf, simulator.cap)
        print(f"After cap: {capped_perf * 100:.2f}%")

    # Check if minimum guarantee applies
    # Assuming we check each date's basket performance
    min_guarantee_triggered = False
    for t in range(1, num_dates):
        period_perfs = [(path[i, 0, t] / path[i, 0, t - 1]) - 1 for i in range(n)]
        period_basket_perf = sum(period_perfs) / len(period_perfs)
        if period_basket_perf >= simulator.min_guarantee:
            min_guarantee_triggered = True
            print(f"Minimum guarantee triggered at period {t} (perf: {period_basket_perf * 100:.2f}%)")
            break

    if min_guarantee_triggered:
        final_perf = max(capped_perf, simulator.min_guarantee)
        print(f"After minimum guarantee: {final_perf * 100:.2f}%")
    else:
        final_perf = capped_perf
        print("Minimum guarantee not triggered")

    # Calculate final payoff
    final_payoff = 1000 * (1 + simulator.participation_rate * final_perf)
    print(f"Final payoff after participation rate: €{final_payoff:.2f}")

    # Calculate dividends
    print("\nDividend Calculations:")
    dividends = []
    for t in range(1, num_dates - 1):
        period_perfs = [(path[i, 0, t] / path[i, 0, t - 1]) - 1 for i in range(n)]

        # Only consider non-excluded indices for dividend
        eligible_perfs = [(i, perf) for i, perf in enumerate(period_perfs)
                          if i not in dividend_excluded[:t - 1]]

        if eligible_perfs:
            best_idx, best_perf = max(eligible_perfs, key=lambda x: x[1])
            dividend = max(0, simulator.dividend_multiplier * best_perf)
            dividends.append(dividend)
            print(f"Period {t}: Best index {simulator.indices[best_idx]} " +
                  f"(perf: {best_perf * 100:.2f}%) → Dividend: €{dividend:.2f}")
        else:
            dividends.append(0)
            print(f"Period {t}: No eligible indices → Dividend: €0.00")

    # Calculate total payoff
    total_payoff = final_payoff + sum(dividends)
    print(f"\nTotal payoff: €{total_payoff:.2f} (Final: €{final_payoff:.2f} + Dividends: €{sum(dividends):.2f})")

    # Run the simulator with this path and compare
    result = simulator.calculate_payoff(path)
    print("\nSimulator results for comparison:")
    print(f"Final payoff: €{result['final_payoff']:.2f}")
    print(f"Dividends: {[f'€{d:.2f}' for d in result['dividends']]}")
    print(f"Total payoff: €{result['total_payoff']:.2f}")

    # Check if manual calculation matches simulator
    if abs(total_payoff - result['total_payoff']) < 0.01:
        print("✓ Manual calculation matches simulator result")
    else:
        print("✗ Manual calculation differs from simulator result")


def verify_delta_hedging(simulator):
    """Verify that delta hedging works by testing portfolio replication."""
    print("\nVerifying delta hedging effectiveness...")

    # Calculate initial deltas and portfolio
    deltas = simulator.calculate_deltas()
    portfolio = simulator.calculate_portfolio(initial_investment=1000.0)

    print("\nInitial Deltas:")
    for idx, delta in deltas.items():
        print(f"{idx}: {delta:.4f}")

    print("\nPortfolio Composition:")
    for idx in simulator.indices:
        print(f"{idx}: €{portfolio[idx]['investment']:.2f} ({portfolio[idx]['quantity']:.6f} units)")
    print(f"Risk-free: €{portfolio['risk_free']['investment']:.2f}")

    # Create a simple test: move all prices up by 10%
    # This should increase the product value by approximately sum(delta_i * price_i * 10%)

    # First run base case
    np.random.seed(42)
    base_paths = simulator.simulate_paths(num_paths=1000, daily_steps=True)

    # Extract observation date paths
    n = len(simulator.indices)
    num_paths = base_paths.shape[1]
    obs_indices = []
    total_days = (simulator.end_date - simulator.start_date).days
    all_dates = [simulator.start_date + timedelta(days=d) for d in range(total_days + 1)]

    for obs_date in simulator.observation_dates:
        idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
        obs_indices.append(idx)

    # Extract paths at observation dates
    obs_paths = np.zeros((n, num_paths, len(simulator.observation_dates)))
    for i, idx in enumerate(obs_indices):
        obs_paths[:, :, i] = base_paths[:, :, idx]

    base_result = simulator.calculate_payoff(obs_paths)
    base_value = base_result['total_payoff']

    # Now create bumped paths (all prices up 10%)
    bump_factor = 1.10
    bump_paths = obs_paths.copy()

    # Bump initial prices
    for i in range(n):
        bump_paths[i, :, 0] *= bump_factor

    # Recalculate future values
    for i in range(n):
        idx = simulator.indices[i]
        vol = simulator.market_data.get_volatility(idx, simulator.start_date)

        for t in range(1, bump_paths.shape[2]):
            dt = (simulator.observation_dates[t] - simulator.observation_dates[t - 1]).days / 365.0

            for p in range(bump_paths.shape[1]):
                drift = (0.0 - 0.5 * vol ** 2) * dt
                diffusion = np.random.normal(0, np.sqrt(dt) * vol)
                bump_paths[i, p, t] = bump_paths[i, p, t - 1] * np.exp(drift + diffusion)

    # Calculate bumped result
    bump_result = simulator.calculate_payoff(bump_paths)
    bump_value = bump_result['total_payoff']

    # Calculate expected change based on deltas
    delta_based_change = 0
    for i, idx in enumerate(simulator.indices):
        price = simulator.market_data.get_price(idx, simulator.start_date)
        delta = deltas[idx]
        delta_based_change += delta * price * 0.1  # 10% price change

    # Calculate actual change
    actual_change = bump_value - base_value

    print(f"\nBase product value: €{base_value:.2f}")
    print(f"Value after 10% price increase: €{bump_value:.2f}")
    print(f"Actual change: €{actual_change:.2f}")
    print(f"Expected change based on deltas: €{delta_based_change:.2f}")

    # Check if delta prediction is within 20% of actual change
    accuracy = (1 - abs(actual_change - delta_based_change) / abs(actual_change)) * 100
    print(f"Delta prediction accuracy: {accuracy:.2f}%")

    if accuracy >= 80:
        print("✓ Delta hedging is effective (accuracy >= 80%)")
    else:
        print("✗ Delta hedging appears inaccurate (accuracy < 80%)")


def test_simulation():
    """Test the simulation on Product 11"""
    # Path to data file
    file_path = "DonneesGPS2025.xlsx"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    # Initialize market data
    market_data = MarketDataSimple(file_path)

    # Indices for Product 11
    indices = ['ASX200', 'DAX', 'FTSE100', 'NASDAQ100', 'SMI']

    # Get dates from data
    start_date = datetime(2009, 1, 5)
    end_date = datetime(2014, 1, 6)

    # Initialize simulator
    simulator = SimpleSimulator(market_data, indices, start_date, end_date)

    # First generate visualizations with daily paths (uses fewer simulations)
    print("Generating visualizations with daily paths...")
    # This will automatically generate daily paths
    simulator.visualize_paths(paths=None, num_to_show=3, daily_paths=True)

    # Then simulate paths for accurate calculations (more simulations for precision)
    print("Simulating price paths for payoff and delta calculations...")
    # We need observation-date paths for payoff calculation
    # Extract the observation dates from daily paths
    daily_paths = simulator.simulate_paths(num_paths=1000, seed=42, daily_steps=True)

    # Calculate payoff using observation dates extracted from daily paths
    print("Calculating payoff...")
    # Extract paths at observation dates for evaluation
    n = len(simulator.indices)
    num_paths = daily_paths.shape[1]
    obs_indices = []

    total_days = (end_date - start_date).days
    all_dates = [start_date + timedelta(days=d) for d in range(total_days + 1)]

    for obs_date in simulator.observation_dates:
        idx = next((i for i, d in enumerate(all_dates) if d >= obs_date), len(all_dates) - 1)
        obs_indices.append(idx)

    # Extract paths at observation dates for evaluation
    obs_paths = np.zeros((n, num_paths, len(simulator.observation_dates)))
    for i, idx in enumerate(obs_indices):
        obs_paths[:, :, i] = daily_paths[:, :, idx]

    result = simulator.calculate_payoff(obs_paths)

    print("\nProduct 11 Simulation Results:")
    print(f"Final Payoff: €{result['final_payoff']:.2f}")
    print(f"Dividends: {[f'€{d:.2f}' for d in result['dividends']]}")
    print(f"Total Payoff: €{result['total_payoff']:.2f}")

    # Calculate deltas
    print("\nCalculating deltas...")
    deltas = simulator.calculate_deltas()

    print("\nDelta Sensitivities:")
    for idx, delta in deltas.items():
        print(f"{idx}: {delta:.4f}")

    # Calculate portfolio composition
    print("\nCalculating portfolio composition...")
    portfolio = simulator.calculate_portfolio(initial_investment=1000.0)

    print("\nRecommended Portfolio Composition:")
    print("Risky Assets:")
    for idx in indices:
        print(f"{idx}: {portfolio[idx]['investment']:.2f}€ ({portfolio[idx]['quantity']:.6f} units)")

    print(
        f"\nRisk-Free Asset: {portfolio['risk_free']['investment']:.2f}€ at {portfolio['risk_free']['rate'] * 100:.2f}%")

    # Verify that total investment equals initial investment
    total_investment = sum(abs(portfolio[idx]['investment']) for idx in indices) + portfolio['risk_free']['investment']
    print(f"\nTotal Investment: {total_investment:.2f}€")
    validate_simulation(simulator)


if __name__ == "__main__":
    test_simulation()