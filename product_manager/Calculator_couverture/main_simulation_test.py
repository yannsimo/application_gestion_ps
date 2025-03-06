#!/usr/bin/env python
# main_simulation_test.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

# Adjust import path as needed based on your project structure
sys.path.append('..')  # Add parent directory to path if needed

# Import the SimpleSimulator class
from testr import SimpleSimulator


def plot_simulation_paths(simulator, paths, indices, num_paths_to_plot=10):
    """
    Plot a subset of simulated price paths for visual inspection.

    Args:
        simulator: The SimpleSimulator instance
        paths: The simulated price paths
        indices: List of index names
        num_paths_to_plot: Number of random paths to plot
    """
    # Generate daily dates between start and end
    total_days = (simulator.end_date - simulator.start_date).days
    all_dates = [simulator.start_date + timedelta(days=d) for d in range(total_days + 1)]

    # Create subplots for each index
    fig, axes = plt.subplots(len(indices), 1, figsize=(12, 3 * len(indices)), sharex=True)
    if len(indices) == 1:
        axes = [axes]  # Make sure axes is a list even for a single subplot

    # Randomly select paths to plot
    random_paths = np.random.choice(paths.shape[1], num_paths_to_plot, replace=False)

    for i, index_name in enumerate(indices):
        ax = axes[i]

        # Initial price for normalization
        initial_price = paths[i, 0, 0]

        # Plot each selected path
        for path_idx in random_paths:
            normalized_path = paths[i, path_idx, :] / initial_price
            ax.plot(all_dates, normalized_path, alpha=0.5)

        # Set title and labels
        ax.set_title(f"{index_name} - Normalized Price Paths")
        ax.set_ylabel("Normalized Price")
        ax.grid(True, linestyle='--', alpha=0.7)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.savefig("simulation_paths.png", dpi=300)
    plt.show()


def plot_payoff_distribution(payoffs):
    """
    Plot the distribution of final payoffs.

    Args:
        payoffs: Array of final payoffs for each simulation path
    """
    plt.figure(figsize=(10, 6))
    plt.hist(payoffs, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=np.mean(payoffs), color='red', linestyle='--',
                label=f'Mean: {np.mean(payoffs):.2f}')
    plt.axvline(x=np.median(payoffs), color='green', linestyle='-',
                label=f'Median: {np.median(payoffs):.2f}')

    plt.title('Distribution of Final Payoffs')
    plt.xlabel('Payoff Amount')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("payoff_distribution.png", dpi=300)
    plt.show()


def main():
    """Main function to run the simulation test"""
    print("Starting Product Simulation Test")

    # Set start and end dates
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2028, 1, 1)  # 5-year product

    # Create simulator
    simulator = SimpleSimulator(start_date, end_date)

    # Print basic product information
    print(f"Product Details:")
    print(f"Start Date: {simulator.start_date}")
    print(f"End Date: {simulator.end_date}")
    print(f"Underlying Indices: {simulator.product_parameter.underlying_indices}")
    print(f"Observation Dates: {[date.strftime('%Y-%m-%d') for date in simulator.product_parameter.observation_dates]}")
    print(f"Cap: {simulator.product_parameter.cap:.2%}")
    print(f"Floor: {simulator.product_parameter.floor:.2%}")
    print(f"Participation Rate: {simulator.product_parameter.participation_rate:.2f}")
    print(f"Minimum Guarantee: {simulator.product_parameter.minimum_guarantee:.2%}")

    # Run simulation
    print("\nRunning Monte Carlo simulation...")
    num_simulations = 10000
    paths = simulator.simulate_paths(num_paths=num_simulations, seed=42, daily_steps=True)
    print(f"Generated {num_simulations} simulation paths")

    # Calculate payoffs
    print("\nCalculating payoffs...")
    payoff_results = simulator.calculate_payoff(paths)

    # Print payoff statistics
    print("\nPayoff Statistics:")
    print(f"Average Final Payoff: {payoff_results['final_payoff']:.2f}")
    print(f"Average Dividends: {', '.join([f'{div:.2f}' for div in payoff_results['dividends']])}")
    print(f"Total Average Payoff: {payoff_results['total_payoff']:.2f}")

    # Calculate present value
    discount_factor = np.exp(-simulator.risk_free_rate * (simulator.end_date - simulator.start_date).days / 365)
    present_value = payoff_results['total_payoff'] * discount_factor
    print(f"Present Value: {present_value:.2f}")

    # Extract raw payoffs data for visualization
    # Note: This assumes calculate_payoff returns raw payoffs as well,
    # which might require modifying that function

    # For demonstration, let's create a synthetic array of final payoffs
    # In practice, you'd want to modify calculate_payoff to return this data
    synthetic_payoffs = np.random.normal(
        loc=payoff_results['final_payoff'],
        scale=payoff_results['final_payoff'] * 0.1,  # 10% standard deviation
        size=num_simulations
    )

    # Plot simulation paths (only a few random ones)
    plot_simulation_paths(
        simulator,
        paths,
        simulator.product_parameter.underlying_indices,
        num_paths_to_plot=20
    )

    # Plot payoff distribution
    plot_payoff_distribution(synthetic_payoffs)

    print("\nSimulation test completed.")
    print("Results and plots have been saved.")


if __name__ == "__main__":
    main()