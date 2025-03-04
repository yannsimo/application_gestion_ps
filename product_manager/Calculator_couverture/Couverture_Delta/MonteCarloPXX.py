import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import pandas as pd

from .model import BlackScholesSimulation

class MonteCarloPXX:
    """
    Monte Carlo simulation for PX and X processes
    
    This class handles:
    - Simulation of PX and X processes with proper correlation
    - Calculation of P paths
    - Shifting prices for delta calculation
    """
    def __init__(self, market_data, volatility_calculator):
        self.market_data = market_data
        self.volatility_calculator = volatility_calculator
    
    def generate_correlated_random_variables(
        self, 
        num_variables: int, 
        num_paths: int, 
        num_steps: int, 
        cholesky_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Generate correlated random variables using Cholesky decomposition
        
        Args:
            num_variables: Number of random variables
            num_paths: Number of simulation paths
            num_steps: Number of time steps
            cholesky_matrix: Cholesky decomposition of correlation matrix
            
        Returns:
            Array of correlated random variables
        """
        # Generate independent standard normal random variables
        Z = np.random.standard_normal((num_variables, num_paths, num_steps))
        
        # Apply Cholesky decomposition to get correlated random variables
        correlated_Z = np.zeros_like(Z)
        
        for t in range(num_steps):
            # For each time step, apply the correlation structure
            for p in range(num_paths):
                # Reshape Z to a column vector
                z_vec = Z[:, p, t].reshape(-1, 1)
                
                # Apply Cholesky decomposition
                correlated_vec = np.dot(cholesky_matrix, z_vec).flatten()
                
                # Store correlated random variables
                correlated_Z[:, p, t] = correlated_vec
        
        return correlated_Z
    
    def simulate(
        self, 
        indices: List[str],
        current_date: datetime,
        time_horizon: float,
        num_paths: int = 1000,
        dt: float = 1/252  # Daily steps
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run Monte Carlo simulation for PX and X processes
        
        Args:
            indices: List of index codes
            current_date: Current date
            time_horizon: Time horizon in years
            num_paths: Number of simulation paths
            dt: Time step in years
            
        Returns:
            Tuple containing dictionaries of simulated PX, X, and P paths
        """
        # Calculate number of time steps
        num_steps = int(time_horizon / dt) + 1
        
        # Get Cholesky decomposition for PX and X processes
        cholesky_matrix, labels = self.volatility_calculator.calculate_px_x_cholesky(indices, current_date)
        
        # Prepare parameters for simulation
        px0_array = []
        x0_array = []
        r_d_array = []
        r_f_array = []
        sigma_px_array = []
        sigma_x_array = []
        
        px_indices = []
        x_indices = []
        
        # For each label, extract index code and determine if it's PX or X
        for label in labels:
            parts = label.split('_')
            process_type = parts[0]  # PX or X
            index_code = '_'.join(parts[1:])  # Rest is the index code
            
            if process_type == 'PX':
                px_indices.append(index_code)
                
                # Get parameters for PX process
                price = self.market_data.get_price(index_code, current_date)
                exchange_rate = self.market_data.get_index_exchange_rate(index_code, current_date)
                
                # Default values if data not available
                if price is None or exchange_rate is None:
                    price = 100.0
                    exchange_rate = 1.0
                    
                px = price * exchange_rate
                px0_array.append(px)
                
                # Domestic interest rate
                domestic_rate = self.market_data.get_interest_rate("EUR", current_date)
                domestic_rate = domestic_rate if domestic_rate is not None else 0.01  # Default to 1%
                r_d_array.append(domestic_rate)
                
                # Volatility for PX
                vol_px = self.volatility_calculator.calculate_px_volatility(index_code, current_date)
                sigma_px_array.append(vol_px)
            
            elif process_type == 'X':
                x_indices.append(index_code)
                
                # Get parameters for X process
                exchange_rate = self.market_data.get_index_exchange_rate(index_code, current_date)
                exchange_rate = exchange_rate if exchange_rate is not None else 1.0  # Default to 1.0
                x0_array.append(exchange_rate)
                
                # Domestic interest rate
                domestic_rate = self.market_data.get_interest_rate("EUR", current_date)
                domestic_rate = domestic_rate if domestic_rate is not None else 0.01  # Default to 1%
                r_d_array.append(domestic_rate)
                
                # Foreign interest rate
                foreign_rate = self.market_data.get_index_interest_rate(index_code, current_date)
                foreign_rate = foreign_rate if foreign_rate is not None else 0.01  # Default to 1%
                r_f_array.append(foreign_rate)
                
                # Volatility for X
                vol_x = self.volatility_calculator.calculate_x_volatility(index_code, current_date)
                sigma_x_array.append(vol_x)
        
        # Convert to numpy arrays
        px0_array = np.array(px0_array)
        x0_array = np.array(x0_array)
        r_d_array = np.array(r_d_array)
        r_f_array = np.array(r_f_array)
        sigma_px_array = np.array(sigma_px_array)
        sigma_x_array = np.array(sigma_x_array)
        
        # Run simulation
        px_paths, x_paths, p_paths = BlackScholesSimulation.simulate_px_x(
            px0_array, x0_array, r_d_array, r_f_array,
            sigma_px_array, sigma_x_array, time_horizon, dt,
            num_paths, cholesky_matrix
        )
        
        # Convert to dictionaries
        px_paths_dict = {px_indices[i]: px_paths[i] for i in range(len(px_indices))}
        x_paths_dict = {x_indices[i]: x_paths[i] for i in range(len(x_indices))}
        p_paths_dict = {px_indices[i]: p_paths[i] for i in range(len(px_indices))}
        
        return px_paths_dict, x_paths_dict, p_paths_dict
    
    def simulate_with_shifts(
        self, 
        indices: List[str],
        current_date: datetime,
        time_horizon: float,
        shift_size: float = 0.01,  # 1% shift
        num_paths: int = 1000,
        dt: float = 1/252
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
        """
        Run Monte Carlo simulation with shifts for calculating deltas
        
        Args:
            indices: List of index codes
            current_date: Current date
            time_horizon: Time horizon in years
            shift_size: Size of the shift for delta calculation (as a proportion)
            num_paths: Number of simulation paths
            dt: Time step in years
            
        Returns:
            Tuple containing base simulation results and shifted simulation results
        """
        # Run base simulation
        base_px_paths, base_x_paths, base_p_paths = self.simulate(
            indices, current_date, time_horizon, num_paths, dt
        )
        
        # Dictionary to store shifted simulations
        shifted_simulations = {}
        
        # Run simulations with shifts for each index
        for index in indices:
            # Get original price
            original_price = self.market_data.get_price(index, current_date)
            
            # Skip if price not available
            if original_price is None:
                continue
            
            # Save original get_price method
            original_get_price = self.market_data.get_price
            
            try:
                # Create a price override function
                def price_override(idx, date=None):
                    if date is None:
                        date = current_date
                    if idx == index and date == current_date:
                        return self._price_override
                    return original_get_price(idx, date)
                
                # Upward shift
                self._price_override = original_price * (1 + shift_size)
                self.market_data.get_price = price_override.__get__(self.market_data)
                
                # Run simulation with upward shift
                up_px_paths, up_x_paths, up_p_paths = self.simulate(
                    indices, current_date, time_horizon, num_paths, dt
                )
                
                # Downward shift
                self._price_override = original_price * (1 - shift_size)
                
                # Run simulation with downward shift
                down_px_paths, down_x_paths, down_p_paths = self.simulate(
                    indices, current_date, time_horizon, num_paths, dt
                )
                
                # Store results
                shifted_simulations[index] = {
                    'up': {
                        'PX': up_px_paths,
                        'X': up_x_paths,
                        'P': up_p_paths
                    },
                    'down': {
                        'PX': down_px_paths,
                        'X': down_x_paths,
                        'P': down_p_paths
                    }
                }
            finally:
                # Restore original get_price method
                self.market_data.get_price = original_get_price
        
        # Return base and shifted simulations
        base_simulation = {
            'PX': base_px_paths,
            'X': base_x_paths,
            'P': base_p_paths
        }
        
        return base_simulation, shifted_simulations
    
    def calculate_deltas(
        self,
        indices: List[str],
        current_date: datetime,
        observation_dates: List[datetime],
        final_date: datetime,
        payoff_calculator,
        excluded_indices: List[str] = None,
        shift_size: float = 0.01,
        num_paths: int = 1000,
        dt: float = 1/252
    ) -> Dict[str, float]:
        """
        Calculate delta hedging ratios for each index
        
        Args:
            indices: List of index codes
            current_date: Current date
            observation_dates: List of observation dates (T1 to T4)
            final_date: Final date (Tc)
            payoff_calculator: PayoffCalculator instance
            excluded_indices: List of indices already excluded from dividend calculation
            shift_size: Size of the shift for delta calculation
            num_paths: Number of simulation paths
            dt: Time step in years
            
        Returns:
            Dictionary of delta values for each index
        """
        if excluded_indices is None:
            excluded_indices = []
            
        # Calculate time horizon in years
        time_horizon = (final_date - current_date).days / 365.0
        
        if time_horizon <= 0:
            return {index: 0.0 for index in indices}
        
        # Run simulations with shifts
        base_simulation, shifted_simulations = self.simulate_with_shifts(
            indices, current_date, time_horizon, shift_size, num_paths, dt
        )
        
        # Extract P paths from base simulation
        base_p_paths = base_simulation['P']
        
        # Calculate base payoff
        base_payoff, _, _, _ = payoff_calculator.calculate_payoff(
            base_p_paths, 
            current_date, 
            observation_dates, 
            final_date,
            current_date,
            excluded_indices
        )
        
        base_expected_payoff = np.mean(base_payoff)
        
        # Calculate deltas
        deltas = {}
        
        for index in indices:
            # Skip if index not in shifted simulations
            if index not in shifted_simulations:
                deltas[index] = 0.0
                continue
                
            # Get shifted simulations
            up_p_paths = shifted_simulations[index]['up']['P']
            down_p_paths = shifted_simulations[index]['down']['P']
            
            # Calculate payoffs for shifted simulations
            up_payoff, _, _, _ = payoff_calculator.calculate_payoff(
                up_p_paths,
                current_date,
                observation_dates,
                final_date,
                current_date,
                excluded_indices
            )
            
            down_payoff, _, _, _ = payoff_calculator.calculate_payoff(
                down_p_paths,
                current_date,
                observation_dates,
                final_date,
                current_date,
                excluded_indices
            )
            
            # Calculate expected payoffs
            up_expected_payoff = np.mean(up_payoff)
            down_expected_payoff = np.mean(down_payoff)
            
            # Calculate delta using central difference approximation
            price = self.market_data.get_price(index, current_date)
            if price is None or price == 0:
                deltas[index] = 0.0
                continue
                
            delta = (up_expected_payoff - down_expected_payoff) / (2 * shift_size * price)
            deltas[index] = delta
        
        return deltas