import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from .model import BlackScholesSimulation
from ...Product.Index import Index
from ...Product.parameter import ProductParameters
from ...Product.parameter.Volatility.VolatilityCalculator import VolatilityCalculator


def get_singleton_market_data():
    from ...Data.SingletonMarketData import SingletonMarketData
    return SingletonMarketData


class StructuredProductPricer:
    """
    Pricer for structured products
    
    This class is responsible for:
    - Running Monte Carlo simulations
    - Calculating expected payoffs
    - Calculating delta hedging ratios
    """
    def __init__(self):
        self.market_data = get_singleton_market_data().get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        
        # Export attributes from ProductParameters
        self.underlying_indices = self.product_parameter.underlying_indices
        self.key_dates = self.product_parameter.key_dates
        self.initial_date = self.product_parameter.initial_date
        self.final_date = self.product_parameter.final_date
        self.observation_dates = self.product_parameter.observation_dates
        self.num_simulations = self.product_parameter.num_simulations
        self.initial_value = self.product_parameter.initial_value
        self.participation_rate = self.product_parameter.participation_rate
        self.cap = self.product_parameter.cap
        self.floor = self.product_parameter.floor
        self.minimum_guarantee = self.product_parameter.minimum_guarantee
        self.dividend_multiplier = self.product_parameter.dividend_multiplier
        
        # Market parameters
        self.volatilities_px = self.product_parameter.volatilities_px
        self.volatilities_x = self.product_parameter.volatilities_x
        self.domestic_rate = self.product_parameter.domestic_rate
        self.foreign_rates = self.product_parameter.foreign_rates
        self.correlation_matrix = self.product_parameter.correlation_matrix
        self.correlation_labels = self.product_parameter.correlation_labels
        self.cholesky_matrix = self.product_parameter.cholesky_matrix
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all indices"""
        return {index: self.market_data.get_price(index, self.market_data.current_date)
                for index in self.underlying_indices}
    
    def get_current_exchange_rates(self) -> Dict[str, float]:
        """Get current exchange rates for all indices"""
        return {index: self.market_data.get_index_exchange_rate(index, self.market_data.current_date)
                for index in self.underlying_indices}
    
    def run_monte_carlo(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run Monte Carlo simulation
        
        Returns:
            Dictionary containing simulated paths
        """
        # Get current date and time to maturity
        current_date = self.market_data.current_date
        maturity_date = self.final_date
        time_to_maturity = (maturity_date - current_date).days / 365.0
        
        if time_to_maturity <= 0:
            return {
                'PX': {},
                'X': {},
                'P': {}
            }
        
        # Get simulation parameters
        dt = 1 / 252  # Daily steps
        
        # Prepare arrays for simulation
        px0_array = []
        x0_array = []
        r_d_array = []
        r_f_array = []
        sigma_px_array = []
        sigma_x_array = []
        indices_px = []
        indices_x = []
        
        # Parse correlation labels to determine which are PX and which are X
        for label in self.correlation_labels:
            parts = label.split('_')
            process_type = parts[0]  # PX or X
            index_code = '_'.join(parts[1:])  # Rest is the index code
            
            if process_type == 'PX':
                indices_px.append(index_code)
                price = self.market_data.get_price(index_code, current_date)
                exchange_rate = self.market_data.get_index_exchange_rate(index_code, current_date)
                
                # PX is price in EUR
                px0 = price * exchange_rate if price is not None and exchange_rate is not None else 100.0
                px0_array.append(px0)
                
                # Domestic interest rate (EUR)
                r_d = self.market_data.get_interest_rate("EUR", current_date)
                r_d = r_d if r_d is not None else 0.01  # Default to 1%
                r_d_array.append(r_d)
                
                # Volatility
                vol_px = self.volatilities_px.get(index_code, 0.2)  # Default to 20%
                sigma_px_array.append(vol_px)
                
            elif process_type == 'X':
                indices_x.append(index_code)
                
                # Exchange rate
                x0 = self.market_data.get_index_exchange_rate(index_code, current_date)
                x0 = x0 if x0 is not None else 1.0  # Default to 1.0
                x0_array.append(x0)
                
                # Domestic and foreign interest rates
                r_d = self.market_data.get_interest_rate("EUR", current_date)
                r_d = r_d if r_d is not None else 0.01  # Default to 1%
                r_d_array.append(r_d)
                
                r_f = self.market_data.get_index_interest_rate(index_code, current_date)
                r_f = r_f if r_f is not None else 0.01  # Default to 1%
                r_f_array.append(r_f)
                
                # Volatility
                vol_x = self.volatilities_x.get(index_code, 0.1)  # Default to 10%
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
            sigma_px_array, sigma_x_array, time_to_maturity, dt,
            self.num_simulations, self.cholesky_matrix
        )
        
        # Convert to dictionaries
        px_paths_dict = {indices_px[i]: px_paths[i] for i in range(len(indices_px))}
        x_paths_dict = {indices_x[i]: x_paths[i] for i in range(len(indices_x))}
        p_paths_dict = {indices_px[i]: p_paths[i] for i in range(len(indices_px))}
        
        return {
            'PX': px_paths_dict,
            'X': x_paths_dict,
            'P': p_paths_dict
        }
    
    def run_monte_carlo_with_shifts(self, shift_size: float = 0.01) -> Tuple[Dict, Dict]:
        """
        Run Monte Carlo simulation with price shifts for delta calculation
        
        Args:
            shift_size: Size of price shift (as a proportion)
            
        Returns:
            Tuple containing base and shifted simulation results
        """
        # Run base simulation
        base_results = self.run_monte_carlo()
        
        # Dictionary to store shifted results
        shifted_results = {}
        
        # Run simulations with shifts for each index
        for index in self.underlying_indices:
            # Get original price
            original_price = self.market_data.get_price(index, self.market_data.current_date)
            if original_price is None:
                continue
                
            # Save get_price method for restoration later
            original_get_price = self.market_data.get_price
            
            try:
                # Define a wrapper method for price overriding
                def override_get_price(idx, date=None):
                    if date is None:
                        date = self.market_data.current_date
                    if idx == index and date == self.market_data.current_date:
                        return self._override_price.get(idx, original_get_price(idx, date))
                    return original_get_price(idx, date)
                
                # Store override price attribute if it doesn't exist
                if not hasattr(self.market_data, '_override_price'):
                    self.market_data._override_price = {}
                
                # Apply upward shift
                shifted_price_up = original_price * (1 + shift_size)
                self.market_data._override_price = {index: shifted_price_up}
                self.market_data.get_price = override_get_price.__get__(self.market_data)
                
                # Run simulation with upward shift
                up_results = self.run_monte_carlo()
                
                # Apply downward shift
                shifted_price_down = original_price * (1 - shift_size)
                self.market_data._override_price = {index: shifted_price_down}
                
                # Run simulation with downward shift
                down_results = self.run_monte_carlo()
                
                # Store results
                shifted_results[index] = {
                    'up': up_results,
                    'down': down_results
                }
            finally:
                # Restore original method
                self.market_data.get_price = original_get_price
                if hasattr(self.market_data, '_override_price'):
                    self.market_data._override_price = {}
        
        return base_results, shifted_results
    
    def calculate_deltas(self, payoff_calculator, excluded_indices=None) -> Dict[str, float]:
        """
        Calculate delta hedging ratios
        
        Args:
            payoff_calculator: PayoffCalculator instance
            excluded_indices: List of indices excluded from dividend calculation
            
        Returns:
            Dictionary of delta values for each index
        """
        if excluded_indices is None:
            excluded_indices = []
        
        # Run simulations with shifts
        base_results, shifted_results = self.run_monte_carlo_with_shifts()
        
        # Calculate base payoff
        base_p_paths = base_results.get('P', {})
        
        if not base_p_paths:
            return {index: 0.0 for index in self.underlying_indices}
        
        # Calculate base payoff
        base_payoff, _, _, _ = payoff_calculator.calculate_payoff(
            base_p_paths,
            self.initial_date,
            self.observation_dates,
            self.final_date,
            self.market_data.current_date,
            excluded_indices
        )
        
        base_expected_payoff = np.mean(base_payoff)
        
        # Calculate deltas
        deltas = {}
        
        for index in self.underlying_indices:
            if index not in shifted_results:
                deltas[index] = 0.0
                continue
            
            # Get shifted results
            shifted_up = shifted_results[index]['up']
            shifted_down = shifted_results[index]['down']
            
            # Get P paths
            up_p_paths = shifted_up.get('P', {})
            down_p_paths = shifted_down.get('P', {})
            
            if not up_p_paths or not down_p_paths:
                deltas[index] = 0.0
                continue
            
            # Calculate payoffs for shifted simulations
            up_payoff, _, _, _ = payoff_calculator.calculate_payoff(
                up_p_paths,
                self.initial_date,
                self.observation_dates,
                self.final_date,
                self.market_data.current_date,
                excluded_indices
            )
            
            down_payoff, _, _, _ = payoff_calculator.calculate_payoff(
                down_p_paths,
                self.initial_date,
                self.observation_dates,
                self.final_date,
                self.market_data.current_date,
                excluded_indices
            )
            
            # Calculate expected payoffs
            up_expected_payoff = np.mean(up_payoff)
            down_expected_payoff = np.mean(down_payoff)
            
            # Get current price
            price = self.market_data.get_price(index, self.market_data.current_date)
            
            if price is None or price == 0:
                deltas[index] = 0.0
                continue
            
            # Calculate delta using central difference
            delta = (up_expected_payoff - down_expected_payoff) / (2 * 0.01 * price)
            deltas[index] = delta
        
        return deltas
    
    def update_parameters(self, current_date: datetime):
        """
        Update parameters based on new date
        
        Args:
            current_date: New date
        """
        self.product_parameter.update_date(current_date)
        
        # Update exported parameters
        self.volatilities_px = self.product_parameter.volatilities_px
        self.volatilities_x = self.product_parameter.volatilities_x
        self.domestic_rate = self.product_parameter.domestic_rate
        self.foreign_rates = self.product_parameter.foreign_rates
        self.correlation_matrix = self.product_parameter.correlation_matrix
        self.correlation_labels = self.product_parameter.correlation_labels
        self.cholesky_matrix = self.product_parameter.cholesky_matrix