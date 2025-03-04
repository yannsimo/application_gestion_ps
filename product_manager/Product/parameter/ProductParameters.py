from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from ...Product.Index import Index
from ...Product.parameter.date.structured_product_dates import KEY_DATES_AUTO
from structured_product.product_manager.Product.parameter.Volatility.VolatilityCalculator import VolatilityCalculator


class ProductParameters:
    """
    Class to manage parameters for Product 11 (Performance Monde)
    
    This class centralizes all parameters needed for simulation and pricing,
    including market parameters (volatilities, correlations, interest rates)
    and product-specific parameters (dates, caps, floors, etc.)
    """
    def __init__(self, market_data, current_date: datetime):
        self.market_data = market_data
        self.current_date = current_date
        
        # Product 11 - indices
        self.underlying_indices = [index.value for index in Index]
        
        # Date parameters
        self.key_dates = KEY_DATES_AUTO
        self.initial_date = self.key_dates.T0
        self.final_date = self.key_dates.Tc
        self.observation_dates = [self.key_dates.get_Ti(i) for i in range(1, 5)]  # T1 to T4
        
        # Simulation parameters
        self.num_simulations = 10000
        self.time_steps_per_year = 252  # Daily steps
        
        # Product specific parameters
        self.initial_value = 1000.0
        self.participation_rate = 0.4  # 40% participation
        self.cap = 0.5  # +50% cap
        self.floor = -0.15  # -15% floor
        self.minimum_guarantee = 0.2  # 20% minimum guarantee if triggered
        self.dividend_multiplier = 50  # Multiplier for dividend calculation
        
        # Excluded indices for dividend calculation
        self.excluded_indices = set()
        
        # Volatility calculator
        self.volatility_calculator = VolatilityCalculator(self.market_data)
        
        # Update market parameters
        self.update_market_parameters()
    
    def update_market_parameters(self):
        """Update all market parameters based on current date"""
        # Volatilities
        self.volatilities_px = self._calculate_px_volatilities()
        self.volatilities_x = self._calculate_x_volatilities()
        
        # Interest rates
        self.domestic_rate = self._get_domestic_rate()
        self.foreign_rates = self._calculate_foreign_rates()
        
        # Correlation matrix
        self.correlation_matrix, self.correlation_labels = self._calculate_correlation_matrix()
        
        # Cholesky decomposition
        self.cholesky_matrix = self._calculate_cholesky_matrix()
    
    def _calculate_px_volatilities(self) -> Dict[str, float]:
        """Calculate PX volatilities for all indices"""
        return {index: self.volatility_calculator.calculate_px_volatility(index, self.current_date)
                for index in self.underlying_indices}
    
    def _calculate_x_volatilities(self) -> Dict[str, float]:
        """Calculate X volatilities for all indices"""
        return {index: self.volatility_calculator.calculate_x_volatility(index, self.current_date)
                for index in self.underlying_indices}
    
    def _get_domestic_rate(self) -> float:
        """Get domestic (EUR) interest rate"""
        rate = self.market_data.get_interest_rate("EUR", self.current_date)
        return rate if rate is not None else 0.01  # Default to 1% if not available
    
    def _calculate_foreign_rates(self) -> Dict[str, float]:
        """Calculate foreign interest rates for all indices"""
        return {index: self.market_data.get_index_interest_rate(index, self.current_date)
                for index in self.underlying_indices}
    
    def _calculate_correlation_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Calculate correlation matrix for PX and X processes"""
        return self.volatility_calculator.build_full_correlation_matrix(
            self.underlying_indices, self.current_date
        )
    
    def _calculate_cholesky_matrix(self) -> np.ndarray:
        """Calculate Cholesky decomposition of correlation matrix"""
        # Get correlation matrix
        correlation_matrix, labels = self.correlation_matrix, self.correlation_labels
        
        # Ensure matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(correlation_matrix))
        if min_eig < 0:
            # Add a small value to diagonal to make positive definite
            correlation_matrix = correlation_matrix - 10*min_eig * np.eye(correlation_matrix.shape[0])
        
        # Calculate Cholesky decomposition
        return np.linalg.cholesky(correlation_matrix)
    
    def get_time_to_maturity(self) -> float:
        """Calculate time to maturity in years"""
        return (self.final_date - self.current_date).days / 365.0
    
    def get_time_step(self) -> float:
        """Calculate time step for simulation"""
        return 1.0 / self.time_steps_per_year
    
    def get_num_time_steps(self) -> int:
        """Calculate number of time steps for simulation"""
        time_to_maturity = self.get_time_to_maturity()
        dt = self.get_time_step()
        return max(1, int(time_to_maturity / dt) + 1)
    
    def is_observation_date(self, date: datetime) -> bool:
        """Check if a date is an observation date"""
        return date in self.observation_dates
    
    def get_next_observation_date(self) -> Optional[datetime]:
        """Get next observation date after current date"""
        future_dates = [d for d in self.observation_dates if d > self.current_date]
        return min(future_dates) if future_dates else None
    
    def get_remaining_observation_dates(self) -> List[datetime]:
        """Get all remaining observation dates"""
        return [d for d in self.observation_dates if d > self.current_date]
    
    def update_date(self, new_date: datetime):
        """Update current date and recalculate market parameters"""
        self.current_date = new_date
        self.update_market_parameters()
    
    def exclude_index(self, index_code: str):
        """Exclude an index from future dividend calculations"""
        if index_code in self.underlying_indices:
            self.excluded_indices.add(index_code)
    
    def get_active_indices(self) -> List[str]:
        """Get list of indices not excluded from dividend calculation"""
        return [idx for idx in self.underlying_indices if idx not in self.excluded_indices]