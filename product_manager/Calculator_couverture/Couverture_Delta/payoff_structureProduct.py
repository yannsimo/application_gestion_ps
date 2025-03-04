import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Product11Payoff:
    """
    Implements the payoff calculation for Product 11 (Performance Monde)
    
    This class focuses on the mathematical formulas for calculating the payoff,
    without the simulation aspects that are handled by other components.
    
    Key features:
    - 5 indices (ASX200, DAX, FTSE100, NASDAQ100, SMI)
    - 5-year maturity
    - Initial investment of 1000€
    - 40% participation rate
    - Final performance capped at +50% and floored at -15%
    - Guaranteed minimum 20% return if triggered
    - Annual dividends based on best-performing index (50x performance)
    - Best-performing index is excluded from future dividends
    """
    def __init__(self, initial_value: float = 1000.0):
        self.initial_value = initial_value
        self.participation_rate = 0.4  # 40% participation rate
        self.cap = 0.5  # +50% cap
        self.floor = -0.15  # -15% floor
        self.guarantee_threshold = 0.2  # 20% guarantee threshold
        self.dividend_multiplier = 50  # Multiplier for dividend calculation
    
    def calculate_basket_performance(self, 
                                     prices_initial: Dict[str, float],
                                     prices_final: Dict[str, float]) -> float:
        """
        Calculate basket performance from initial to final prices
        
        Args:
            prices_initial: Dictionary of initial prices for each index
            prices_final: Dictionary of final prices for each index
            
        Returns:
            Basket performance (average performance across indices)
        """
        performances = []
        
        for index, initial_price in prices_initial.items():
            if index in prices_final and initial_price > 0:
                final_price = prices_final[index]
                performance = (final_price / initial_price) - 1.0
                performances.append(performance)
        
        if not performances:
            return 0.0
        
        # Average performance across all indices
        basket_performance = sum(performances) / len(performances)
        
        return basket_performance
    
    def apply_caps_and_floors(self, performance: float, guarantee_active: bool) -> float:
        """
        Apply caps, floors, and guarantees to performance
        
        Args:
            performance: Raw performance
            guarantee_active: Whether 20% guarantee is active
            
        Returns:
            Adjusted performance
        """
        # Apply caps and floors
        capped_performance = min(performance, self.cap)
        floored_performance = max(capped_performance, self.floor)
        
        # Apply minimum guarantee if active
        if guarantee_active:
            floored_performance = max(floored_performance, self.guarantee_threshold)
        
        return floored_performance
    
    def calculate_final_performance(self, 
                                    prices_initial: Dict[str, float],
                                    prices_final: Dict[str, float],
                                    guarantee_active: bool) -> float:
        """
        Calculate final performance with all constraints applied
        
        Args:
            prices_initial: Dictionary of initial prices for each index
            prices_final: Dictionary of final prices for each index
            guarantee_active: Whether 20% guarantee is active
            
        Returns:
            Final performance (after participation rate)
        """
        # Calculate raw basket performance
        basket_performance = self.calculate_basket_performance(prices_initial, prices_final)
        
        # Apply caps, floors, and guarantees
        adjusted_performance = self.apply_caps_and_floors(basket_performance, guarantee_active)
        
        # Apply participation rate
        final_performance = self.participation_rate * adjusted_performance
        
        return final_performance
    
    def calculate_final_payoff(self, 
                               prices_initial: Dict[str, float],
                               prices_final: Dict[str, float],
                               guarantee_active: bool) -> float:
        """
        Calculate final payoff
        
        Args:
            prices_initial: Dictionary of initial prices for each index
            prices_final: Dictionary of final prices for each index
            guarantee_active: Whether 20% guarantee is active
            
        Returns:
            Final payoff amount
        """
        final_performance = self.calculate_final_performance(
            prices_initial, prices_final, guarantee_active
        )
        
        # Calculate payoff
        payoff = self.initial_value * (1 + final_performance)
        
        return payoff
    
    def calculate_dividend(self, 
                           prices_previous: Dict[str, float],
                           prices_current: Dict[str, float],
                           excluded_indices: Optional[List[str]] = None) -> Tuple[float, Optional[str]]:
        """
        Calculate dividend and determine best performing index
        
        Args:
            prices_previous: Dictionary of prices at previous observation date
            prices_current: Dictionary of prices at current observation date
            excluded_indices: List of indices already excluded from dividend calculation
            
        Returns:
            Tuple containing:
            - Dividend amount
            - Best performing index (to be excluded)
        """
        if excluded_indices is None:
            excluded_indices = []
        
        # Calculate annual returns for each active index
        annual_returns = {}
        
        for index, previous_price in prices_previous.items():
            if index in excluded_indices:
                continue
                
            if index in prices_current and previous_price > 0:
                current_price = prices_current[index]
                annual_return = (current_price / previous_price) - 1.0
                annual_returns[index] = annual_return
        
        if not annual_returns:
            return 0.0, None
        
        # Find best performing index
        best_index = max(annual_returns, key=annual_returns.get)
        best_return = annual_returns[best_index]
        
        # Calculate dividend (50 × best return)
        dividend = self.dividend_multiplier * max(0, best_return)
        
        return dividend, best_index
    
    def check_guarantee_trigger(self, 
                                prices_previous: Dict[str, float],
                                prices_current: Dict[str, float],
                                excluded_indices: Optional[List[str]] = None) -> bool:
        """
        Check if 20% guarantee is triggered
        
        Args:
            prices_previous: Dictionary of prices at previous observation date
            prices_current: Dictionary of prices at current observation date
            excluded_indices: List of indices already excluded from dividend calculation
            
        Returns:
            True if guarantee is triggered, False otherwise
        """
        if excluded_indices is None:
            excluded_indices = []
        
        # Calculate annual returns for each active index
        annual_returns = {}
        
        for index, previous_price in prices_previous.items():
            if index in excluded_indices:
                continue
                
            if index in prices_current and previous_price > 0:
                current_price = prices_current[index]
                annual_return = (current_price / previous_price) - 1.0
                annual_returns[index] = annual_return
        
        if not annual_returns:
            return False
        
        # Calculate average annual return
        avg_annual_return = sum(annual_returns.values()) / len(annual_returns)
        
        # Check if guarantee is triggered
        return avg_annual_return >= self.guarantee_threshold
    
    def calculate_path_dependent_payoff(self,
                                       prices_at_dates: Dict[datetime, Dict[str, float]],
                                       observation_dates: List[datetime],
                                       final_date: datetime) -> Tuple[float, Dict[datetime, float], List[str]]:
        """
        Calculate complete payoff including dividends and path-dependent features
        
        Args:
            prices_at_dates: Dictionary of prices at each date (indexed by date)
            observation_dates: List of observation dates
            final_date: Final date
            
        Returns:
            Tuple containing:
            - Final payoff
            - Dictionary of dividends by date
            - List of excluded indices
        """
        excluded_indices = []
        dividends = {}
        guarantee_active = False
        
        # Initial date (T0)
        initial_date = min(prices_at_dates.keys())
        prices_initial = prices_at_dates[initial_date]
        
        # Process each observation date
        for i, obs_date in enumerate(observation_dates):
            if obs_date not in prices_at_dates:
                continue
                
            # Get prices at this observation date
            prices_current = prices_at_dates[obs_date]
            
            # Get previous observation date
            prev_date = observation_dates[i-1] if i > 0 else initial_date
            prices_previous = prices_at_dates[prev_date]
            
            # Check for guarantee trigger
            if self.check_guarantee_trigger(prices_previous, prices_current, excluded_indices):
                guarantee_active = True
            
            # Calculate dividend
            dividend, best_index = self.calculate_dividend(prices_previous, prices_current, excluded_indices)
            dividends[obs_date] = dividend
            
            # Exclude best performing index
            if best_index is not None:
                excluded_indices.append(best_index)
        
        # Calculate final payoff
        prices_final = prices_at_dates[final_date]
        final_payoff = self.calculate_final_payoff(prices_initial, prices_final, guarantee_active)
        
        return final_payoff, dividends, excluded_indices