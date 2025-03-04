from datetime import datetime
from .Basket import Basket
from typing import List, Dict, Tuple, Optional

class StructuredProduct:
    """
    Represents Product 11 ("Performance Monde") structured product
    
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
    def __init__(self, 
                initial_date: datetime, 
                final_date: datetime,
                observation_dates: List[datetime], 
                initial_value: float = 1000.0,
                participation_rate: float = 0.4):
        self.initial_date = initial_date
        self.final_date = final_date
        self.observation_dates = observation_dates
        self.initial_value = initial_value
        self.participation_rate = participation_rate
        self.basket = Basket()
        self.min_guarantee_active = False
        self.paid_dividends = {}  # Track paid dividends by date
        self.excluded_indices = []  # Track excluded indices
    
    def calculate_final_performance(self, market_data) -> float:
        """
        Calculate the final product performance with all constraints
        
        Args:
            market_data: MarketData instance
            
        Returns:
            Final performance (after applying participation rate)
        """
        # Calculate raw basket performance
        basket_perf = self.basket.calculate_performance(
            market_data,
            self.initial_date,
            self.final_date
        ) / 100.0  # Convert from percentage to decimal
        
        # Apply constraints (cap, floor, guarantee)
        if basket_perf < 0:
            basket_perf = max(basket_perf, -0.15)  # Floor at -15%
        else:
            basket_perf = min(basket_perf, 0.5)  # Cap at +50%
        
        # Apply minimum guarantee if active
        if self.min_guarantee_active:
            basket_perf = max(basket_perf, 0.2)  # Minimum 20%
        
        # Apply participation rate (40%)
        final_perf = self.participation_rate * basket_perf
        
        return final_perf
    
    def calculate_final_payoff(self, market_data) -> float:
        """
        Calculate the final payoff amount
        
        Args:
            market_data: MarketData instance
            
        Returns:
            Final payoff amount in EUR
        """
        final_perf = self.calculate_final_performance(market_data)
        return self.initial_value * (1 + final_perf)
    
    def calculate_dividend(self, market_data, observation_date: datetime) -> Tuple[float, Optional[str]]:
        """
        Calculate dividend at an observation date
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            Tuple containing:
            - Dividend amount
            - Excluded index (or None)
        """
        # Skip if date is not an observation date
        if observation_date not in self.observation_dates:
            return 0.0, None
        
        # Skip if already calculated for this date
        if observation_date in self.paid_dividends:
            return self.paid_dividends[observation_date], None
        
        # Get active indices (not excluded)
        active_indices = [idx for idx in self.basket.indices if idx not in self.excluded_indices]
        
        if not active_indices:
            return 0.0, None
        
        # Calculate performance for each active index
        performances = {}
        for idx in active_indices:
            # Calculate performance from initial date to observation date
            start_price = market_data.get_price(idx, self.initial_date)
            current_price = market_data.get_price(idx, observation_date)
            
            if start_price and current_price and start_price > 0:
                perf = (current_price / start_price - 1)
                performances[idx] = perf
        
        if not performances:
            return 0.0, None
        
        # Find best performing index
        best_index = max(performances, key=performances.get)
        best_performance = performances[best_index]
        
        # Calculate dividend (50 times performance)
        dividend = 50 * max(0, best_performance)
        
        # Store dividend
        self.paid_dividends[observation_date] = dividend
        
        # Exclude this index from future dividend calculations
        self.excluded_indices.append(best_index)
        
        return dividend, best_index
    
    def check_guarantee_trigger(self, market_data, observation_date: datetime) -> bool:
        """
        Check if the 20% minimum guarantee is triggered
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            True if guarantee triggered, False otherwise
        """
        # Skip if not an observation date
        if observation_date not in self.observation_dates:
            return False
        
        # Calculate average basket performance
        active_indices = [idx for idx in self.basket.indices if idx not in self.excluded_indices]
        performances = []
        
        for idx in active_indices:
            annual_return = market_data.get_return(idx, observation_date)
            if annual_return is not None:
                performances.append(annual_return)
        
        # Calculate average performance
        if not performances:
            return False
        
        avg_performance = sum(performances) / len(performances)
        
        # Check if performance >= 20%
        if avg_performance >= 0.2:
            self.min_guarantee_active = True
            return True
        
        return False
    
    def process_observation_date(self, market_data, observation_date: datetime) -> Dict:
        """
        Process an observation date (checking guarantee, calculating dividend)
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            Dictionary with results
        """
        # Skip if not an observation date
        if observation_date not in self.observation_dates:
            return {
                'is_observation_date': False,
                'dividend': 0.0,
                'excluded_index': None,
                'guarantee_triggered': False
            }
        
        # Check guarantee trigger
        guarantee_triggered = self.check_guarantee_trigger(market_data, observation_date)
        
        # Calculate dividend
        dividend, excluded_index = self.calculate_dividend(market_data, observation_date)
        
        return {
            'is_observation_date': True,
            'dividend': dividend,
            'excluded_index': excluded_index,
            'guarantee_triggered': guarantee_triggered,
            'min_guarantee_active': self.min_guarantee_active
        }