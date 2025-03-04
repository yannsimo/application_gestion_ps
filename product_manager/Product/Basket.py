from .Index import Index
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
import numpy as np

class Basket:
    def __init__(self):
        # Initialisation with the 5 indices specific to Product 11
        self.indices = [idx.value for idx in Index]
        self.weights = {idx.value: 0.2 for idx in Index}  # Equal weights of 20%
        self.excluded_indices = []  # Indices excluded after dividend payments
        self.guarantee_threshold = 0.2  # 20% threshold for minimum guarantee
        self.min_guarantee_triggered = False  # Flag for minimum guarantee
        self.cap = 0.5  # +50% cap
        self.floor = -0.15  # -15% floor

    def calculate_performance(self, market_data, start_date: datetime, end_date: datetime) -> float:
        """Calculate the average performance of the 5 indices"""
        performances = []
        for idx in self.indices:
            start_price = market_data.get_price(idx, start_date)
            end_price = market_data.get_price(idx, end_date)
            if start_price and end_price and start_price > 0:
                perf = (end_price / start_price - 1) * 100
                performances.append(perf * self.weights[idx])

        return sum(performances) if performances else 0

    def calculate_annual_performance(self, market_data, date: datetime) -> float:
        """Calculate the annual performance of active indices"""
        performances = []
        active_indices = [idx for idx in self.indices if idx not in self.excluded_indices]
        for idx in active_indices:
            perf = market_data.get_return(idx, date)
            if perf is not None:
                performances.append(perf)
        return np.mean(performances) if performances else 0

    def get_active_indices(self) -> List[str]:
        """Return the list of indices not excluded from dividend calculation"""
        return [idx for idx in self.indices if idx not in self.excluded_indices]
    
    def exclude_best_performing_index(self, market_data, observation_date: datetime) -> Optional[str]:
        """
        Identify and exclude the best performing index at an observation date
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            The excluded index code or None
        """
        active_indices = self.get_active_indices()
        if not active_indices:
            return None
            
        # Calculate performance for each active index
        performances = {}
        for idx in active_indices:
            # Calculate performance from start date to observation date
            start_price = market_data.get_price(idx, self.get_initial_date(market_data))
            current_price = market_data.get_price(idx, observation_date)
            
            if start_price and current_price and start_price > 0:
                perf = (current_price / start_price - 1)
                performances[idx] = perf
        
        if not performances:
            return None
            
        # Find the best performing index
        best_index = max(performances, key=performances.get)
        
        # Add to excluded list
        self.excluded_indices.append(best_index)
        
        return best_index
    
    def calculate_dividend(self, market_data, observation_date: datetime) -> float:
        """
        Calculate dividend at an observation date based on best performing index
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            Dividend amount
        """
        active_indices = self.get_active_indices()
        if not active_indices:
            return 0.0
            
        # Calculate performance for each active index
        performances = {}
        for idx in active_indices:
            # Calculate performance from start date to observation date
            start_price = market_data.get_price(idx, self.get_initial_date(market_data))
            current_price = market_data.get_price(idx, observation_date)
            
            if start_price and current_price and start_price > 0:
                perf = (current_price / start_price - 1)
                performances[idx] = perf
        
        if not performances:
            return 0.0
            
        # Find the best performing index
        best_index = max(performances, key=performances.get)
        best_performance = performances[best_index]
        
        # Calculate dividend (50 times performance)
        dividend = 50 * max(0, best_performance)
        
        return dividend
    
    def check_guarantee_trigger(self, market_data, observation_date: datetime) -> bool:
        """
        Check if the 20% minimum guarantee is triggered
        
        Args:
            market_data: MarketData instance
            observation_date: Observation date
            
        Returns:
            True if guarantee triggered, False otherwise
        """
        # Calculate basket performance
        performance = self.calculate_annual_performance(market_data, observation_date)
        
        # Check if performance >= 20%
        if performance >= self.guarantee_threshold:
            self.min_guarantee_triggered = True
            return True
            
        return False
    
    def get_initial_date(self, market_data) -> datetime:
        """Get the initial date from market data"""
        return min(market_data.dates)
    
    def apply_caps_and_floors(self, performance: float) -> float:
        """Apply caps and floors to the performance"""
        capped_performance = min(performance, self.cap)
        floored_performance = max(capped_performance, self.floor)
        
        # Apply minimum guarantee if triggered
        if self.min_guarantee_triggered:
            floored_performance = max(floored_performance, self.guarantee_threshold)
            
        return floored_performance