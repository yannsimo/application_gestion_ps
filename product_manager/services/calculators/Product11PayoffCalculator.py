import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

class Product11PayoffCalculator:
    """
    Payoff calculator for Product 11 (Performance Monde)
    
    Key features:
    - 5 indices (ASX200, DAX, FTSE100, NASDAQ100, SMI)
    - Final performance capped at +50% and floored at -15%
    - Guaranteed minimum 20% return if triggered
    - Annual dividends based on best-performing index
    - Best-performing index is removed from pool after each payment date
    """
    def __init__(self, initial_value: float = 1000.0):
        self.initial_value = initial_value
        self.participation_rate = 0.4  # 40% participation rate
        self.cap = 0.5  # +50% cap
        self.floor = -0.15  # -15% floor
        self.guarantee_threshold = 0.2  # 20% guarantee threshold
        self.dividend_multiplier = 50  # Multiplier for dividend calculation
    
    def calculate_annual_returns(
        self,
        simulated_paths: Dict[str, np.ndarray],
        start_date: datetime,
        observation_dates: List[datetime],
        current_date: datetime
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate annual returns for each index at the observation dates
        
        Args:
            simulated_paths: Dictionary of simulated paths for each index
            start_date: Initial date (T0)
            observation_dates: List of observation dates (T1 to T4)
            current_date: Current date
            
        Returns:
            Dictionary of annual returns for each index at each observation date
        """
        # Find the nearest observation dates that are still in the future
        future_dates = [d for d in observation_dates if d > current_date]
        
        if not future_dates:
            return {}  # No future dates to consider
        
        # Calculate days between dates
        days_to_future_dates = [(d - current_date).days for d in future_dates]
        
        # Determine which indices to include at each observation date
        active_indices = list(simulated_paths.keys())
        
        # Get dimensions from the first path
        first_index = active_indices[0]
        num_paths = simulated_paths[first_index].shape[0]
        path_length = simulated_paths[first_index].shape[1]
        
        # Calculate index positions for the observation dates
        future_indices = [min(int(days / 365 * path_length), path_length - 1) for days in days_to_future_dates]
        
        annual_returns = {}
        
        for i, future_date in enumerate(future_dates):
            # Create a key for this observation date
            date_key = future_date.strftime('%Y-%m-%d')
            
            # Get the path index for this observation date
            path_idx = future_indices[i]
            
            # Calculate annual returns for each index
            annual_returns[date_key] = {}
            
            for index in active_indices:
                # Get the paths for this index
                paths = simulated_paths[index]
                
                # Calculate returns from start (paths[:, 0]) to this observation date (paths[:, path_idx])
                returns = paths[:, path_idx] / paths[:, 0] - 1.0
                
                annual_returns[date_key][index] = returns
        
        return annual_returns
    
    def check_guarantee_trigger(
        self,
        annual_returns: Dict[str, Dict[str, np.ndarray]],
        excluded_indices: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Check if the 20% guarantee is triggered based on annual returns
        
        Args:
            annual_returns: Dictionary of annual returns for each index at each observation date
            excluded_indices: List of indices already excluded from dividend calculation
            
        Returns:
            Tuple containing:
            - Boolean: True if guarantee is triggered
            - Dict: Mean annual returns for each observation date
        """
        if excluded_indices is None:
            excluded_indices = []
        
        mean_annual_returns = {}
        guarantee_triggered = False
        
        for date_key, returns_by_index in annual_returns.items():
            # Get returns for active indices only
            active_returns = {idx: ret for idx, ret in returns_by_index.items() if idx not in excluded_indices}
            
            if not active_returns:
                continue
            
            # Calculate mean return across all paths for active indices
            all_returns = np.concatenate([ret for ret in active_returns.values()])
            mean_return = np.mean(all_returns)
            
            mean_annual_returns[date_key] = mean_return
            
            # Check if guarantee is triggered (mean return >= 20%)
            if mean_return >= self.guarantee_threshold:
                guarantee_triggered = True
        
        return guarantee_triggered, mean_annual_returns
    
    def calculate_dividends(
        self,
        annual_returns: Dict[str, Dict[str, np.ndarray]],
        excluded_indices: Optional[List[str]] = None
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        Calculate dividends based on the best-performing index at each observation date
        
        Args:
            annual_returns: Dictionary of annual returns for each index at each observation date
            excluded_indices: List of indices already excluded from dividend calculation
            
        Returns:
            Tuple containing:
            - Dict: Dividend amount for each observation date
            - List: Additional indices to exclude after dividend calculation
        """
        if excluded_indices is None:
            excluded_indices = []
        
        dividends = {}
        new_excluded_indices = []
        
        for date_key, returns_by_index in annual_returns.items():
            # Filter out already excluded indices
            active_returns = {idx: ret for idx, ret in returns_by_index.items() if idx not in excluded_indices}
            
            if not active_returns:
                dividends[date_key] = 0.0
                continue
            
            # Calculate mean return for each index across all paths
            mean_returns = {idx: np.mean(ret) for idx, ret in active_returns.items()}
            
            # Find best performing index
            best_index = max(mean_returns, key=mean_returns.get)
            best_return = mean_returns[best_index]
            
            # Calculate dividend
            dividend = self.dividend_multiplier * max(0, best_return)
            dividends[date_key] = dividend
            
            # Add best index to excluded list for future calculations
            new_excluded_indices.append(best_index)
        
        return dividends, new_excluded_indices
    
    def calculate_final_performance(
        self,
        simulated_paths: Dict[str, np.ndarray],
        min_guarantee_active: bool = False
    ) -> np.ndarray:
        """
        Calculate the final performance of the product
        
        Args:
            simulated_paths: Dictionary of simulated paths for each index
            min_guarantee_active: Whether the 20% minimum guarantee is active
            
        Returns:
            Array of final performance values for each simulation path
        """
        # Get all indices
        indices = list(simulated_paths.keys())
        
        if not indices:
            return np.zeros(1)
        
        # Get the number of paths and the final time step
        num_paths = simulated_paths[indices[0]].shape[0]
        final_step = simulated_paths[indices[0]].shape[1] - 1
        
        # Calculate final performance for each index
        performances = np.zeros((len(indices), num_paths))
        
        for i, index in enumerate(indices):
            # Calculate performance from start to end for each path
            performances[i, :] = simulated_paths[index][:, final_step] / simulated_paths[index][:, 0] - 1.0
        
        # Calculate average performance across all indices for each path
        basket_performance = np.mean(performances, axis=0)
        
        # Apply floor (-15%) and cap (50%)
        capped_performance = np.clip(basket_performance, self.floor, self.cap)
        
        # Apply minimum guarantee if active
        if min_guarantee_active:
            capped_performance = np.maximum(capped_performance, self.guarantee_threshold)
        
        # Apply participation rate (40%)
        final_performance = self.participation_rate * capped_performance
        
        return final_performance
    
    def calculate_payoff(
        self,
        simulated_paths: Dict[str, np.ndarray],
        start_date: datetime,
        observation_dates: List[datetime],
        final_date: datetime,
        current_date: datetime,
        excluded_indices: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, float], List[str], bool]:
        """
        Calculate the complete payoff of the product
        
        Args:
            simulated_paths: Dictionary of simulated paths for each index
            start_date: Initial date (T0)
            observation_dates: List of observation dates (T1 to T4)
            final_date: Final date (Tc)
            current_date: Current date
            excluded_indices: List of indices already excluded from dividend calculation
            
        Returns:
            Tuple containing:
            - Array of final payoff values for each simulation path
            - Dict of dividend amounts for each observation date
            - List of indices to exclude for future calculations
            - Boolean indicating if minimum guarantee is active
        """
        if excluded_indices is None:
            excluded_indices = []
        
        # Calculate annual returns at observation dates
        annual_returns = self.calculate_annual_returns(
            simulated_paths, start_date, observation_dates, current_date
        )
        
        # Check if guarantee is triggered
        guarantee_triggered, _ = self.check_guarantee_trigger(annual_returns, excluded_indices)
        
        # Calculate dividends
        dividends, new_excluded_indices = self.calculate_dividends(annual_returns, excluded_indices)
        
        # Calculate final performance
        final_performance = self.calculate_final_performance(simulated_paths, guarantee_triggered)
        
        # Calculate final payoff
        final_payoff = self.initial_value * (1 + final_performance)
        
        return final_payoff, dividends, new_excluded_indices, guarantee_triggered