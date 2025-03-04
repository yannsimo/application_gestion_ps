import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from numba import jit, prange

class VolatilityCalculator:
    def __init__(self, market_data):
        self.market_data = market_data

    @staticmethod
    @jit(nopython=True)
    def _calculate_volatility_numba(log_returns):
        return np.std(log_returns) * np.sqrt(252)  # Annualized volatility

    @staticmethod
    @jit(nopython=True)
    def _calculate_log_returns_numba(prices):
        return np.log(prices[1:] / prices[:-1])

    def calculate_volatility(self, index_code: str, current_date: datetime) -> float:
        """
        Calculate volatility using all available data up to current_date
        
        Args:
            index_code: Index code
            current_date: Current date
            
        Returns:
            Annualized volatility
        """
        # Get all available dates up to current_date
        all_dates = [d for d in self.market_data.dates if d <= current_date]
        
        if len(all_dates) < 2:
            return 0.2  # Default volatility if not enough data
            
        # Get prices for these dates
        prices = []
        valid_dates = []
        
        for date in all_dates:
            price = self.market_data.get_price(index_code, date)
            if price is not None and price > 0:
                prices.append(price)
                valid_dates.append(date)
        
        if len(prices) < 10:  # Need at least 10 data points for reliable volatility
            return 0.2  # Default volatility
            
        # Calculate log returns and volatility
        price_array = np.array(prices)
        log_returns = self._calculate_log_returns_numba(price_array)
        
        return self._calculate_volatility_numba(log_returns)

    def calculate_px_volatility(self, index_code: str, current_date: datetime) -> float:
        """
        Calculate volatility for PX (price in EUR) using all available data
        
        Args:
            index_code: Index code
            current_date: Current date
            
        Returns:
            Annualized volatility for PX
        """
        # Get all available dates up to current_date
        all_dates = [d for d in self.market_data.dates if d <= current_date]
        
        if len(all_dates) < 2:
            return 0.25  # Default volatility if not enough data
            
        # Get prices and exchange rates to calculate PX
        px_values = []
        valid_dates = []
        
        for date in all_dates:
            price = self.market_data.get_price(index_code, date)
            exchange_rate = self.market_data.get_index_exchange_rate(index_code, date)
            
            if price is not None and exchange_rate is not None and price > 0:
                # Calculate price in EUR (PX)
                px = price * exchange_rate
                px_values.append(px)
                valid_dates.append(date)
        
        if len(px_values) < 10:
            return 0.25  # Default volatility if not enough data
        
        # Calculate log returns and volatility
        px_array = np.array(px_values)
        log_returns = self._calculate_log_returns_numba(px_array)
        
        return self._calculate_volatility_numba(log_returns)

    def calculate_x_volatility(self, index_code: str, current_date: datetime) -> float:
        """
        Calculate volatility for X (exchange rate) using all available data
        
        Args:
            index_code: Index code
            current_date: Current date
            
        Returns:
            Annualized volatility for X
        """
        # Get all available dates up to current_date
        all_dates = [d for d in self.market_data.dates if d <= current_date]
        
        if len(all_dates) < 2:
            return 0.1  # Default volatility if not enough data
            
        # Get exchange rates
        x_values = []
        valid_dates = []
        
        for date in all_dates:
            exchange_rate = self.market_data.get_index_exchange_rate(index_code, date)
            
            if exchange_rate is not None and exchange_rate > 0:
                x_values.append(exchange_rate)
                valid_dates.append(date)
        
        if len(x_values) < 10:
            return 0.1  # Default volatility if not enough data
        
        # Calculate log returns and volatility
        x_array = np.array(x_values)
        log_returns = self._calculate_log_returns_numba(x_array)
        
        return self._calculate_volatility_numba(log_returns)

    def build_full_correlation_matrix(self, indices: List[str], current_date: datetime) -> Tuple[np.ndarray, List[str]]:
        """
        Build a correlation matrix for both PX and X processes
        
        Args:
            indices: List of index codes
            current_date: Current date
            
        Returns:
            Tuple containing:
            - np.ndarray: The correlation matrix
            - List[str]: Labels for the matrix rows/columns in format 'PX_INDEX' or 'X_INDEX'
        """
        # Get all available dates up to current_date
        all_dates = [d for d in self.market_data.dates if d <= current_date]
        
        if len(all_dates) < 10:
            # Not enough data, return identity matrix
            num_indices = len(indices)
            px_x_labels = []
            
            for index in indices:
                px_x_labels.append(f"PX_{index}")
                
                # Add X process if not EUR
                index_info = self.market_data.indices.get(index)
                if index_info and index_info.foreign_currency != 'EUR':
                    px_x_labels.append(f"X_{index}")
            
            n = len(px_x_labels)
            return np.eye(n), px_x_labels
            
        # Calculate returns for PX and X processes
        all_returns = {}
        px_x_labels = []
        
        for index in indices:
            # Calculate PX returns
            px_returns = self._get_px_returns(index, all_dates)
            if px_returns is not None and len(px_returns) > 1:
                all_returns[f"PX_{index}"] = px_returns
                px_x_labels.append(f"PX_{index}")
            
            # Calculate X returns (if not EUR)
            index_info = self.market_data.indices.get(index)
            if index_info and index_info.foreign_currency != 'EUR':
                x_returns = self._get_x_returns(index, all_dates)
                if x_returns is not None and len(x_returns) > 1:
                    all_returns[f"X_{index}"] = x_returns
                    px_x_labels.append(f"X_{index}")
        
        # Calculate correlation matrix
        n = len(px_x_labels)
        corr_matrix = np.eye(n)  # Start with identity matrix
        
        for i in range(n):
            for j in range(i+1, n):
                label_i = px_x_labels[i]
                label_j = px_x_labels[j]
                
                # Get returns
                returns_i = all_returns[label_i]
                returns_j = all_returns[label_j]
                
                # Find common dates
                common_dates = set(returns_i.index).intersection(set(returns_j.index))
                
                if len(common_dates) < 10:
                    # Not enough common data points, use default correlation
                    if 'PX_' in label_i and 'PX_' in label_j:
                        # Correlation between two PX processes
                        corr = 0.5
                    elif 'X_' in label_i and 'X_' in label_j:
                        # Correlation between two X processes
                        corr = 0.7
                    else:
                        # Correlation between PX and X
                        # Extract index codes
                        idx_i = label_i.split('_')[1]
                        idx_j = label_j.split('_')[1]
                        
                        if idx_i == idx_j:
                            # Same index, stronger correlation
                            corr = 0.8
                        else:
                            # Different indices
                            corr = 0.3
                else:
                    # Align series on common dates
                    aligned_i = returns_i.loc[list(common_dates)]
                    aligned_j = returns_j.loc[list(common_dates)]
                    
                    # Calculate correlation
                    corr = np.corrcoef(aligned_i, aligned_j)[0, 1]
                    
                    # Handle NaN values
                    if np.isnan(corr):
                        corr = 0.0
                
                # Set correlation in matrix (symmetric)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        
        return corr_matrix, px_x_labels

    def _get_px_returns(self, index_code: str, dates: List[datetime]) -> pd.Series:
        """
        Calculate log returns for PX (price in EUR)
        
        Args:
            index_code: Index code
            dates: List of dates
            
        Returns:
            Series of log returns indexed by date
        """
        # Get prices and exchange rates
        px_values = []
        valid_dates = []
        
        for date in dates:
            price = self.market_data.get_price(index_code, date)
            exchange_rate = self.market_data.get_index_exchange_rate(index_code, date)
            
            if price is not None and exchange_rate is not None and price > 0:
                # Calculate price in EUR (PX)
                px = price * exchange_rate
                px_values.append(px)
                valid_dates.append(date)
        
        if len(px_values) < 2:
            return None
        
        # Calculate log returns
        px_series = pd.Series(px_values, index=valid_dates)
        px_series = px_series.sort_index()  # Ensure dates are in order
        
        # Calculate log returns
        log_returns = np.log(px_series / px_series.shift(1)).dropna()
        
        return log_returns

    def _get_x_returns(self, index_code: str, dates: List[datetime]) -> pd.Series:
        """
        Calculate log returns for X (exchange rate)
        
        Args:
            index_code: Index code
            dates: List of dates
            
        Returns:
            Series of log returns indexed by date
        """
        # Get exchange rates
        x_values = []
        valid_dates = []
        
        for date in dates:
            exchange_rate = self.market_data.get_index_exchange_rate(index_code, date)
            
            if exchange_rate is not None and exchange_rate > 0:
                x_values.append(exchange_rate)
                valid_dates.append(date)
        
        if len(x_values) < 2:
            return None
        
        # Calculate log returns
        x_series = pd.Series(x_values, index=valid_dates)
        x_series = x_series.sort_index()  # Ensure dates are in order
        
        # Calculate log returns
        log_returns = np.log(x_series / x_series.shift(1)).dropna()
        
        return log_returns

    def calculate_cholesky_matrix(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Cholesky decomposition of correlation matrix
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Cholesky decomposition
        """
        # Ensure matrix is positive definite
        min_eig = np.min(np.linalg.eigvals(correlation_matrix))
        
        if min_eig < 0:
            # Add a small value to diagonal to make positive definite
            correlation_matrix = correlation_matrix - 10*min_eig * np.eye(correlation_matrix.shape[0])
        
        # Calculate Cholesky decomposition
        try:
            cholesky = np.linalg.cholesky(correlation_matrix)
            return cholesky
        except np.linalg.LinAlgError:
            # If still not positive definite, return identity matrix
            print("Warning: Could not compute Cholesky decomposition, using identity matrix")
            return np.eye(correlation_matrix.shape[0])

    def calculate_px_x_cholesky(self, indices: List[str], current_date: datetime) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate Cholesky decomposition for PX and X correlation matrix
        
        Args:
            indices: List of index codes
            current_date: Current date
            
        Returns:
            Tuple containing:
            - np.ndarray: Cholesky decomposition
            - List[str]: Labels for matrix rows/columns
        """
        # Build correlation matrix
        correlation_matrix, labels = self.build_full_correlation_matrix(indices, current_date)
        
        # Calculate Cholesky decomposition
        cholesky = self.calculate_cholesky_matrix(correlation_matrix)
        
        return cholesky, labels