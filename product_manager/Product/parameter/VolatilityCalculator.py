import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
from numba import jit, prange

class VolatilityCalculator:
    def __init__(self, market_data):
        self.market_data = market_data

    @staticmethod
    @jit(nopython=True)
    def _calculate_volatility_numba(log_returns):
        return np.std(log_returns) * np.sqrt(252)

    @staticmethod
    @jit(nopython=True)
    def _calculate_log_returns_numba(prices):
        return np.log(prices[1:] / prices[:-1])

    def calculate_volatility(self, index_code: str, current_date: datetime) -> float:
        start_date = current_date - timedelta(days=6 * 365)
        prices = self._get_prices(index_code, start_date, current_date)

        if len(prices) < 2:
            return None

        log_returns = self._calculate_log_returns_numba(prices.values)
        return self._calculate_volatility_numba(log_returns)

    def _get_prices(self, index_code: str, start_date: datetime, end_date: datetime) -> pd.Series:
        prices = []
        dates = []
        current = start_date
        while current <= end_date:
            price = self.market_data.get_price(index_code, current)
            if price is not None:
                prices.append(price)
                dates.append(current)
            current += timedelta(days=1)

        return pd.Series(prices, index=dates)

    def calculate_all_volatilities(self, current_date: datetime) -> Dict[str, float]:
        volatilities = {}
        for index_code in self.market_data.get_available_indices():
            volatility = self.calculate_volatility(index_code, current_date)
            if volatility is not None:
                volatilities[index_code] = volatility

        return volatilities

    @staticmethod
    @jit(nopython=True)
    def _calculate_correlation_matrix_numba(returns):
        return np.corrcoef(returns)

    def calculate_correlation_matrix(self, indices: List[str], current_date: datetime) -> pd.DataFrame:
        start_date = current_date - timedelta(days=6 * 365)
        returns_list = []

        for index in indices:
            prices = self._get_prices(index, start_date, current_date)
            returns = self._calculate_log_returns_numba(prices.values)
            returns_list.append(returns)

        correlation_matrix = self._calculate_correlation_matrix_numba(np.array(returns_list))
        return pd.DataFrame(correlation_matrix, index=indices, columns=indices)

    @staticmethod
    @jit(nopython=True)
    def _calculate_cholesky_numba(volatilities, correlation_matrix):
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        return np.linalg.cholesky(cov_matrix)

    def calculate_vol_cholesky(self, indices: List[str], current_date: datetime) -> np.ndarray:
        volatilities = np.array([self.calculate_volatility(index, current_date) for index in indices])
        correlation_matrix = self.calculate_correlation_matrix(indices, current_date).values
        return self._calculate_cholesky_numba(volatilities, correlation_matrix)

# Exemple d'utilisation
if __name__ == "__main__":
    # Vous devrez implémenter une classe MockMarketData pour les tests
    class MockMarketData:
        def get_price(self, index_code, date):
            # Retourner un prix fictif pour les tests
            return 100 + np.random.randn()

        def get_available_indices(self):
            return ["ASX200", "DAX", "FTSE100", "NASDAQ100", "SMI"]

    market_data = MockMarketData()
    volatility_calculator = VolatilityCalculator(market_data)
    current_date = datetime.now()
    indices = ["ASX200", "DAX", "FTSE100", "NASDAQ100", "SMI"]

    # Calculer la matrice de Cholesky des volatilités
    vol_cholesky = volatility_calculator.calculate_vol_cholesky(indices, current_date)

    print("Matrice de Cholesky des volatilités :")
    print(vol_cholesky)