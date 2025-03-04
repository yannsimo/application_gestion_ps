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
    def calculate_volatility_echange(self, index_code: str, current_date: datetime) -> float:
        start_date = current_date - timedelta(days=6 * 365)
        prices_echange = self._get_prices_echange(index_code, start_date, current_date)

        if len(prices_echange) < 2:
            return None

        log_returns = self._calculate_log_returns_numba(prices_echange.values)
        return self._calculate_volatility_numba(log_returns)

    def _get_prices_echange(self, index_code: str, start_date: datetime, end_date: datetime) -> pd.Series:
        prices = []
        dates = []
        current = start_date
        while current <= end_date:
            echange_price = self.market_data.get_exchange_rate(index_code, current)
            if echange_price is not None:
                prices.append(echange_price)
                dates.append(current)
            current += timedelta(days=1)
        return pd.Series(prices, index=dates)

    def calculate_volatility_rate(self, index_code: str, current_date: datetime) -> float:
        start_date = current_date - timedelta(days=6 * 365)
        prices_rate = self._get_prices_rate(index_code, start_date, current_date)

        if len(prices_rate) < 2:
            return None

        log_returns = self._calculate_log_returns_numba(prices_rate.values)
        return self._calculate_volatility_numba(log_returns)

    def _get_prices_rate(self, index_code: str, start_date: datetime, end_date: datetime) -> pd.Series:
        prices = []
        dates = []
        current = start_date
        while current <= end_date:
            rate_price = self.market_data.get_interest_rate(index_code, current)
            if rate_price is not None:
                prices.append(rate_price)
                dates.append(current)
            current += timedelta(days=1)
        return pd.Series(prices, index=dates)

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
        #corrige ma matrice de correlation parce qu'il doit avoir le prix des actifs avec les taux de change
        correlation_matrix = self._calculate_correlation_matrix_numba(np.array(returns_list))
        return pd.DataFrame(correlation_matrix, index=indices, columns=indices)

    @staticmethod
    @jit(nopython=True)
    def _calculate_cholesky_numba(volatilities, correlation_matrix):
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        return np.linalg.cholesky(cov_matrix)

    def calculate_correlation_matrix_combined(self, indices: List[str], current_date: datetime) -> pd.DataFrame:
        start_date = current_date - timedelta(days=6 * 365)
        all_returns_data = {}
        combined_factors = []

        # Collecter les rendements des prix des indices
        for index in indices:
            factor_name = f"{index}_price"
            combined_factors.append(factor_name)

            prices = self._get_prices(index, start_date, current_date)
            if len(prices) >= 2:
                log_returns = self._calculate_log_returns_numba(prices.values)
                all_returns_data[factor_name] = log_returns
            else:
                # Gérer le cas des données insuffisantes
                all_returns_data[factor_name] = np.array([])

        # Collecter les rendements des taux de change
        for index in indices:
            factor_name = f"{index}_exchange"
            combined_factors.append(factor_name)

            exchange_rates = self._get_prices_echange(index, start_date, current_date)
            if len(exchange_rates) >= 2:
                log_returns = self._calculate_log_returns_numba(exchange_rates.values)
                all_returns_data[factor_name] = log_returns
            else:
                all_returns_data[factor_name] = np.array([])

        # S'assurer que toutes les séries ont la même longueur en prenant le minimum
        non_empty_returns = [returns for returns in all_returns_data.values() if len(returns) > 0]
        if not non_empty_returns:
            # Retourner une matrice d'identité si pas de données
            n = len(combined_factors)
            return pd.DataFrame(np.eye(n), index=combined_factors, columns=combined_factors)

        min_length = min(len(returns) for returns in non_empty_returns)

        # Extraire et aligner les données
        aligned_returns = []
        aligned_factors = []

        for factor in combined_factors:
            returns = all_returns_data[factor]
            if len(returns) >= min_length:
                aligned_returns.append(returns[-min_length:])
                aligned_factors.append(factor)

        # Calculer la matrice de corrélation
        if len(aligned_returns) >= 2:
            correlation_matrix = self._calculate_correlation_matrix_numba(np.array(aligned_returns))
            return pd.DataFrame(correlation_matrix, index=aligned_factors, columns=aligned_factors)
        else:
            # Retourner une matrice d'identité si pas assez de séries
            n = len(aligned_factors)
            return pd.DataFrame(np.eye(n), index=aligned_factors, columns=aligned_factors)

    def calculate_vol_cholesky(self, indices: List[str], current_date: datetime) -> np.ndarray:
        # Créer une liste combinée d'indices et leurs taux de change
        combined_factors = []
        combined_volatilities = []

        # Calculer les volatilités des prix des indices
        for index in indices:
            vol_price = self.calculate_volatility(index, current_date)
            if vol_price is not None:
                combined_factors.append(f"{index}_price")
                combined_volatilities.append(vol_price)

        # Calculer les volatilités des taux de change
        for index in indices:
            vol_exchange = self.calculate_volatility_echange(index, current_date)
            if vol_exchange is not None:
                combined_factors.append(f"{index}_exchange")
                combined_volatilities.append(vol_exchange)

        # Si nous n'avons pas assez de facteurs, retourner une matrice vide
        if len(combined_factors) < 2:
            return np.array([])

        # Calculer la matrice de corrélation combinée
        correlation_matrix = self.calculate_correlation_matrix_combined(indices, current_date)

        # Filtrer la matrice de corrélation pour inclure uniquement les facteurs pour lesquels nous avons des volatilités
        filtered_corr = correlation_matrix.loc[combined_factors, combined_factors].values

        # Calculer la matrice de Cholesky
        combined_volatilities = np.array(combined_volatilities)

        try:
            return self._calculate_cholesky_numba(combined_volatilities, filtered_corr)
        except np.linalg.LinAlgError:
            # En cas d'erreur (matrice non définie positive), ajouter une petite perturbation
            n = len(filtered_corr)
            adjusted_corr = filtered_corr + np.eye(n) * 1e-6
            return self._calculate_cholesky_numba(combined_volatilities, adjusted_corr)
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