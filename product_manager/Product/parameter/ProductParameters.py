from datetime import datetime
from typing import Dict
import numpy as np
from ...Product.Index import Index
from ...Product.parameter.date.structured_product_dates import KEY_DATES_AUTO
from .Volatility.VolatilityCalculator import VolatilityCalculator


class ProductParameters:
    def __init__(self, market_data, current_date: datetime):
        self.market_data = market_data
        self.current_date = current_date
        self.underlying_indices = [index.value for index in Index]
        self.key_dates = KEY_DATES_AUTO
        self.initial_date = self.key_dates.T0
        self.final_date = self.key_dates.Tc
        self.observation_dates = [self.key_dates.get_Ti(i) for i in range(1, 5)]  # T1 à T4
        self.excluded_indices = set()  # Liste des indices exclus après versement d'un dividende
        self.num_simulations = 10000
        self.initial_value = 1000.0
        self.participation_rate = 0.4
        self.cap = 0.5
        self.floor = -0.15
        self.minimum_guarantee = 0.2
        self.dividend_multiplier = 50
        self.volatility_calculator = VolatilityCalculator(self.market_data)
        self.update_market_parameters()
    

    def update_market_parameters(self):
        """Met à jour les paramètres de marché basés sur la date courante."""
        self.volatilities = self._calculate_volatilities()
        self.volatilities_echange = self._calculate_volatilities_echange()
        self.volatilities_rate = self._calculate_volatilities_rate()
        self.risk_free_rates = self._calculate_risk_free_rates()
        self.cholesky_matrix = self._calculate_cholesky_matrix()

    def _calculate_volatilities_echange(self) -> Dict[str, float]:
        """Calcule les volatilités pour le taux de change de chaque indice."""
        return {index: self.volatility_calculator.calculate_volatility_echange(index, self.current_date)
                for index in self.underlying_indices}

    def _calculate_volatilities_rate(self) -> Dict[str, float]:
        """Calcule les volatilités pour le taux de change de chaque indice."""
        return {index: self.volatility_calculator.calculate_volatility_rate(index, self.current_date)
                for index in self.underlying_indices}

    def _calculate_volatilities(self) -> Dict[str, float]:
        """Calcule les volatilités pour chaque indice."""
        return {index: self.volatility_calculator.calculate_volatility(index, self.current_date)
                for index in self.underlying_indices}

    def _calculate_risk_free_rates(self) -> Dict[str, float]:
        """Calcule les taux sans risque pour chaque indice."""
        return {index: self.market_data.get_index_interest_rate(index, self.current_date)
                for index in self.underlying_indices}

    def _calculate_cholesky_matrix(self) -> np.ndarray:
        """Calcule la matrice de Cholesky."""
        return self.volatility_calculator.calculate_vol_cholesky(self.underlying_indices, self.current_date)

    def get_time_to_maturity(self) -> float:
        """Calcule le temps jusqu'à la maturité en années."""
        return (self.final_date - self.current_date).days / 365.0

    def is_observation_date(self, date: datetime) -> bool:
        """Vérifie si une date donnée est une date d'observation."""
        return date in self.observation_dates

    def get_next_observation_date(self) -> datetime:
        """Retourne la prochaine date d'observation après la date courante."""
        for date in self.observation_dates:
            if date > self.current_date:
                return date
        return self.final_date

    def calculate_performance(self, initial_prices: dict, final_prices: dict) -> float:
        """Calcule la performance du panier d'indices."""
        performances = [
            (final_prices[index] / initial_prices[index] - 1)
            for index in self.underlying_indices
        ]
        basket_performance = sum(performances) / len(performances)

        if basket_performance < 0:
            return max(basket_performance, self.floor)
        else:
            return min(basket_performance, self.cap)

    def apply_minimum_guarantee(self, performance: float) -> float:
        """Applique la garantie minimale si nécessaire."""
        return max(performance, self.minimum_guarantee)

    def calculate_final_payout(self, performance: float) -> float:
        """Calcule le paiement final du produit."""
        return self.initial_value * (1 + self.participation_rate * performance)

    def update_date(self, new_date: datetime):
        """Met à jour la date courante et recalcule les paramètres de marché."""
        self.current_date = new_date
        self.update_market_parameters()


# Exemple d'utilisation
if __name__ == "__main__":
    from ...views import SingletonMarketData  # Assurez-vous que ce chemin d'importation est correct

    market_data = SingletonMarketData.get_instance()
    current_date = datetime.now()
    params = ProductParameters(market_data, current_date)

    print(f"Date courante : {params.current_date}")
    print(f"Indices sous-jacents : {params.underlying_indices}")
    print(f"Volatilités : {params.volatilities}")
    print(f"Taux sans risque : {params.risk_free_rates}")
    print(f"Matrice de Cholesky :\n{params.cholesky_matrix}")
    print(f"Temps jusqu'à maturité : {params.get_time_to_maturity()} années")