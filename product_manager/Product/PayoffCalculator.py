from datetime import datetime
from typing import List, Dict
from .StructuredProduct import StructuredProduct
import numpy as np
from ..Data.SingletonMarketData import SingletonMarketData
from numba import jit, float64, njit


class PayoffCalculator:
    def __init__(self, structured_product: StructuredProduct):
        """
        Initialise le calculateur de payoff avec un produit structuré.

        Args:
            structured_product: Le produit structuré dont on veut calculer le payoff
        """
        self.product = structured_product
        self.market_data = SingletonMarketData.get_instance()

        # Simuler le cycle de vie du produit une seule fois lors de l'initialisation
        self.lifecycle_results = self.product.simulate_product_lifecycle()

    def _get_risk_free_rate(self, date: datetime) -> float:
        """Obtient le taux sans risque à une date donnée"""
        # On utilise l'EUR comme devise de référence
        return self.market_data.get_interest_rate("REUR", date)

    def calculate_discount_factor(self, from_date: datetime, to_date: datetime) -> float:
        """
        Calcule le facteur d'actualisation entre deux dates.

        Args:
            from_date: Date initiale
            to_date: Date finale

        Returns:
            Le facteur d'actualisation e^(-r*(to_date-from_date))
        """
        # Obtenir le taux sans risque
        rate = self._get_risk_free_rate(from_date)

        # Si le taux est None, utiliser une valeur par défaut raisonnable
        if rate is None:
            rate = 0.03  # 3% comme taux par défaut

        # Calculer la durée en années
        duration_years = (to_date - from_date).days / 365.0

        # Utiliser la fonction optimisée par Numba
        return self._calculate_discount_factor(rate, duration_years)

    @staticmethod
    @njit
    def _calculate_discount_factor(rate: float, duration_years: float) -> float:
        """Version optimisée avec Numba du calcul du facteur d'actualisation"""
        return np.exp(-rate * duration_years)

    def calculate_discounted_dividends(self) -> Dict[datetime, float]:
        """
        Calcule les dividendes actualisés à chaque date d'observation.

        Returns:
            Un dictionnaire avec les dates d'observation et les dividendes actualisés
        """
        discounted_dividends = {}

        # Préparer les données pour la fonction Numba
        dates = []
        dividends = []

        # Collecter les données
        for date, dividend in self.lifecycle_results['dividends'].items():
            dates.append(date)
            dividends.append(dividend)

        # Calculer tous les facteurs d'actualisation en une fois
        discount_factors = self._calculate_all_discount_factors(
            self.product.initial_date,
            dates
        )

        # Calculer tous les dividendes actualisés en une fois
        discounted_values = self._apply_discount_factors(dividends, discount_factors)

        # Reconstruire le dictionnaire
        for i, date in enumerate(dates):
            discounted_dividends[date] = discounted_values[i]

        return discounted_dividends

    def _calculate_all_discount_factors(self, initial_date, dates):
        """Calcule tous les facteurs d'actualisation pour une liste de dates"""
        factors = []
        for date in dates:
            factors.append(self.calculate_discount_factor(initial_date, date))
        return factors

    @staticmethod
    @njit
    def _apply_discount_factors(values, factors):
        """Applique des facteurs d'actualisation à une liste de valeurs (optimisé avec Numba)"""
        result = np.zeros(len(values))
        for i in range(len(values)):
            result[i] = values[i] * factors[i]
        return result

    def calculate_discounted_final_value(self) -> float:
        """
        Calcule la valeur finale actualisée du produit.

        Returns:
            La valeur finale actualisée à la date initiale
        """
        # Récupérer la valeur finale du produit depuis les résultats du cycle de vie
        final_value = self.lifecycle_results['final_value']

        # Calculer le facteur d'actualisation
        discount_factor = self.calculate_discount_factor(self.product.initial_date, self.product.final_date)

        # Actualiser la valeur finale (utilisation de Numba non nécessaire ici car c'est une simple multiplication)
        discounted_final_value = final_value * discount_factor

        return discounted_final_value

    def calculate_total_payoff(self) -> Dict:
        """
        Calcule le payoff total du produit, en actualisant tous les flux.

        Returns:
            Un dictionnaire avec les détails du payoff
        """
        # Calculer les dividendes actualisés
        discounted_dividends = self.calculate_discounted_dividends()

        # Calculer la valeur finale actualisée
        discounted_final_value = self.calculate_discounted_final_value()

        # Calculer le payoff total avec Numba
        dividend_values = list(discounted_dividends.values())
        total_discounted_payoff = self._sum_with_value(dividend_values, discounted_final_value)

        # Préparer les résultats
        payoff_results = {
            'dividends': self.lifecycle_results['dividends'],
            'discounted_dividends': discounted_dividends,
            'final_value': self.lifecycle_results['final_value'],
            'discounted_final_value': discounted_final_value,
            'total_payoff': total_discounted_payoff,
            'guarantee_activated': self.lifecycle_results['guarantee_activated'],
            'final_performance': self.lifecycle_results['final_performance']
        }

        return payoff_results

    @staticmethod
    @njit
    def _sum_with_value(array, value):
        """Calcule la somme d'un tableau + une valeur (optimisé avec Numba)"""
        total = 0.0
        for x in array:
            total += x
        return total + value

    def calculate_fair_price(self) -> float:
        """
        Calcule le prix théorique juste du produit.

        Returns:
            Le prix théorique juste (valeur actualisée de tous les flux)
        """
        payoff_results = self.calculate_total_payoff()
        return payoff_results['total_payoff']