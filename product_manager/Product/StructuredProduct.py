from datetime import datetime
from .Basket import Basket
from typing import List, Dict
from .parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData
from numba import jit, njit, float64


class StructuredProduct:
    def __init__(self, initial_date: datetime, final_date: datetime,
                 observation_dates: List[datetime], initial_value: float = 1000.0):
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        self.initial_date = initial_date
        self.final_date = final_date
        self.observation_dates = observation_dates
        self.initial_value = initial_value
        self.basket = Basket(self.market_data, self.product_parameter)
        self.min_guaranteed_performance = None
        self.dividend_multiplier = 50
        self.excluded_indices = set()

        # Précalculer les dates importantes pour éviter les recherches répétées
        self._observation_dates_set = set(observation_dates)
        self._previous_observation_cache = {}

    def calculate_dividends(self, observation_date: datetime) -> float:
        """
        Calcule les dividendes distribués à une date spécifique (T1, ..., T4).
        Exclut définitivement l'indice ayant la meilleure rentabilité.
        """
        observation_dates = self.product_parameter.observation_dates

        # Vérification rapide avec l'ensemble précalculé
        if observation_date not in self._observation_dates_set:
            return 0.0  # Pas de dividende si ce n'est pas une date d'observation

        # Utiliser le cache pour la date d'observation précédente
        if observation_date in self._previous_observation_cache:
            previous_observation = self._previous_observation_cache[observation_date]
        else:
            # Trouver la dernière date d'observation avant celle-ci
            previous_observation = min(observation_dates)  # T0
            for date in observation_dates:
                if date >= observation_date:
                    break
                previous_observation = date

            # Stocker dans le cache
            self._previous_observation_cache[observation_date] = previous_observation

        # Calculer la rentabilité maximale et récupérer l'indice correspondant
        best_index, best_return = self.basket.calculate_max_annual_return(previous_observation, observation_date)

        if best_index is None:
            return 0.0  # Aucun dividende si aucun retour disponible

        # Exclure définitivement cet indice des futurs calculs
        self.excluded_indices.add(best_index)
        self.basket.excluded_indices.add(best_index)  # Ajouter aussi au panier

        # Calcul du dividende optimisé avec Numba
        return self._calculate_dividend(best_return, self.dividend_multiplier)

    @staticmethod
    @njit
    def _calculate_dividend(best_return: float, multiplier: float) -> float:
        """Calcule le dividende avec Numba pour plus de rapidité"""
        return multiplier * best_return

    def calculate_final_performance(self) -> float:
        """Calcule la performance finale avec les contraintes spécifiques"""
        # Suppression du paramètre market_data inutile
        basket_perf = self.basket.calculate_performance(
            self.product_parameter.key_dates.T0,
            self.product_parameter.key_dates.Tc
        )

        # Application des règles spécifiques avec Numba
        basket_perf = self._apply_performance_constraints(
            basket_perf,
            -0.15,  # Protection à -15%
            0.5,  # Plafond à +50%
            0.2 if self.min_guaranteed_performance is not None else None  # Garantie à 20% si activée
        )

        return basket_perf

    @staticmethod
    @njit
    def _apply_performance_constraints(perf: float, min_negative: float, max_positive: float,
                                       guaranteed_perf: float = None) -> float:
        """Applique les contraintes de performance avec Numba"""
        if perf < 0:
            result = max(perf, min_negative)
        else:
            result = min(perf, max_positive)

        # Appliquer la garantie si elle est activée
        if guaranteed_perf is not None:
            result = max(result, guaranteed_perf)

        return result

    def check_minimum_guarantee(self, observation_date: datetime) -> bool:
        """Vérifie si la performance annuelle déclenche la garantie de 20%"""
        # Trouver la date précédente (utiliser le cache si possible)
        if observation_date in self._previous_observation_cache:
            previous_date = self._previous_observation_cache[observation_date]
        else:
            previous_date = self.initial_date
            for date in self.observation_dates:
                if date >= observation_date:
                    break
                previous_date = date
            self._previous_observation_cache[observation_date] = previous_date

        annual_perf = self.basket.calculate_annual_basket_performance(previous_date, observation_date)

        # Vérification optimisée avec Numba
        if self._check_guarantee_threshold(annual_perf, 0.20):
            self.min_guaranteed_performance = 0.20
            return True
        return False

    @staticmethod
    @njit
    def _check_guarantee_threshold(performance: float, threshold: float) -> bool:
        """Vérifie si la performance atteint le seuil avec Numba"""
        return performance >= threshold

    def calculate_final_liquidative_value(self) -> float:
        """
        Calcule la valeur liquidative finale du produit à la date d'échéance (Tc).

        Returns:
            Valeur liquidative = valeur initiale + 40% de la performance finale
        """
        # Calculer la performance finale
        final_performance = self.calculate_final_performance()

        # Appliquer le coefficient de participation avec Numba
        return self._calculate_final_value(self.initial_value, final_performance, 0.40)

    @staticmethod
    @njit
    def _calculate_final_value(initial_value: float, performance: float, participation: float) -> float:
        """Calcule la valeur finale avec Numba"""
        return initial_value * (1 + participation * performance)

    def simulate_product_lifecycle(self) -> Dict:
        """
        Simule le cycle de vie complet du produit structuré, y compris:
        - Les dividendes versés à chaque date d'observation
        - La vérification de la garantie à 20%
        - La performance finale et la valeur liquidative

        Returns:
            Un dictionnaire contenant les résultats de la simulation
        """
        results = {
            'dividends': {},
            'guarantee_activated': False,
            'guarantee_activation_date': None,
            'final_performance': 0,
            'final_value': 0
        }

        # Parcourir toutes les dates d'observation
        for date in self.observation_dates:
            # Calculer le dividende à cette date
            dividend = self.calculate_dividends(date)
            results['dividends'][date] = dividend

            # Vérifier si la garantie de 20% est activée
            if not results['guarantee_activated'] and self.check_minimum_guarantee(date):
                results['guarantee_activated'] = True
                results['guarantee_activation_date'] = date

        # Calculer la performance finale
        results['final_performance'] = self.calculate_final_performance()

        # Calculer la valeur liquidative finale
        results['final_value'] = self.calculate_final_liquidative_value()

        return results
