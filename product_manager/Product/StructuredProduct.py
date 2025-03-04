from datetime import datetime
from .Basket import Basket
from typing import List, Dict
from .parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData

class StructuredProduct:
    def __init__(self, initial_date: datetime, final_date: datetime,
                 observation_dates: List[datetime], initial_value: float = 1000.0):
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        self.initial_date = initial_date
        self.final_date = final_date
        self.observation_dates = observation_dates
        self.initial_value = initial_value
        self.basket = Basket(self.market_data, self.product_parameter)  # Passez les paramètres requis
        self.min_guaranteed_performance = None
        self.dividend_multiplier = 50
        self.excluded_indices = set()  # Ajout pour corriger l'erreur

    def calculate_dividends(self, observation_date: datetime) -> float:
        """
        Calcule les dividendes distribués à une date spécifique (T1, ..., T4).
        Exclut définitivement l'indice ayant la meilleure rentabilité.
        """
        observation_dates = self.product_parameter.key_dates

        if observation_date not in observation_dates:
            return 0.0  # Pas de dividende si ce n'est pas une date d'observation

        # Trouver la dernière date d'observation avant celle-ci
        previous_observation = min(observation_dates)  # T0
        for date in observation_dates:
            if date >= observation_date:
                break
            previous_observation = date

        # Calculer la rentabilité maximale et récupérer l'indice correspondant
        best_index, best_return = self.basket.calculate_max_annual_return(previous_observation, observation_date)

        if best_index is None:
            return 0.0  # Aucun dividende si aucun retour disponible

        # Exclure définitivement cet indice des futurs calculs
        self.excluded_indices.add(best_index)
        self.basket.excluded_indices.add(best_index)  # Ajouter aussi au panier

        # Calcul du dividende : 50 × performance max
        dividend = self.dividend_multiplier * best_return

        return dividend

    def calculate_final_performance(self) -> float:
        """Calcule la performance finale avec les contraintes spécifiques"""
        # Suppression du paramètre market_data inutile
        basket_perf = self.basket.calculate_performance( self.product_parameter.key_dates.T0, self.product_parameter.key_dates.TC)

        # Application des règles spécifiques
        if basket_perf < 0:
            basket_perf = max(basket_perf, -15)  # Protection à -15%
        else:
            basket_perf = min(basket_perf, 50)  # Plafond à +50%

        # Application de la garantie de 20% si activée
        if self.min_guaranteed_performance is not None:
            basket_perf = max(basket_perf, 20)

        return basket_perf

    def check_minimum_guarantee(self, observation_date: datetime) -> bool:
        """Vérifie si la performance annuelle déclenche la garantie de 20%"""
        # Trouver la date précédente
        previous_date = self.initial_date
        for date in self.observation_dates:
            if date >= observation_date:
                break
            previous_date = date

        annual_perf = self.basket.calculate_annual_basket_performance(previous_date, observation_date)
        if annual_perf >= 0.20:  # 20%
            self.min_guaranteed_performance = 0.20
            return True
        return False

    def calculate_final_liquidative_value(self) -> float:
        """
        Calcule la valeur liquidative finale du produit à la date d'échéance (Tc).

        Returns:
            Valeur liquidative = valeur initiale + 40% de la performance finale
        """
        # Calculer la performance finale (avec bornes -15%/+50% et garantie de 20% si activée)
        final_performance = self.calculate_final_performance()

        # Appliquer le coefficient de participation de 40%
        participation_factor = 0.40

        # Calculer la valeur liquidative finale
        final_value = self.initial_value * (1 + participation_factor * final_performance)

        return final_value

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
            if self.check_minimum_guarantee(date):
                results['guarantee_activated'] = True
                results['guarantee_activation_date'] = date

        # Calculer la performance finale
        results['final_performance'] = self.calculate_final_performance()

        # Calculer la valeur liquidative finale
        results['final_value'] = self.calculate_final_liquidative_value()

        return results