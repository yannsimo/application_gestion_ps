
from datetime import datetime
from  .Basket import  Basket
from typing import List, Dict

class StructuredProduct:
    def __init__(self, initial_date: datetime, final_date: datetime,
                 observation_dates: List[datetime], initial_value: float = 1000.0):
        self.initial_date = initial_date
        self.final_date = final_date
        self.observation_dates = observation_dates
        self.initial_value = initial_value
        self.basket = Basket()
        self.min_guaranteed_performance = None

    def calculate_final_performance(self, market_data) -> float:
        """Calcule la performance finale avec les contraintes spécifiques"""
        basket_perf = self.basket.calculate_performance(
            market_data,
            self.initial_date,
            self.final_date
        )

        # Application des règles spécifiques
        if basket_perf < 0:
            basket_perf = max(basket_perf, -15)  # Protection à -15%
        else:
            basket_perf = min(basket_perf, 50)  # Plafond à +50%

        # Application de la garantie de 20% si activée
        if self.min_guaranteed_performance is not None:
            basket_perf = max(basket_perf, 20)

        return basket_perf

    def calculate_dividend(self, market_data, observation_date: datetime) -> float:
        """Calcule le dividende avec exclusion de l'indice le plus performant"""
        if observation_date not in self.observation_dates:
            return 0

        active_indices = self.basket.get_active_indices()
        performances = {}

        # Calcul des performances pour chaque indice actif
        for idx in active_indices:
            perf = market_data.get_return(idx, observation_date)
            if perf is not None:
                performances[idx] = perf

        if not performances:
            return 0

        # Trouve l'indice avec la meilleure performance
        best_index = max(performances.items(), key=lambda x: x[1])[0]
        best_performance = performances[best_index]

        # Exclut l'indice pour les prochains calculs de dividendes
        self.basket.excluded_indices.append(best_index)

        # Le dividende est 50 fois la meilleure performance
        return 50 * best_performance

    def check_minimum_guarantee(self, market_data, observation_date: datetime) -> bool:
        """Vérifie si la performance annuelle déclenche la garantie de 20%"""
        annual_perf = self.basket.calculate_annual_performance(market_data, observation_date)
        if annual_perf >= 20:
            self.min_guaranteed_performance = 20
            return True
        return False
