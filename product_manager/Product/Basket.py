
from .Index import Index
from datetime import datetime
from enum import Enum
from typing import List, Dict
import numpy as np
class Basket:
    def __init__(self):
        # Initialisation avec les 5 indices spécifiques
        self.indices = [idx.value for idx in Index]
        self.weights = {idx.value: 0.2 for idx in Index}  # Poids égaux de 20%
        self.excluded_indices = []

    def calculate_performance(self, market_data, start_date: datetime, end_date: datetime) -> float:
        """Calcule la performance moyenne des 5 indices"""
        performances = []
        for idx in self.indices:
            start_price = market_data.get_price(idx, start_date)
            end_price = market_data.get_price(idx, end_date)
            if start_price and end_price and start_price > 0:
                perf = (end_price / start_price - 1) * 100
                performances.append(perf * self.weights[idx])

        return sum(performances) if performances else 0

    def calculate_annual_performance(self, market_data, date: datetime) -> float:
        """Calcule la performance annuelle des indices non exclus"""
        performances = []
        valid_indices = [idx for idx in self.indices if idx not in self.excluded_indices]
        for idx in valid_indices:
            perf = market_data.get_return(idx, date)
            if perf is not None:
                performances.append(perf)
        return np.mean(performances) if performances else 0

    def get_active_indices(self) -> List[str]:
        """Retourne la liste des indices non exclus"""
        return [idx for idx in self.indices if idx not in self.excluded_indices]