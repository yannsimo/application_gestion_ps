from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import numpy as np


@dataclass
class Basket:
    indices: List[str]  # Liste des codes des indices
    weights: Dict[str, float]  # Poids de chaque indice
    excluded_indices: List[str] = None  # Indices exclus pour les dividendes

    def __init__(self, indices: List[str], equal_weight: bool = True):
        self.indices = indices
        self.weights = {idx: 1 / len(indices) for idx in indices} if equal_weight else {}
        self.excluded_indices = []

    def calculate_performance(self, market_data, start_date: datetime, end_date: datetime) -> float:
        """Calcule la performance du panier entre deux dates"""
        total_perf = 0
        for idx in self.indices:
            start_price = market_data.get_index_price(idx, start_date)
            end_price = market_data.get_index_price(idx, end_date)
            if start_price and end_price:
                perf = (end_price / start_price - 1) * 100
                total_perf += perf * self.weights[idx]
        return total_perf

    def calculate_annual_performance(self, market_data, date: datetime) -> float:
        """Calcule la performance annuelle du panier à une date donnée"""
        performances = []
        for idx in [i for i in self.indices if i not in self.excluded_indices]:
            perf = market_data.get_index_return(idx, date)
            if perf is not None:
                performances.append(perf)
        return np.mean(performances) if performances else 0