from .Index import Index
from datetime import datetime
from enum import Enum
from typing import List, Dict
import numpy as np
from .parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData


class Basket:
    def __init__(self, market_data, product_parameter):
        self.market_data = market_data
        self.product_parameter = product_parameter
        self.excluded_indices = set()  # Indices exclus après versement de dividende

    def get_annual_returns(self, start_date: datetime, end_date: datetime):
        """Calcule les rentabilités annuelles de chaque indice entre start_date et end_date."""
        annual_returns = {}

        for idx in Index:
            if idx in self.excluded_indices:  # Ne pas inclure les indices déjà exclus
                continue

            start_price = self.market_data.get_price(idx.value, start_date) * self.market_data.get_exchange_rate(
                idx.value, start_date)
            end_price = self.market_data.get_price(idx.value, end_date) * self.market_data.get_exchange_rate(idx.value,
                                                                                                             end_date)

            if start_price is None or end_price is None or start_price == 0:
                continue  # Ignorer si on n'a pas de données valides

            # Rentabilité annuelle = (Prix final / Prix initial) - 1
            annual_returns[idx] = (end_price / start_price) - 1

        return annual_returns

    def calculate_max_annual_return(self, start_date: datetime, end_date: datetime):
        """
        Calcule la rentabilité maximale annuelle entre start_date et end_date
        pour les indices actifs (non exclus).

        Returns:
            Tuple contenant (indice_max, rendement_max) ou (None, 0) si aucun rendement
        """
        returns = self.get_annual_returns(start_date, end_date)

        # Si aucun rendement n'est disponible
        if not returns:
            return None, 0

        # Trouver l'indice avec le rendement maximum
        best_index = max(returns, key=returns.get)
        best_return = returns[best_index]

        return best_index, best_return

    def calculate_annual_basket_performance(self, previous_date, current_date):
        """
        Calcule la performance annuelle du panier entre deux dates de constatation
        consécutives (Ti-1 et Ti).

        Args:
            previous_date: Date de constatation précédente (Ti-1)
            current_date: Date de constatation actuelle (Ti)

        Returns:
            La moyenne des rentabilités annuelles des indices
        """
        # Calculer les rendements annuels pour chaque indice
        annual_returns = self.calculate_annual_indices(previous_date, current_date)

        # Si aucun rendement n'est disponible
        if not annual_returns:
            return 0.0

        # Calculer la moyenne des rentabilités annuelles
        total_return = sum(annual_returns.values())
        average_return = total_return / len(annual_returns)

        return average_return

    def calculate_performance(self, initial_date, final_date):
        """Calcule la performance du panier d'indices en tenant compte des taux de change"""
        performances = []

        # Parcourir les 5 indices spécifiés (ASX200, DAX, FTSE100, NASDAQ100, SMI)
        for idx in Index:
            # Pour la performance finale, nous incluons tous les indices (pas d'exclusion)
            # Les exclusions ne s'appliquent qu'au calcul des dividendes intermédiaires
            # Obtenir les prix initiaux et finaux en devise locale
            initial_price_local = self.market_data.get_price(idx.value, initial_date)
            final_price_local = self.market_data.get_price(idx.value, final_date)

            # Obtenir les taux de change initiaux et finaux
            initial_fx = self.market_data.get_index_exchange_rate(idx.value, initial_date)
            final_fx = self.market_data.get_index_exchange_rate(idx.value, final_date)

            if (initial_price_local is None or final_price_local is None or
                    initial_fx is None or final_fx is None or
                    initial_price_local == 0):
                continue

            # Convertir en EUR pour calculer la performance réelle
            initial_price_eur = initial_price_local * initial_fx
            final_price_eur = final_price_local * final_fx

            # Performance individuelle = (Prix final en EUR / Prix initial en EUR) - 1
            individual_perf = (final_price_eur / initial_price_eur) - 1
            performances.append(individual_perf)

        # Performance du panier = moyenne des performances individuelles des 5 indices

        basket_perf = sum(performances) / len(performances)

        return basket_perf

    def calculate_annual_indices(self, start_date: datetime, end_date: datetime):
         """Calcule les rentabilités annuelles de chaque indice entre start_date et end_date."""
         annual_returns = {}

         for idx in Index:
             start_price = self.market_data.get_price(idx.value, start_date) * self.market_data.get_exchange_rate(
                 idx.value, start_date)
             end_price = self.market_data.get_price(idx.value, end_date) * self.market_data.get_exchange_rate(idx.value,
                                                                                                              end_date)
             if start_price is None or end_price is None or start_price == 0:
                 continue  # Ignorer si on n'a pas de données valides

             # Rentabilité annuelle = (Prix final / Prix initial) - 1
             annual_returns[idx] = (end_price / start_price) - 1

         return annual_returns


