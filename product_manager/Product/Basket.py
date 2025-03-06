from. Index import Index
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from .parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData
from numba import jit, njit, float64


class Basket:
    def __init__(self, market_data, product_parameter):
        self.market_data = market_data
        self.product_parameter = product_parameter
        self.excluded_indices = set()  # Indices exclus après versement de dividende

        # Caches pour stocker les prix et taux déjà récupérés
        self._price_cache = {}
        self._exchange_rate_cache = {}
        self._returns_cache = {}

    def get_annual_returns_simulater(self, start_date: datetime, end_date: datetime):
        """Calcule les rentabilités annuelles de chaque indice entre start_date et end_date."""
        # Vérifier si les résultats sont déjà en cache
        cache_key = f"{start_date}_{end_date}"
        if cache_key in self._returns_cache:
            return self._returns_cache[cache_key]
        annual_returns = {}

        # Préparer les données pour l'accélération Numba
        indices = []
        prices_start = []
        prices_end = []
        rates_start = []
        rates_end = []

        for idx in Index:
            if idx in self.excluded_indices:  # Ne pas inclure les indices déjà exclus
                continue

            # Utilisation des méthodes get_last_available_* pour obtenir des données robustes
            start_price_local = self.market_data.get_last_available_price(idx.value, start_date) #
            end_price_local = self.market_data.get_last_available_price(idx.value, end_date)

            start_exchange_rate = self.market_data.get_last_available_exchange_rate(idx.value, start_date)
            end_exchange_rate = self.market_data.get_last_available_exchange_rate(idx.value, end_date)

            if (start_price_local is None or end_price_local is None or
                    start_exchange_rate is None or end_exchange_rate is None or
                    start_price_local == 0):
                continue  # Ignorer si on n'a pas de données valides

            # Ajouter aux tableaux pour traitement par lots
            indices.append(idx)
            prices_start.append(start_price_local)
            prices_end.append(end_price_local)
            rates_start.append(start_exchange_rate)
            rates_end.append(end_exchange_rate)

        # Utiliser Numba pour calculer les rendements en une seule opération
        if indices:
            returns = self._calculate_returns(
                np.array(prices_start),
                np.array(prices_end),
                np.array(rates_start),
                np.array(rates_end)
            )

            # Reconstruire le dictionnaire de résultats
            for i, idx in enumerate(indices):
                annual_returns[idx] = returns[i]

        # Mettre en cache les résultats
        self._returns_cache[cache_key] = annual_returns
        return annual_returns

    @staticmethod
    @njit
    def _calculate_returns(prices_start, prices_end, rates_start, rates_end):
        """Calcule les rendements en une seule opération avec Numba"""
        n = len(prices_start)
        returns = np.zeros(n)

        for i in range(n):
            # Convertir en EUR pour calculer la rentabilité
            start_price = prices_start[i] * rates_start[i]
            end_price = prices_end[i] * rates_end[i]

            # Rentabilité annuelle = (Prix final / Prix initial) - 1
            returns[i] = (end_price / start_price) - 1

        return returns

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

        # Trouver l'indice avec le rendement maximum avec Numba
        best_index = None
        best_return = -float('inf')

        for idx, ret in returns.items():
            if ret > best_return:
                best_return = ret
                best_index = idx

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

        # Calculer la moyenne des rentabilités annuelles avec Numba
        values = list(annual_returns.values())
        return self._calculate_average(np.array(values))

    @staticmethod
    @njit
    def _calculate_average(values):
        """Calcule la moyenne d'un tableau avec Numba"""
        if len(values) == 0:
            return 0.0
        return np.sum(values) / len(values)

    def calculate_performance(self, initial_date, final_date):
        """Calcule la performance du panier d'indices en tenant compte des taux de change"""
        # Préparer les tableaux pour l'accélération Numba
        initial_prices_eur = []
        final_prices_eur = []

        # Parcourir les indices
        for idx in Index:
            # Utilisation des méthodes get_last_available_* pour obtenir des données robustes
            initial_price_local = self.market_data.get_last_available_price(idx.value, initial_date)
            final_price_local = self.market_data.get_last_available_price(idx.value, final_date)

            initial_fx = self.market_data.get_last_available_exchange_rate(idx.value, initial_date)
            final_fx = self.market_data.get_last_available_exchange_rate(idx.value, final_date)

            if (initial_price_local is None or final_price_local is None or
                    initial_fx is None or final_fx is None or
                    initial_price_local == 0):
                continue

            # Convertir en EUR pour calculer la performance réelle
            initial_price_eur = initial_price_local * initial_fx
            final_price_eur = final_price_local * final_fx

            initial_prices_eur.append(initial_price_eur)
            final_prices_eur.append(final_price_eur)

        # Utiliser Numba pour calculer les performances et la moyenne
        if not initial_prices_eur:
            return 0.0  # Liste vide

        performances = self._calculate_performances(
            np.array(initial_prices_eur),
            np.array(final_prices_eur)
        )

        return self._calculate_average(performances)

    @staticmethod
    @njit
    def _calculate_performances(initial_prices, final_prices):
        """Calcule les performances individuelles avec Numba"""
        n = len(initial_prices)
        performances = np.zeros(n)

        for i in range(n):
            # Performance individuelle = (Prix final en EUR / Prix initial en EUR) - 1
            performances[i] = (final_prices[i] / initial_prices[i]) - 1

        return performances

    def calculate_annual_indices(self, start_date: datetime, end_date: datetime):
        """Calcule les rentabilités annuelles de chaque indice entre start_date et end_date."""
        # Cette méthode étant très similaire à get_annual_returns, on la réutilise
        # mais en filtrant différemment les indices exclus

        # Vérifier si les résultats sont déjà en cache
        cache_key = f"annual_{start_date}_{end_date}"
        if cache_key in self._returns_cache:
            return self._returns_cache[cache_key]

        annual_returns = {}

        # Préparer les données pour l'accélération Numba
        indices = []
        prices_start = []
        prices_end = []
        rates_start = []
        rates_end = []

        for idx in Index:
            if idx in self.excluded_indices:
                continue

            # Utilisation des méthodes get_last_available_* pour obtenir des données robustes
            start_price_local = self.market_data.get_last_available_price(idx.value, start_date)
            end_price_local = self.market_data.get_last_available_price(idx.value, end_date)

            start_exchange_rate = self.market_data.get_last_available_exchange_rate(idx.value, start_date)
            end_exchange_rate = self.market_data.get_last_available_exchange_rate(idx.value, end_date)

            if (start_price_local is None or end_price_local is None or
                    start_exchange_rate is None or end_exchange_rate is None or
                    start_price_local == 0):
                continue  # Ignorer si on n'a pas de données valides

            # Ajouter aux tableaux pour traitement par lots
            indices.append(idx)
            prices_start.append(start_price_local)
            prices_end.append(end_price_local)
            rates_start.append(start_exchange_rate)
            rates_end.append(end_exchange_rate)

        # Utiliser Numba pour calculer les rendements en une seule opération
        if indices:
            returns = self._calculate_returns(
                np.array(prices_start),
                np.array(prices_end),
                np.array(rates_start),
                np.array(rates_end)
            )

            # Reconstruire le dictionnaire de résultats
            for i, idx in enumerate(indices):
                annual_returns[idx] = returns[i]

        # Mettre en cache les résultats
        self._returns_cache[cache_key] = annual_returns
        return annual_returns