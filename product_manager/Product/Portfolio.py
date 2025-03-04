from .Index import Index
from datetime import datetime
from .parameter.ProductParameters import ProductParameters
from ..Data.SingletonMarketData import SingletonMarketData

class Portfolio:
    def __init__(self, initial_capital: float = 1000.0):
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        self.initial_capital = initial_capital
        self.positions = {idx: 0 for idx in Index}  # Index -> quantité
        self.current_prices = {idx: 0 for idx in Index}  # Index -> prix actuel
        self._initial_prices = {}  # Pour stocker les prix initiaux
        self._is_initialized = False
        self.excluded_indices = set()  # Indices exclus après versement de dividende
        self.dividend_multiplier = 50 

    def initialize_equal_weights(self, market_data, date: datetime):
        """Initialise le portfolio avec des poids égaux (une seule fois)"""
        if self._is_initialized:
            return

        amount_per_index = self.initial_capital / len(Index)

        for idx in Index:
            price = market_data.get_price(idx.value, date)
            if price and price > 0:
                # Calcule la quantité initiale
                quantity = amount_per_index / price
                self.positions[idx] = quantity
                self.current_prices[idx] = price
                self._initial_prices[idx] = price
        self._is_initialized = True

    def update_prices(self, market_data, date: datetime):
        """Met à jour uniquement les prix sans modifier les quantités"""
        for idx in Index:
            new_price = market_data.get_price(idx.value, date)
            if new_price and new_price > 0:
                self.current_prices[idx] = new_price

    def get_total_value(self) -> float:
        """Calcule la valeur totale du portefeuille"""
        return sum(self.positions[idx] * self.current_prices[idx] for idx in Index)

    def get_position_value(self, index: Index) -> float:
        """Calcule la valeur d'une position spécifique"""
        return self.positions[index] * self.current_prices[index]

    def get_position_weight(self, index: Index) -> float:
        """Calcule le poids d'une position dans le portefeuille"""
        position_value = self.get_position_value(index)
        total_value = self.get_total_value()
        return (position_value / total_value * 100) if total_value > 0 else 0

    def get_pnl(self) -> float:
        """Calcule le P&L en pourcentage"""
        current_value = self.get_total_value()
        return ((current_value - self.initial_capital) / self.initial_capital) * 100
    def get_annual_returns(self, start_date: datetime, end_date: datetime):
        """Calcule les rentabilités annuelles de chaque indice entre start_date et end_date."""
        annual_returns = {}

        for idx in Index:
            if idx in self.excluded_indices:  # Ne pas inclure les indices déjà exclus
                continue  

            start_price = self.market_data.get_price(idx.value, start_date)* self.market_data.get_exchange_rate(idx.value,start_date)
            end_price = self.market_data.get_price(idx.value, end_date)*self.market_data.get_exchange_rate(idx.value,start_date)

            if start_price is None or end_price is None or start_price == 0:
                continue  # Ignorer si on n'a pas de données valides

            # Rentabilité annuelle = (Prix final / Prix initial) - 1
            annual_returns[idx] = (end_price / start_price) - 1  

        return annual_returns
    def calculate_max_annual_return(self, start_date: datetime, end_date: datetime) -> dict:
        """
        Calcule la rentabilité maximale annuelle entre start_date et end_date
        pour les indices actifs (non exclus).
        """
        active_indices = [idx for idx in Index if idx not in self.excluded_indices]

        max_annual_returns = {}
        for index in active_indices:
            returns = self.get_annual_returns(start_date, end_date)
            if returns:
                max_annual_returns[index] = max(returns)

        return max_annual_returns

    def calculate_dividends(self, market_data, observation_date: datetime) -> float:
        """
        Calcule les dividendes distribués à une date spécifique (T1, ..., T4).
        Exclut définitivement l'indice ayant la meilleure rentabilité.
        """
        observation_dates =  self.product_parameter.get_observation_dates

        if observation_date not in observation_dates:
            return 0.0  # Pas de dividende si ce n'est pas une date d'observation

        # Trouver la dernière date d'observation avant celle-ci
        previous_observation = min(observation_dates)  # T0
        for date in observation_dates:
            if date >= observation_date:
                break
            previous_observation = date

        # Calculer la rentabilité maximale sur les indices restants
        max_annual_returns = self.calculate_max_annual_return(market_data, previous_observation, observation_date)

        if not max_annual_returns:
            return 0.0  # Aucun dividende si aucun retour disponible

        # Trouver l'indice avec la meilleure performance
        best_index = max(max_annual_returns, key=max_annual_returns.get)
        best_return = max_annual_returns[best_index]

        # Exclure définitivement cet indice des futurs calculs
        self.excluded_indices.add(best_index)

        # Calcul du dividende : 50 × performance max
        dividend = self.dividend_multiplier * best_return

        return dividend