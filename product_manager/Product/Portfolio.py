from .Index import Index
from datetime import datetime


class Portfolio:
    def __init__(self, initial_capital: float = 1000.0):
        self.initial_capital = initial_capital
        self.positions = {idx: 0 for idx in Index}  # Index -> quantité
        self.current_prices = {idx: 0 for idx in Index}  # Index -> prix actuel
        self._initial_prices = {}  # Pour stocker les prix initiaux
        self._is_initialized = False

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
