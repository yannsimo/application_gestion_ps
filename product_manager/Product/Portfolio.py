from .Index import Index
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

def get_singleton_market_data():
    from product_manager.Data.SingletonMarketData import SingletonMarketData
    return SingletonMarketData

class Portfolio:
    def __init__(self, initial_capital: float = 1000.0):
        self.market_data = get_singleton_market_data().get_instance()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {idx: 0 for idx in Index}  # Index -> quantity
        self.current_prices = {idx: 0 for idx in Index}  # Index -> current price
        self._initial_prices = {}  # For storing initial prices
        self.cash = initial_capital  # Cash position (risk-free)
        self._is_initialized = False
        self.excluded_indices = set()  # Indices excluded after dividend payment
        self.transaction_history = []  # Track all transactions
        self.dividend_multiplier = 50  # Multiplier for dividend calculation

    def initialize_equal_weights(self, market_data, date: datetime):
        """Initialize the portfolio with equal weights (only once)"""
        if self._is_initialized:
            return

        amount_per_index = self.initial_capital / len(Index)

        for idx in Index:
            price = market_data.get_price(idx.value, date)
            if price and price > 0:
                # Calculate initial quantity
                exchange_rate = market_data.get_index_exchange_rate(idx.value, date)
                price_eur = price * exchange_rate
                quantity = amount_per_index / price_eur if price_eur > 0 else 0
                
                self.positions[idx] = quantity
                self.current_prices[idx] = price
                self._initial_prices[idx] = price
        
        # Calculate remaining cash
        self.cash = self.initial_capital - self.get_invested_value()
        self._is_initialized = True
        
        # Record initial transaction
        self.transaction_history.append({
            'date': date,
            'type': 'initialization',
            'positions': {idx: self.positions[idx] for idx in Index},
            'cash': self.cash
        })

    def update_prices(self, market_data, date: datetime):
        """Update prices without modifying quantities"""
        for idx in Index:
            new_price = market_data.get_price(idx.value, date)
            if new_price and new_price > 0:
                self.current_prices[idx] = new_price

    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        position_value = self.get_invested_value()
        return position_value + self.cash

    def get_invested_value(self) -> float:
        """Calculate value of invested positions"""
        return sum(self.get_position_value(idx) for idx in Index)

    def get_position_value(self, index: Index) -> float:
        """Calculate value of a specific position"""
        price_eur = self.current_prices[index] * self.market_data.get_index_exchange_rate(index.value, self.market_data.current_date)
        return self.positions[index] * price_eur

    def get_position_weight(self, index: Index) -> float:
        """Calculate weight of a position in the portfolio"""
        position_value = self.get_position_value(index)
        total_value = self.get_total_value()
        return (position_value / total_value * 100) if total_value > 0 else 0

    def get_pnl(self) -> float:
        """Calculate P&L percentage"""
        current_value = self.get_total_value()
        return ((current_value - self.initial_capital) / self.initial_capital) * 100
    
    def update_positions_with_deltas(self, deltas: Dict[str, float], date: datetime):
        """
        Update portfolio positions based on delta hedging ratios
        
        Args:
            deltas: Dictionary of delta values for each index
            date: Current date
        """
        # Calculate current portfolio value
        portfolio_value = self.get_total_value()
        
        # Calculate new positions
        new_positions = {}
        new_cash = portfolio_value
        
        for index in Index:
            idx_code = index.value
            if idx_code in deltas:
                delta = deltas[idx_code]
                
                price = self.current_prices[index]
                exchange_rate = self.market_data.get_index_exchange_rate(idx_code, date)
                price_eur = price * exchange_rate
                
                if price_eur > 0:
                    # Calculate EUR exposure needed
                    eur_exposure = delta * portfolio_value
                    
                    # Convert to quantity
                    quantity = eur_exposure / price_eur
                    new_positions[index] = quantity
                    
                    # Deduct from cash
                    new_cash -= eur_exposure
                else:
                    new_positions[index] = 0
            else:
                new_positions[index] = 0
        
        # Update positions and cash
        self.positions = new_positions
        self.cash = new_cash
        
        # Record transaction
        self.transaction_history.append({
            'date': date,
            'type': 'rebalance',
            'positions': {idx: self.positions[idx] for idx in Index},
            'cash': self.cash,
            'portfolio_value': portfolio_value
        })
    
    def pay_dividend(self, amount: float, date: datetime):
        """
        Process a dividend payment
        
        Args:
            amount: Dividend amount
            date: Payment date
        """
        # Subtract dividend from cash
        self.cash -= amount
        
        # Record transaction
        self.transaction_history.append({
            'date': date,
            'type': 'dividend_payment',
            'amount': amount,
            'cash': self.cash
        })
    
    def get_annual_returns(self, start_date: datetime, end_date: datetime) -> Dict[Index, float]:
        """
        Calculate annual returns for each index between start_date and end_date
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of annual returns by index
        """
        annual_returns = {}

        for idx in Index:
            if idx in self.excluded_indices:  # Skip excluded indices
                continue  

            start_price = self.market_data.get_price(idx.value, start_date) * self.market_data.get_index_exchange_rate(idx.value, start_date)
            end_price = self.market_data.get_price(idx.value, end_date) * self.market_data.get_index_exchange_rate(idx.value, end_date)

            if start_price is None or end_price is None or start_price == 0:
                continue  # Skip if no valid data

            # Annual return = (Final price / Initial price) - 1
            annual_returns[idx] = (end_price / start_price) - 1  

        return annual_returns