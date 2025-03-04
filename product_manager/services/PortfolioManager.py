from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

class PortfolioManager:
    """
    Manages the portfolio for hedging a structured product
    
    This class handles:
    - Portfolio initialization
    - Delta hedging updates
    - Dividend payments
    - Performance tracking
    """
    def __init__(self, 
                initial_capital: float = 1000.0, 
                risk_free_rate: float = 0.0):
        """
        Initialize the portfolio manager
        
        Args:
            initial_capital: Initial capital in EUR
            risk_free_rate: Risk-free interest rate
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.positions = {}  # Index -> quantity
        self.cash = initial_capital  # Cash position (risk-free)
        self.transaction_history = []  # Track all transactions
        self.pnl_history = []  # Track PnL over time
        
    def initialize_portfolio(self, 
                            deltas: Dict[str, float], 
                            prices: Dict[str, float],
                            exchange_rates: Dict[str, float]):
        """
        Initialize the portfolio based on delta hedging ratios
        
        Args:
            deltas: Dictionary of delta values for each index
            prices: Dictionary of current prices for each index (in local currency)
            exchange_rates: Dictionary of exchange rates for each index
        """
        # Reset portfolio
        self.positions = {}
        
        # Calculate total delta exposure in EUR
        total_delta_exposure = 0.0
        for index, delta in deltas.items():
            if index not in prices or index not in exchange_rates:
                continue
                
            # Convert to EUR exposure
            index_exposure = delta * self.initial_capital
            total_delta_exposure += abs(index_exposure)
        
        # Calculate scaling factor if total exposure exceeds capital
        scaling_factor = 1.0
        if total_delta_exposure > self.initial_capital:
            scaling_factor = self.initial_capital / total_delta_exposure
        
        # Initialize positions
        remaining_capital = self.initial_capital
        
        for index, delta in deltas.items():
            if index not in prices or index not in exchange_rates:
                continue
                
            price = prices[index]
            exchange_rate = exchange_rates[index]
            
            # Calculate EUR exposure
            index_exposure = delta * self.initial_capital * scaling_factor
            
            # Calculate quantity in local currency units
            if price > 0 and exchange_rate > 0:
                # Convert EUR exposure to local currency, then divide by price
                quantity = index_exposure / (price * exchange_rate)
                self.positions[index] = quantity
                
                # Update remaining capital
                remaining_capital -= index_exposure
            else:
                self.positions[index] = 0.0
        
        # Set cash position (invested at risk-free rate)
        self.cash = remaining_capital
        
        # Record initial transaction
        self.transaction_history.append({
            'date': datetime.now(),
            'type': 'initialization',
            'positions': self.positions.copy(),
            'cash': self.cash
        })
    
    def update_portfolio(self, 
                        deltas: Dict[str, float], 
                        prices: Dict[str, float],
                        exchange_rates: Dict[str, float],
                        current_date: datetime):
        """
        Update the portfolio based on new delta hedging ratios
        
        Args:
            deltas: Dictionary of new delta values for each index
            prices: Dictionary of current prices for each index (in local currency)
            exchange_rates: Dictionary of exchange rates for each index
            current_date: Current date
        """
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value(prices, exchange_rates)
        
        # Calculate total delta exposure in EUR
        total_delta_exposure = 0.0
        for index, delta in deltas.items():
            if index not in prices or index not in exchange_rates:
                continue
                
            # Convert to EUR exposure
            index_exposure = delta * portfolio_value
            total_delta_exposure += abs(index_exposure)
        
        # Calculate scaling factor if total exposure exceeds portfolio value
        scaling_factor = 1.0
        if total_delta_exposure > portfolio_value:
            scaling_factor = portfolio_value / total_delta_exposure
        
        # Calculate new positions
        new_positions = {}
        remaining_capital = portfolio_value
        
        for index, delta in deltas.items():
            if index not in prices or index not in exchange_rates:
                continue
                
            price = prices[index]
            exchange_rate = exchange_rates[index]
            
            # Calculate EUR exposure
            index_exposure = delta * portfolio_value * scaling_factor
            
            # Calculate quantity in local currency units
            if price > 0 and exchange_rate > 0:
                # Convert EUR exposure to local currency, then divide by price
                quantity = index_exposure / (price * exchange_rate)
                new_positions[index] = quantity
                
                # Update remaining capital
                remaining_capital -= index_exposure
            else:
                new_positions[index] = 0.0
        
        # Calculate trading costs (if any)
        trading_costs = 0.0  # For simplicity, assuming zero trading costs
        
        # Update positions and cash
        self.positions = new_positions
        self.cash = remaining_capital - trading_costs
        
        # Record transaction
        self.transaction_history.append({
            'date': current_date,
            'type': 'rebalance',
            'positions': self.positions.copy(),
            'cash': self.cash,
            'portfolio_value': portfolio_value
        })
    
    def pay_dividend(self, 
                    dividend_amount: float, 
                    payment_date: datetime):
        """
        Process a dividend payment
        
        Args:
            dividend_amount: Amount of dividend to pay
            payment_date: Date of dividend payment
        """
        # Subtract dividend from cash
        self.cash -= dividend_amount
        
        # Record transaction
        self.transaction_history.append({
            'date': payment_date,
            'type': 'dividend_payment',
            'amount': dividend_amount,
            'cash': self.cash
        })
    
    def calculate_portfolio_value(self, 
                                prices: Dict[str, float],
                                exchange_rates: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value
        
        Args:
            prices: Dictionary of current prices for each index (in local currency)
            exchange_rates: Dictionary of exchange rates for each index
            
        Returns:
            Current portfolio value in EUR
        """
        # Calculate value of positions
        position_value = 0.0
        
        for index, quantity in self.positions.items():
            if index in prices and index in exchange_rates:
                price = prices[index]
                exchange_rate = exchange_rates[index]
                
                # Convert to EUR
                position_value += quantity * price * exchange_rate
        
        # Add cash position
        total_value = position_value + self.cash
        
        return total_value
    
    def calculate_pnl(self, 
                     prices: Dict[str, float],
                     exchange_rates: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate the profit and loss
        
        Args:
            prices: Dictionary of current prices for each index (in local currency)
            exchange_rates: Dictionary of exchange rates for each index
            
        Returns:
            Tuple containing:
            - Absolute PnL in EUR
            - Percentage PnL
        """
        # Calculate current portfolio value
        current_value = self.calculate_portfolio_value(prices, exchange_rates)
        
        # Calculate absolute and percentage PnL
        absolute_pnl = current_value - self.initial_capital
        percentage_pnl = (absolute_pnl / self.initial_capital) * 100.0
        
        # Record PnL
        self.pnl_history.append({
            'date': datetime.now(),
            'portfolio_value': current_value,
            'absolute_pnl': absolute_pnl,
            'percentage_pnl': percentage_pnl
        })
        
        return absolute_pnl, percentage_pnl
    
    def get_portfolio_summary(self, 
                             prices: Dict[str, float],
                             exchange_rates: Dict[str, float]) -> Dict:
        """
        Get a summary of the current portfolio
        
        Args:
            prices: Dictionary of current prices for each index (in local currency)
            exchange_rates: Dictionary of exchange rates for each index
            
        Returns:
            Dictionary containing portfolio summary information
        """
        # Calculate position values
        position_values = {}
        total_position_value = 0.0
        
        for index, quantity in self.positions.items():
            if index in prices and index in exchange_rates:
                price = prices[index]
                exchange_rate = exchange_rates[index]
                
                # Calculate value in EUR
                value = quantity * price * exchange_rate
                position_values[index] = value
                total_position_value += value
        
        # Calculate PnL
        absolute_pnl, percentage_pnl = self.calculate_pnl(prices, exchange_rates)
        
        # Prepare summary
        summary = {
            'total_value': total_position_value + self.cash,
            'position_values': position_values,
            'cash': self.cash,
            'cash_percentage': (self.cash / (total_position_value + self.cash)) * 100.0 if (total_position_value + self.cash) > 0 else 0.0,
            'absolute_pnl': absolute_pnl,
            'percentage_pnl': percentage_pnl,
            'positions': self.positions.copy()
        }
        
        return summary