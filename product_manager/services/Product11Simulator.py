from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Product11Simulator:
    """
    Master simulator for Product 11 (Performance Monde)
    
    This class coordinates:
    - Market data management
    - Monte Carlo simulation
    - Payoff calculation
    - Delta hedging
    - Portfolio management
    """
    def __init__(self, 
                market_data, 
                volatility_calculator,
                monte_carlo_simulator,
                payoff_calculator,
                portfolio_manager,
                indices: List[str],
                start_date: datetime,
                observation_dates: List[datetime],
                final_date: datetime,
                initial_capital: float = 1000.0):
        """
        Initialize the simulator
        
        Args:
            market_data: MarketData instance
            volatility_calculator: VolatilityCalculator instance
            monte_carlo_simulator: MonteCarloPXX instance
            payoff_calculator: Product11PayoffCalculator instance
            portfolio_manager: PortfolioManager instance
            indices: List of index codes
            start_date: Initial date (T0)
            observation_dates: List of observation dates (T1 to T4)
            final_date: Final date (Tc)
            initial_capital: Initial capital in EUR
        """
        self.market_data = market_data
        self.volatility_calculator = volatility_calculator
        self.monte_carlo_simulator = monte_carlo_simulator
        self.payoff_calculator = payoff_calculator
        self.portfolio_manager = portfolio_manager
        self.indices = indices
        self.start_date = start_date
        self.observation_dates = observation_dates
        self.final_date = final_date
        self.initial_capital = initial_capital
        
        # State variables
        self.current_date = start_date
        self.excluded_indices = []
        self.min_guarantee_active = False
        self.simulation_results = {}
        self.deltas = {index: 0.0 for index in indices}
        self.expected_payoff = 0.0
        self.dividends_paid = {}
        
        # Performance tracking
        self.portfolio_history = []
        self.expected_payoff_history = []
        
        # Initialize portfolio
        self._initialize_portfolio()
    
    def _initialize_portfolio(self):
        """Initialize the portfolio with delta hedging"""
        # Calculate initial deltas
        self._calculate_deltas()
        
        # Get current prices and exchange rates
        prices = {index: self.market_data.get_price(index, self.current_date) for index in self.indices}
        exchange_rates = {index: self.market_data.get_index_exchange_rate(index, self.current_date) for index in self.indices}
        
        # Initialize portfolio
        self.portfolio_manager.initialize_portfolio(self.deltas, prices, exchange_rates)
        
        # Record initial portfolio state
        self._record_portfolio_state()
    
    def _calculate_deltas(self):
        """Calculate delta hedging ratios"""
        # Calculate time to maturity
        time_to_maturity = (self.final_date - self.current_date).days / 365.0
        
        # Run simulation
        if time_to_maturity <= 0:
            self.deltas = {index: 0.0 for index in self.indices}
            return
        
        # Calculate deltas using Monte Carlo simulation
        self.deltas = self.monte_carlo_simulator.calculate_deltas(
            self.indices,
            self.current_date,
            self.observation_dates,
            self.final_date,
            self.payoff_calculator,
            self.excluded_indices,
            shift_size=0.01,
            num_paths=1000,
            dt=1/252
        )
    
    def _calculate_expected_payoff(self):
        """Calculate expected payoff"""
        # Calculate time to maturity
        time_to_maturity = (self.final_date - self.current_date).days / 365.0
        
        # Run simulation
        if time_to_maturity <= 0:
            self.expected_payoff = 0.0
            return
        
        # Simulate paths
        px_paths, x_paths, p_paths = self.monte_carlo_simulator.simulate(
            self.indices,
            self.current_date,
            time_to_maturity,
            num_paths=1000,
            dt=1/252
        )
        
        # Calculate payoff
        payoff, dividends, new_excluded_indices, guarantee_triggered = self.payoff_calculator.calculate_payoff(
            p_paths,
            self.start_date,
            self.observation_dates,
            self.final_date,
            self.current_date,
            self.excluded_indices
        )
        
        # Update state
        future_observation_dates = [d for d in self.observation_dates if d > self.current_date]
        for date in future_observation_dates:
            date_key = date.strftime('%Y-%m-%d')
            if date_key in dividends:
                self.simulation_results[date_key] = {
                    'dividend': dividends[date_key],
                    'excluded_index': new_excluded_indices[future_observation_dates.index(date)] if len(new_excluded_indices) > future_observation_dates.index(date) else None
                }
        
        self.min_guarantee_active = guarantee_triggered
        self.expected_payoff = np.mean(payoff)
    
    def _record_portfolio_state(self):
        """Record current portfolio state"""
        # Get current prices and exchange rates
        prices = {index: self.market_data.get_price(index, self.current_date) for index in self.indices}
        exchange_rates = {index: self.market_data.get_index_exchange_rate(index, self.current_date) for index in self.indices}
        
        # Get portfolio summary
        portfolio_summary = self.portfolio_manager.get_portfolio_summary(prices, exchange_rates)
        
        # Add additional information
        portfolio_summary['date'] = self.current_date
        portfolio_summary['expected_payoff'] = self.expected_payoff
        portfolio_summary['min_guarantee_active'] = self.min_guarantee_active
        portfolio_summary['excluded_indices'] = self.excluded_indices.copy()
        portfolio_summary['deltas'] = self.deltas.copy()
        
        # Record state
        self.portfolio_history.append(portfolio_summary)
    
    def step_to_date(self, target_date: datetime):
        """
        Step the simulation to a specific date
        
        Args:
            target_date: Target date to step to
        """
        # Check if target date is valid
        if target_date < self.current_date:
            raise ValueError("Cannot step backward in time")
        
        # Loop through all dates between current date and target date
        current = self.current_date
        while current < target_date:
            # Find next significant date (next observation date or target date)
            next_observation = min([d for d in self.observation_dates if d > current], default=self.final_date)
            next_date = min(next_observation, target_date)
            
            # Step to next date
            self._step_to_next_date(next_date)
            
            current = next_date
    
    def _step_to_next_date(self, next_date: datetime):
        """
        Step to the next date in simulation
        
        Args:
            next_date: Next date
        """
        # Update current date
        self.current_date = next_date
        
        # Check if it's an observation date
        is_observation_date = next_date in self.observation_dates
        
        # Calculate deltas
        self._calculate_deltas()
        
        # Calculate expected payoff
        self._calculate_expected_payoff()
        
        # Get current prices and exchange rates
        prices = {index: self.market_data.get_price(index, self.current_date) for index in self.indices}
        exchange_rates = {index: self.market_data.get_index_exchange_rate(index, self.current_date) for index in self.indices}
        
        # Update portfolio
        self.portfolio_manager.update_portfolio(self.deltas, prices, exchange_rates, self.current_date)
        
        # Process observation date
        if is_observation_date:
            self._process_observation_date(next_date)
        
        # Record portfolio state
        self._record_portfolio_state()
    
    def _process_observation_date(self, observation_date: datetime):
        """
        Process an observation date (dividend payment, index exclusion)
        
        Args:
            observation_date: Observation date
        """
        # Check for guarantee trigger
        remaining_indices = [idx for idx in self.indices if idx not in self.excluded_indices]
        annual_performances = {}
        
        for idx in remaining_indices:
            # Calculate annual performance
            perf = self.market_data.get_return(idx, observation_date)
            if perf is not None:
                annual_performances[idx] = perf
        
        # Check if guarantee is triggered (avg performance >= 20%)
        if annual_performances:
            avg_performance = sum(annual_performances.values()) / len(annual_performances)
            if avg_performance >= 0.2:
                self.min_guarantee_active = True
        
        # Find best performing index
        if annual_performances:
            best_index = max(annual_performances, key=annual_performances.get)
            best_performance = annual_performances[best_index]
            
            # Calculate dividend
            dividend = max(0, best_performance) * 50 * self.initial_capital
            
            # Pay dividend
            self.portfolio_manager.pay_dividend(dividend, observation_date)
            
            # Exclude index from future dividend calculations
            self.excluded_indices.append(best_index)
            
            # Record dividend payment
            self.dividends_paid[observation_date] = {
                'amount': dividend,
                'excluded_index': best_index
            }
    
    def get_current_state(self) -> Dict:
        """
        Get the current state of the simulation
        
        Returns:
            Dictionary containing current state
        """
        # Get current prices and exchange rates
        prices = {index: self.market_data.get_price(index, self.current_date) for index in self.indices}
        exchange_rates = {index: self.market_data.get_index_exchange_rate(index, self.current_date) for index in self.indices}
        
        # Get portfolio summary
        portfolio_summary = self.portfolio_manager.get_portfolio_summary(prices, exchange_rates)
        
        return {
            'date': self.current_date,
            'portfolio': portfolio_summary,
            'expected_payoff': self.expected_payoff,
            'deltas': self.deltas,
            'min_guarantee_active': self.min_guarantee_active,
            'excluded_indices': self.excluded_indices,
            'dividends_paid': self.dividends_paid
        }
    
    def get_history(self) -> Dict:
        """
        Get the history of the simulation
        
        Returns:
            Dictionary containing history
        """
        return {
            'portfolio_history': self.portfolio_history,
            'dividends_paid': self.dividends_paid
        }
    
    def plot_portfolio_value(self):
        """Plot portfolio value over time"""
        if not self.portfolio_history:
            print("No portfolio history to plot")
            return
        
        dates = [entry['date'] for entry in self.portfolio_history]
        values = [entry['total_value'] for entry in self.portfolio_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, marker='o')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (EUR)')
        plt.title('Portfolio Value Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_deltas(self):
        """Plot delta values over time"""
        if not self.portfolio_history:
            print("No portfolio history to plot")
            return
        
        dates = [entry['date'] for entry in self.portfolio_history]
        
        plt.figure(figsize=(12, 8))
        
        for index in self.indices:
            deltas = [entry['deltas'].get(index, 0.0) for entry in self.portfolio_history]
            plt.plot(dates, deltas, marker='o', label=index)
        
        plt.xlabel('Date')
        plt.ylabel('Delta')
        plt.title('Delta Values Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()