from enum import Enum
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from datetime import datetime, timedelta
import numpy as np

from .Product.Portfolio import Portfolio
from .Product.parameter.date.structured_product_dates import KEY_DATES_AUTO, KEY_DATES_CUSTOM
from .Product.Index import Index
from .Product.StructuredProduct import StructuredProduct
from .Data.Market_data import MarketData
from .Data.SingletonMarketData import SingletonMarketData
from .services.calculators.PerformanceCalculator import PerformanceCalculator
from .services.data_loader import MarketDataLoader

# Import new components (these will be created later)
# from .Calculator_couverture.Couverture_Delta.MonteCarloPXX import MonteCarloPXX
# from .services.calculators.Product11PayoffCalculator import Product11PayoffCalculator
# from .services.PortfolioManager import PortfolioManager
# from .services.Product11Simulator import Product11Simulator

# Simulator instance (will be initialized properly once we create the component)
product_simulator = None

class HoldingView(View):
    def __init__(self):
        super().__init__()
        self.market_data = SingletonMarketData.get_instance()
        self.portfolio = Portfolio()
        self.initialize_portfolio()

    def initialize_portfolio(self):
        print("Initialisation du portfolio")
        self.portfolio.initialize_equal_weights(self.market_data, self.market_data.current_date)
        self.portfolio.update_prices(self.market_data, self.market_data.current_date)

    def get_rate(self, index, rate_type):
        index_code = index.value if isinstance(index, Enum) else index
        if rate_type == 'exchange':
            rate = self.market_data.get_index_exchange_rate(index_code, self.market_data.current_date)
        elif rate_type == 'interest':
            rate = self.market_data.get_index_interest_rate(index_code, self.market_data.current_date)
        else:
            return None

        if rate is None:
            print(f"Avertissement : Taux {rate_type} non trouvé pour {index_code}")
        return rate

    def get_detail(self, index, detail_type):
        if detail_type == 'rendement':
            return {
                'name': self.market_data.indices.get(index.value).rate_interest,
                'quantity': 0,
                'price_euro': self.get_rate(index, 'exchange') * self.get_rate(index, 'interest'),
                'price': self.get_rate(index, 'interest'),
                'value': 0
            }
        elif detail_type == 'position':
            return {
                'name': index.value,
                'quantity': self.portfolio.positions[index],
                'price_euro': self.get_rate(index, 'exchange') * self.portfolio.current_prices[index],
                'price': self.portfolio.current_prices[index],
                'value': self.portfolio.get_position_value(index)
            }

    # In your view function (HoldingView.get)
def get(self, request):
    self.market_data.next_date()
    self.portfolio.update_prices(self.market_data, self.market_data.current_date)
    
    # Make sure deltas is defined as a dictionary
    deltas = {}  # Add this if it's missing
    # If you have a simulation, get deltas from there
    # deltas = product_simulator.deltas if product_simulator else {}
    
    context = {
        'Rendement_detail': {index: self.get_detail(index, 'rendement') for index in Index},
        'positions': {index: self.get_detail(index, 'position') for index in Index},
        'total_value': self.portfolio.get_total_value(),
        'pl_percentage': self.portfolio.get_pnl(),
        'liquidative_value': self.portfolio.get_total_value(),
        'initial_capital': self.portfolio,
        'current_date': self.market_data.current_date.strftime('%Y-%m-%d'),
        'deltas': deltas,  # Make sure this is a dictionary
        'delta_quantities': {},  # Add this 
    }
    
    return render(request, 'holdings/list.html', context)

    def post(self, request):
        action = request.path.split('/')[-2]
        if action == 'update_market_data':
            try:
                self.market_data.next_date()
                self.portfolio.update_prices(self.market_data, self.market_data.current_date)
                
                # Update the simulator if it exists
                global product_simulator
                if product_simulator:
                    # This will be implemented once we create the simulator
                    pass
                
                return JsonResponse({
                    'success': True,
                    'date': self.market_data.current_date.strftime('%Y-%m-%d'),
                    'positions': {index.value: self.get_detail(index, 'position') for index in Index},
                    'total_value': self.portfolio.get_total_value()
                })
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)}, status=400)
        return JsonResponse({'error': 'Invalid action'}, status=400)

@require_POST
def update_current_date(request):
    """Met à jour la date courante"""
    market_data = SingletonMarketData.get_instance()

    return JsonResponse({
        'success': True,
        'date': market_data.current_date.strftime('%Y-%m-%d')
    })


@require_POST
def set_current_date(request):
    """Définit une nouvelle date courante"""
    try:
        data = json.loads(request.body)
        new_date = datetime.strptime(data['date'], '%Y-%m-%d')

        market_data = SingletonMarketData.get_instance()
        market_data.current_date = new_date
        
        # Update the simulator if it exists
        global product_simulator
        if product_simulator:
            # This will be implemented once we create the simulator
            pass

        return JsonResponse({
            'success': True,
            'date': market_data.current_date.strftime('%Y-%m-%d')
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


class MarketInfoView(View):
    def get(self, request):
        self.market_data = SingletonMarketData.get_instance()
        calculator = PerformanceCalculator()

        performances = calculator.calculate_performances(self.market_data.current_date)

        return render(request, 'holdings/market_info.html', {
            'indices': performances,
            'current_date': self.market_data.current_date
        })

# Get market data for a specific index
def get_market_data(request, index_code):
    market_data = SingletonMarketData.get_instance()
    current_date = market_data.current_date

    loader = MarketDataLoader()
    prices_data = loader.get_year_prices(index_code, current_date)

    # Prepare the data for the frontend
    prices = {
        str(date): float(price)
        for date, price in prices_data.items()
    }

    return JsonResponse({
        'dates': list(prices.keys()),
        'prices': list(prices.values())
    })

# New view functions for simulation and portfolio management

@require_POST
def run_simulation(request):
    """Run a Monte Carlo simulation"""
    try:
        global product_simulator
        if not product_simulator:
            # This will be implemented once we create the simulator
            return JsonResponse({
                'success': False,
                'error': 'Simulator not initialized'
            }, status=400)
            
        # Run the simulation
        # This will be implemented once we create the simulator
        
        return JsonResponse({
            'success': True,
            'message': 'Simulation run successfully'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_POST
def get_portfolio_summary(request):
    """Get a summary of the current portfolio"""
    try:
        global product_simulator
        if not product_simulator:
            # This will be implemented once we create the simulator
            return JsonResponse({
                'success': False,
                'error': 'Simulator not initialized'
            }, status=400)
            
        # Get portfolio summary
        # This will be implemented once we create the simulator
        
        return JsonResponse({
            'success': True,
            'portfolio_summary': {}  # Will be filled once simulator is created
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_POST
def get_deltas(request):
    """Get the current delta hedging ratios"""
    try:
        global product_simulator
        if not product_simulator:
            # This will be implemented once we create the simulator
            return JsonResponse({
                'success': False,
                'error': 'Simulator not initialized'
            }, status=400)
            
        # Get deltas
        # This will be implemented once we create the simulator
        
        return JsonResponse({
            'success': True,
            'deltas': {}  # Will be filled once simulator is created
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

@require_POST
def get_expected_payoff(request):
    """Get the expected payoff"""
    try:
        global product_simulator
        if not product_simulator:
            # This will be implemented once we create the simulator
            return JsonResponse({
                'success': False,
                'error': 'Simulator not initialized'
            }, status=400)
            
        # Get expected payoff
        # This will be implemented once we create the simulator
        
        return JsonResponse({
            'success': True,
            'expected_payoff': 0.0  # Will be filled once simulator is created
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)

def simulation_results_view(request):
    """View for displaying simulation results"""
    market_data = SingletonMarketData.get_instance()
    
    global product_simulator
    if not product_simulator:
        # Display a message if simulator not initialized
        return render(request, 'holdings/simulation_results.html', {
            'current_date': market_data.current_date,
            'simulator_initialized': False
        })
    
    # Get simulation results
    # This will be implemented once we create the simulator
    
    return render(request, 'holdings/simulation_results.html', {
        'current_date': market_data.current_date,
        'simulator_initialized': True,
        'simulation_results': {}  # Will be filled once simulator is created
    })

def delta_hedging_view(request):
    """View for displaying delta hedging information"""
    market_data = SingletonMarketData.get_instance()
    
    global product_simulator
    if not product_simulator:
        # Display a message if simulator not initialized
        return render(request, 'holdings/delta_hedging.html', {
            'current_date': market_data.current_date,
            'simulator_initialized': False
        })
    
    # Get delta hedging information
    # This will be implemented once we create the simulator
    
    return render(request, 'holdings/delta_hedging.html', {
        'current_date': market_data.current_date,
        'simulator_initialized': True,
        'deltas': {}  # Will be filled once simulator is created
    })