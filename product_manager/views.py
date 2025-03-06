from enum import Enum
import numpy as np
from django.views import View
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from datetime import datetime, timedelta

from .Calculator_couverture.testr import SimpleSimulator
from .Product.PayoffCalculator import PayoffCalculator

from .Product.parameter.ProductParameters import ProductParameters
from .Product.Portfolio import Portfolio
from .Product.parameter.date.structured_product_dates import KEY_DATES_AUTO, KEY_DATES_CUSTOM
from .Product.Index import Index
from .Product.StructuredProduct import StructuredProduct
from .Data.Market_data import MarketData
from .services.calculators.PerformanceCalculator import PerformanceCalculator
from .services.data_loader import MarketDataLoader
from  .Calculator_couverture.Couverture_Delta.model import BlackScholesSimulation
from  .Data.SingletonMarketData import SingletonMarketData


class HoldingView(View):
    _portfolio_instance = None

    def __init__(self):
        super().__init__()
        self.market_data = SingletonMarketData.get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)
        # Utiliser une instance partagée du Portfolio pour toutes les vues
        if HoldingView._portfolio_instance is None:
            HoldingView._portfolio_instance = Portfolio()
            self._initialize_portfolio()

        self.portfolio = HoldingView._portfolio_instance

        # Stockage des taux pour éviter les appels multiples
        self._exchange_rates = {}
        self._interest_rates = {}
        self.product = StructuredProduct(
            initial_date=self.product_parameter.initial_date,
            final_date=self.product_parameter.final_date,
            observation_dates=self.product_parameter.observation_dates,
            initial_value=1000.0
        )
    def _initialize_portfolio(self):
        """Initialiser le portfolio une seule fois"""
        print("Initialisation du portfolio")
        self.portfolio.initialize_equal_weights(self.market_data, self.market_data.current_date)
        self.portfolio.update_prices(self.market_data, self.market_data.current_date)

    def get_rate(self, index, rate_type):
        """Récupérer un taux avec mise en cache locale"""
        index_code = index.value if isinstance(index, Enum) else index

        # Vérifier si le taux est déjà en mémoire
        if rate_type == 'exchange':
            if index_code in self._exchange_rates:
                return self._exchange_rates[index_code]

            rate = self.market_data.get_index_exchange_rate(index_code, self.market_data.current_date)
            self._exchange_rates[index_code] = rate

        elif rate_type == 'interest':
            if index_code in self._interest_rates:
                return self._interest_rates[index_code]

            rate = self.market_data.get_index_interest_rate(index_code, self.market_data.current_date)
            self._interest_rates[index_code] = rate

        else:
            return None

        if rate is None:
            print(f"Avertissement : Taux {rate_type} non trouvé pour {index_code}")

        return rate

    def get_details_batch(self):
        """Récupérer tous les détails en une seule opération"""
        # Précalculer tous les taux en une seule passe
        self._exchange_rates = {}
        self._interest_rates = {}

        for idx in Index:
            self._exchange_rates[idx.value] = self.market_data.get_index_exchange_rate(
                idx.value, self.market_data.current_date)
            self._interest_rates[idx.value] = self.market_data.get_index_interest_rate(
                idx.value, self.market_data.current_date)

        # Créer les dictionnaires de résultats
        rendement_details = {}
        position_details = {}

        for idx in Index:
            # Détails de rendement
            rendement_details[idx] = {
                'name': self.market_data.indices.get(idx.value).rate_interest,
                'quantity': 0,
                'price_euro': (self._exchange_rates[idx.value] or 0) * (self._interest_rates[idx.value] or 0),
                'price': self._interest_rates[idx.value] or 0,
                'value': 0
            }

            # Détails de position
            position_details[idx] = {
                'name': idx.value,
                'quantity': self.portfolio.positions[idx],
                'price_euro': (self._exchange_rates[idx.value] or 0) * self.portfolio.current_prices[idx],
                'price': self.portfolio.current_prices[idx],
                'value': self.portfolio.get_position_value(idx)
            }

        return rendement_details, position_details

    def get(self, request):
        # Mettre à jour la date et les prix
        self.market_data.next_date()
        self.portfolio.update_prices(self.market_data, self.market_data.current_date)

        simulator = SimpleSimulator()

        # Exécuter la simulation avec 1000 chemins
        paths = simulator.simulate_paths(num_paths=50000, seed=42, daily_steps=False)

        # Calculer le payoff
        results = simulator.calculate_payoff(paths)

        # Afficher les résultats
        print("\n=== RÉSULTATS DE LA SIMULATION ===")
        print(f"Payoff final moyen: {results['final_payoff']:.2f} €")

        # Affichage des dividendes moyens
        dividends = results['dividends']
        total_dividends = np.sum(dividends)
        print(f"Total des dividendes moyens: {total_dividends:.2f} €")
        print(f"Payoff total moyen: {results['total_payoff']:.2f} €")

        # Afficher chaque dividende
        for i, div in enumerate(dividends):
            obs_date = simulator.product_parameter.observation_dates[i + 1]
            print(f"Dividende à {obs_date.strftime('%d/%m/%Y')}: {div:.2f} €")
        # Obtenir tous les détails en une seule opération
        rendement_details, position_details = self.get_details_batch()

        # Calculer la valeur totale une seule fois
        total_value = self.portfolio.get_total_value()

        context = {
            'Rendement_detail': rendement_details,
            'positions': position_details,
            'total_value': total_value,
            'pl_percentage': self.portfolio.get_pnl(),
            'liquidative_value': total_value,  # Utiliser la valeur déjà calculée
            'initial_capital': self.portfolio,
            'current_date': self.market_data.current_date.strftime('%Y-%m-%d')
        }

        return render(request, 'holdings/list.html', context)

    def post(self, request):
        action = request.path.split('/')[-2]
        if action == 'update_market_data':
            try:
                self.market_data.next_date()
                self.portfolio.update_prices(self.market_data, self.market_data.current_date)

                # Récupérer les détails avec la méthode optimisée
                _, position_details = self.get_details_batch()

                # Convertir les clés d'énumération en chaînes pour JSON
                positions_json = {idx.value: details for idx, details in position_details.items()}

                return JsonResponse({
                    'success': True,
                    'date': self.market_data.current_date.strftime('%Y-%m-%d'),
                    'positions': positions_json,
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

        # Assurez-vous de réinitialiser les caches dans HoldingView si nécessaire
        # Si vous avez un cache global, vous devriez le vider ici

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
    _performances_cache = {}  # Cache partagé par toutes les instances

    def get(self, request):
        self.market_data = SingletonMarketData.get_instance()
        current_date = self.market_data.current_date

        # Vérifier si les performances sont déjà en cache
        cache_key = current_date.strftime('%Y-%m-%d')
        if cache_key in MarketInfoView._performances_cache:
            performances = MarketInfoView._performances_cache[cache_key]
        else:
            # Calculer et mettre en cache
            calculator = PerformanceCalculator()
            performances = calculator.calculate_performances(current_date)
            MarketInfoView._performances_cache[cache_key] = performances

        return render(request, 'holdings/market_info.html', {
            'indices': performances,
            'current_date': current_date
        })


# Optimisé pour les données de marché sur une année
def get_market_data(request, index_code):
    market_data = SingletonMarketData.get_instance()
    current_date = market_data.current_date

    # Note: À l'idéal, on utiliserait directement market_data.get_year_prices()
    # au lieu de créer un nouveau loader

    # Utiliser directement les fonctions du market_data
    prices_data = market_data.get_year_prices(index_code, current_date)

    # Convertir les dates en chaînes pour JSON en une seule passe
    # et en gardant le tri chronologique
    dates = []
    prices = []

    # Trier les dates pour garantir l'ordre chronologique
    for date in sorted(prices_data.keys()):
        dates.append(date.strftime('%Y-%m-%d'))
        prices.append(float(prices_data[date]))

    return JsonResponse({
        'dates': dates,
        'prices': prices
    })