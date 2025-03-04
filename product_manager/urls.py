# urls.py
from django.urls import path
from . import views
from .views import HoldingView, MarketInfoView

urlpatterns = [
    # Main views
    path('', HoldingView.as_view(), name='holdings'),
    path('holdings/', HoldingView.as_view(), name='holdings'),
    path('market_info/', MarketInfoView.as_view(), name='market_info'),
    
    # Data update endpoints
    path('update_market_data/', HoldingView.as_view(), name='update_market_data'),
    path('update_current_date/', views.update_current_date, name='update_current_date'),
    path('set_current_date/', views.set_current_date, name='set_current_date'),
    
    # API endpoints for market data
    path('api/market_data/<str:index_code>/', views.get_market_data, name='api_market_data'),
    
    # New API endpoints for simulation and portfolio management
    path('api/run_simulation/', views.run_simulation, name='run_simulation'),
    path('api/get_portfolio_summary/', views.get_portfolio_summary, name='get_portfolio_summary'),
    path('api/get_deltas/', views.get_deltas, name='get_deltas'),
    path('api/get_expected_payoff/', views.get_expected_payoff, name='get_expected_payoff'),
    
    # New views for simulation results
    path('simulation_results/', views.simulation_results_view, name='simulation_results'),
    path('delta_hedging/', views.delta_hedging_view, name='delta_hedging'),
]