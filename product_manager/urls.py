# urls.py
from django.urls import path
from . import views
from .views import HoldingView, MarketInfoView

urlpatterns = [
    # Vue principale du portfolio
    path('', HoldingView.as_view(), name='holdings'),
    path('holdings/', HoldingView.as_view(), name='holdings'),

    # Endpoint pour la mise à jour des données
    path('update_market_data/', HoldingView.as_view(), name='update_market_data'),

    # Autres URLs
    path('market_info/', MarketInfoView.as_view(), name='market_info'),
    path('update_current_date/', views.update_current_date, name='update_current_date'),
    path('set_current_date/', views.set_current_date, name='set_current_date'),
    path('market_data/<str:index_code>/', MarketInfoView.as_view(), name='market_data'),
    path('api/market_data/<str:index_code>/', views.get_market_data, name='api_market_data'),
]