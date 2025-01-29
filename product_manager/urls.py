# urls.py dans l'application product_manager
from django.urls import path

from . import views
from .views import HoldingView, MarketInfoView

urlpatterns = [
    path('', HoldingView.as_view(), name='holdings'),

   path('holdings/', HoldingView.as_view(), name='holdings'),
# urls.py
   path('update-market-data/', HoldingView.update_market_data, name='update_market_data'),
   path('market-info/', MarketInfoView.as_view(), name='market_info'),
# urls.py
   path('update-current-date/', views.update_current_date, name='update_current_date'),
   path('set-current-date/', views.set_current_date, name='set_current_date'),
   path('market-info/', MarketInfoView.as_view(), name='market_info'),
   path('market-data/<str:index_code>/', MarketInfoView.as_view(), name='market_data'),
   path('api/market-data/<str:index_code>/', views.get_market_data, name='market_data'),
]