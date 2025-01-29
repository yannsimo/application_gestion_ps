import pandas as pd
from django.views import View
from django.shortcuts import render

from django.http import JsonResponse
from datetime import datetime, timedelta
from .models import Holding, Index, MarketData, StructuredProduct, IndexBasket
from .services.data_loader import MarketDataLoader
from .services.holding_service import HoldingService
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import json
from datetime import datetime, timedelta
from .services.calculators.PerformanceCalculator import PerformanceCalculator

@require_POST
def update_current_date(request):
    product = StructuredProduct.objects.first()
    product.courant_date += timedelta(days=1)
    product.save()
    return JsonResponse({'new_date': product.courant_date.strftime('%Y-%m-%d')})


@require_POST
def set_current_date(request):
    data = json.loads(request.body)
    new_date = datetime.strptime(data['date'], '%Y-%m-%d').date()
    product = StructuredProduct.objects.first()
    product.courant_date = new_date
    product.save()
    return JsonResponse({'new_date': product.courant_date.strftime('%Y-%m-%d')})
class HoldingView(View):
    def get(self, request):
        service = HoldingService()
        service.update_holdings()
        holdings = Holding.objects.all()
        product = StructuredProduct.objects.first()
        return render(request, 'holdings/list.html', {'holdings': holdings,'product': product})

    def update_market_data(request):
        current_date = datetime.strptime(request.GET.get('date'), '%Y-%m-%d')
        next_date = current_date + timedelta(days=1)

        service = HoldingService()
        service.update_holdings_for_date(next_date)

        return JsonResponse({
            'date': next_date.strftime('%Y-%m-%d'),
            'holdings': list(Holding.objects.values())
        })

    # views.py


class MarketInfoView(View):
    def get(self, request):
        calculator = PerformanceCalculator()
        product = StructuredProduct.objects.first()
        performances = calculator.calculate_performances(product.courant_date)

        return render(request, 'holdings/market_info.html', {
            'indices': performances,
            'current_date': product.courant_date
        })


def update_current_date(request):
    if request.method == 'POST':
        product = StructuredProduct.objects.first()
        if 'date' in request.POST:
            new_date = datetime.strptime(request.POST['date'], '%Y-%m-%d').date()
            product.courant_date = new_date
        else:
            product.courant_date += timedelta(days=1)
        product.save()

        return JsonResponse({
            'new_date': product.courant_date.strftime('%Y-%m-%d')
        })

def get_market_data(request, index_code):
    # Obtenir la date courante depuis le produit structuré
    product = StructuredProduct.objects.first()
    current_date = product.courant_date

    # Charger les prix depuis le loader
    loader = MarketDataLoader()

    # Obtenir les prix pour l'année écoulée
    prices_data = loader.get_year_prices(index_code, current_date)
    print(prices_data)
    # Préparer les données pour le frontend
    prices = {
        str(date): float(price)
        for date, price in prices_data.items()
    }

    return JsonResponse({
        'dates': list(prices.keys()),
        'prices': list(prices.values())
    })