from datetime import timedelta
from decimal import Decimal

import pandas as pd
from django.core.exceptions import ObjectDoesNotExist, MultipleObjectsReturned

from ..data_loader import MarketDataLoader
from ...models import MarketData, IndexBasket, StructuredProduct, Index


class PerformanceCalculator:
    def __init__(self):
        self.loader = MarketDataLoader()

    def get_nearest_price(self, index_code, target_date, window=5):
        for i in range(window):
            for date_offset in [0, i, -i]:
                check_date = target_date + timedelta(days=date_offset)
                value = self.loader.get_price(index_code, check_date)
                if pd.notna(value):  # Check for valid value
                    return Decimal(str(float(value)))
        return None

    def calculate_performances(self, current_date):
        performances = []
        for index in Index.objects.all():
            today_price = self.get_nearest_price(index.code, current_date)
            yesterday_price = self.get_nearest_price(index.code, current_date - timedelta(days=1))
            six_months_price = self.get_nearest_price(index.code, current_date - timedelta(days=180))
            year_price = self.get_nearest_price(index.code, current_date - timedelta(days=365))

            if all([today_price, yesterday_price, six_months_price, year_price]):
                performances.append({
                    'index': index,
                    'daily': (today_price - yesterday_price) / yesterday_price * 100,
                    'six_month': (today_price - six_months_price) / six_months_price * 100,
                    'annual': (today_price - year_price) / year_price * 100
                })
        return performances