import os
from typing import Dict, List
from datetime import datetime, timedelta
from django.conf import settings
from .MarketIndex import MarketIndex
from .data_loader import ExcelDataLoader
from .strategies.price_strategy import PriceStrategy
from .strategies.return_strategy import ReturnStrategy
from .strategies.interest_rate_strategy import InterestRateStrategy
from .strategies.exchange_rate_strategy import ExchangeRateStrategy
from .factories.market_index_factory import MarketIndexFactory
import numpy as np
from numba import jit


class MarketData:
    def __init__(self):
        self.file_path = os.path.join(settings.BASE_DIR, 'data', 'DonneesGPS2025.xlsx')
        self.indices: Dict[str, MarketIndex] = {}
        self.interest_rates: Dict[str, Dict[datetime, float]] = {}
        self.exchange_rates: Dict[str, Dict[datetime, float]] = {}
        self.dates: List[datetime] = []
        self._current_date = None

        self.data_loader = ExcelDataLoader()
        self.price_strategy = PriceStrategy(self.data_loader)
        self.return_strategy = ReturnStrategy(self.data_loader)
        self.interest_rate_strategy = InterestRateStrategy(self.data_loader)
        self.exchange_rate_strategy = ExchangeRateStrategy(self.data_loader)

    @property
    def current_date(self) -> datetime:
        if self._current_date is None and self.dates:
            self._current_date = self.dates[0]
        return self._current_date

    @current_date.setter
    def current_date(self, date: datetime) -> None:
        if date in self.dates:
            self._current_date = date
        else:
            raise ValueError(f"Date {date} non disponible dans les données")

    def next_date(self) -> None:
        if self.current_date:
            current_index = self.dates.index(self.current_date)
            if current_index < len(self.dates) - 1:
                self._current_date = self.dates[current_index + 1]

    def previous_date(self) -> None:
        if self.current_date:
            current_index = self.dates.index(self.current_date)
            if current_index > 0:
                self._current_date = self.dates[current_index - 1]

    def load_from_excel(self) -> None:
        self.price_strategy.load_data(self.file_path)
        self.return_strategy.load_data(self.file_path)
        self.interest_rate_strategy.load_data(self.file_path)
        self.exchange_rate_strategy.load_data(self.file_path)

        info_df = self.data_loader.load_data(self.file_path, 'Infos')
        for _, row in info_df.iterrows():
            index = MarketIndexFactory.create_market_index(row)
            self.indices[index.code] = index

        self.dates = sorted(self.price_strategy.get_dates())
        if self.dates:
            self._current_date = self.dates[0]

    def get_price(self, index_code: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        return self.price_strategy.get_data(index_code, date)

    def get_return(self, index_code: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        return self.return_strategy.get_data(index_code, date)

    def get_interest_rate(self, currency: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        return self.interest_rate_strategy.get_data(currency, date)

    def get_exchange_rate(self, currency_pair: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        return self.exchange_rate_strategy.get_data(currency_pair, date)

    def get_index_interest_rate(self, index_code: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        if not self.indices:
            print("self.indices est vide. Données non chargées ?")
            return None
        index = self.indices.get(index_code)
        if index is None:
            print(f"Index non trouvé pour le code {index_code}")
            return None
        rate_interest_code = index.rate_interest
        return self.get_interest_rate(rate_interest_code, date)

    def get_index_exchange_rate(self, index_code: str, date: datetime = None) -> float:
        if date is None:
            date = self.current_date
        if not self.indices:
            print("self.indices est vide. Données non chargées ?")
            return None
        index = self.indices.get(index_code)
        if index is None:
            print(f"Index non trouvé pour le code {index_code}")
            return None
        foreign_currency_code = index.foreign_currency
        if foreign_currency_code != 'EUR':
            return self.get_exchange_rate(foreign_currency_code, date)
        else:
            return 1

    @staticmethod
    @jit(nopython=True)
    def _calculate_year_prices(prices, dates, current_date, days_in_year):
        end_index = np.searchsorted(dates, current_date)
        start_index = np.searchsorted(dates, current_date - days_in_year)
        return prices[start_index:end_index + 1], dates[start_index:end_index + 1]

    def get_year_prices(self, index_code: str, current_date: datetime = None) -> Dict[datetime, float]:
        if current_date is None:
            current_date = self.current_date
        all_prices = self.price_strategy.get_all_data(index_code)
        prices = np.array(list(all_prices.values()))
        dates = np.array(list(all_prices.keys()))

        year_prices, year_dates = self._calculate_year_prices(prices, dates, current_date, timedelta(days=365))
        return dict(zip(year_dates, year_prices))

    @staticmethod
    @jit(nopython=True)
    def _filter_current_data(data, dates, current_date):
        current_index = np.searchsorted(dates, current_date)
        return data[current_index]

    def get_current_prices(self) -> Dict[str, float]:
        return {code: self._filter_current_data(
            np.array(list(self.price_strategy.get_all_data(code).values())),
            np.array(list(self.price_strategy.get_all_data(code).keys())),
            self.current_date
        ) for code in self.indices.keys()}

    def get_current_returns(self) -> Dict[str, float]:
        return {code: self._filter_current_data(
            np.array(list(self.return_strategy.get_all_data(code).values())),
            np.array(list(self.return_strategy.get_all_data(code).keys())),
            self.current_date
        ) for code in self.indices.keys()}

    def get_current_interest_rates(self) -> Dict[str, float]:
        return {currency: self._filter_current_data(
            np.array(list(self.interest_rate_strategy.get_all_data(currency).values())),
            np.array(list(self.interest_rate_strategy.get_all_data(currency).keys())),
            self.current_date
        ) for currency in self.get_available_currencies()}

    def get_current_exchange_rates(self) -> Dict[str, float]:
        return {pair: self._filter_current_data(
            np.array(list(self.exchange_rate_strategy.get_all_data(pair).values())),
            np.array(list(self.exchange_rate_strategy.get_all_data(pair).keys())),
            self.current_date
        ) for pair in self.get_available_currency_pairs()}

    def get_available_indices(self) -> List[str]:
        return self.price_strategy.get_available_codes()

    def get_date_range(self) -> tuple:
        return min(self.dates), max(self.dates)

    def get_available_currencies(self) -> List[str]:
        return self.interest_rate_strategy.get_available_codes()

    def get_available_currency_pairs(self) -> List[str]:
        return self.exchange_rate_strategy.get_available_codes()