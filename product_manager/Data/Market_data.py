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

        # Caches pour améliorer les performances
        self._price_cache = {}
        self._return_cache = {}
        self._interest_rate_cache = {}
        self._exchange_rate_cache = {}
        self._last_price_cache = {}
        self._last_exchange_rate_cache = {}
        self._last_interest_rate_cache = {}
        self._year_prices_cache = {}
        self._date_indices_cache = {}

        # Pour les recherches de dates rapides
        self._dates_array = None
        self._date_to_index = {}

        # Mappings pour accès rapide
        self._index_currency_map = {}
        self._index_interest_map = {}

    def _get_date_index(self, date: datetime) -> int:
        """
        Obtient l'index d'une date dans la liste des dates.
        Utilise une recherche binaire pour de meilleures performances.
        """
        try:
            return self.dates.index(date)
        except ValueError:
            return -1
    @property
    def current_date(self) -> datetime:
        if self._current_date is None and self.dates:
            self._current_date = self.dates[0]
        return self._current_date

    @current_date.setter
    def current_date(self, date: datetime) -> None:
        if date in self.dates:
            # Effacer les caches liés à la date courante
            self._current_date = date
            self._clear_current_date_caches()
        else:
            raise ValueError(f"Date {date} non disponible dans les données")

    def _clear_current_date_caches(self):
        """Vide les caches qui dépendent de la date courante"""
        self._price_cache = {}
        self._return_cache = {}
        self._interest_rate_cache = {}
        self._exchange_rate_cache = {}
        self._year_prices_cache = {}

    def next_date(self) -> None:
        if self.current_date:
            current_index = self._get_date_index(self.current_date)
            if current_index < len(self.dates) - 1:
                self._current_date = self.dates[current_index + 1]
                self._clear_current_date_caches()
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

    def get_last_available_price(self, index_code: str, target_date: datetime) -> float:
        """
        Obtient le dernier prix disponible pour un indice à une date donnée.
        Si le prix n'est pas disponible à la date exacte, utilise le dernier prix connu.

        Args:
            index_code: Code de l'indice
            target_date: Date cible

        Returns:
            Le dernier prix disponible, ou None si aucun prix n'est disponible
        """
        # Essayer d'obtenir le prix à la date exacte
        price = self.get_price(index_code, target_date)

        # Si le prix est disponible, le retourner
        if price is not None:
            return price

        # Si le prix n'est pas disponible, chercher le dernier prix connu
        # Trouver la dernière date avant target_date
        previous_dates = [date for date in self.dates if date < target_date]

        if not previous_dates:
            return None  # Aucune date antérieure disponible

        # Obtenir la dernière date avant target_date
        last_available_date = max(previous_dates)

        # Retourner le prix à cette date
        return self.get_price(index_code, last_available_date)

    def get_last_available_exchange_rate(self, index_code: str, target_date: datetime) -> float:
        """
        Obtient le dernier taux de change disponible pour un indice à une date donnée.
        Si le taux n'est pas disponible à la date exacte, utilise le dernier taux connu.

        Args:
            index_code: Code de l'indice
            target_date: Date cible

        Returns:
            Le dernier taux de change disponible, ou None si aucun taux n'est disponible
        """
        # Essayer d'obtenir le taux à la date exacte
        rate = self.get_index_exchange_rate(index_code, target_date)

        # Si le taux est disponible, le retourner
        if rate is not None:
            return rate

        # Si le taux n'est pas disponible, chercher le dernier taux connu
        # Trouver la dernière date avant target_date
        previous_dates = [date for date in self.dates if date < target_date]

        if not previous_dates:
            return None  # Aucune date antérieure disponible

        # Obtenir la dernière date avant target_date
        last_available_date = max(previous_dates)

        # Retourner le taux à cette date
        return self.get_index_exchange_rate(index_code, last_available_date)

    def get_last_available_interest_rate(self, currency: str, target_date: datetime) -> float:
        """
        Obtient le dernier taux d'intérêt disponible pour une devise à une date donnée.
        Si le taux n'est pas disponible à la date exacte, utilise le dernier taux connu.

        Args:
            currency: Code de la devise
            target_date: Date cible

        Returns:
            Le dernier taux d'intérêt disponible, ou None si aucun taux n'est disponible
        """
        # Essayer d'obtenir le taux à la date exacte
        rate = self.get_interest_rate(currency, target_date)

        # Si le taux est disponible, le retourner
        if rate is not None:
            return rate

        # Si le taux n'est pas disponible, chercher le dernier taux connu
        # Trouver la dernière date avant target_date
        previous_dates = [date for date in self.dates if date < target_date]

        if not previous_dates:
            return None  # Aucune date antérieure disponible

        # Obtenir la dernière date avant target_date
        last_available_date = max(previous_dates)

        # Retourner le taux à cette date
        return self.get_interest_rate(currency, last_available_date)

    def get_last_available_index_interest_rate(self, index_code: str, target_date: datetime) -> float:
        """
        Obtient le dernier taux d'intérêt disponible pour l'indice à une date donnée.
        Si le taux n'est pas disponible à la date exacte, utilise le dernier taux connu.

        Args:
            index_code: Code de l'indice
            target_date: Date cible

        Returns:
            Le dernier taux d'intérêt disponible, ou None si aucun taux n'est disponible
        """
        if not self.indices:
            print("self.indices est vide. Données non chargées ?")
            return None

        index = self.indices.get(index_code)
        if index is None:
            print(f"Index non trouvé pour le code {index_code}")
            return None

        rate_interest_code = index.rate_interest
        return self.get_last_available_interest_rate(rate_interest_code, target_date)
