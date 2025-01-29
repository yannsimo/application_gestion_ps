from datetime import timedelta

from django.utils import timezone
from decimal import Decimal
import pandas as pd
import os
from django.conf import settings
from django.core.exceptions import ObjectDoesNotExist
from ..models import MarketData, Index, StructuredProduct
from django.db import transaction

from django.db import IntegrityError


class MarketDataLoader:
   def __init__(self):
       self.file_path = os.path.join(settings.BASE_DIR, 'data', 'DonneesGPS2025.xlsx')
       self._data = None

   def load_prices(self):
       if self._data is None:
           df = pd.read_excel(self.file_path, sheet_name="ClosePrice")
           df.set_index('Date', inplace=True)
           self._data = df.fillna(0)
       return self._data

   def get_price(self, index_code, date):
       df = self.load_prices()
       if index_code in df.columns:
           try:
               # Convert date to pandas timestamp before lookup
               pd_date = pd.Timestamp(date)
               return df.loc[pd_date, index_code]
           except KeyError:
               return None
       return None

   def get_year_prices(self, index_code, current_date):
       df = self.load_prices()
       # Normaliser les dates pour éviter les problèmes d'heures


       # Définir l'intervalle de filtrage
       year_ago = current_date - timedelta(days=365)

       # Vérifier que les dates sont bien dans l'intervalle attendu
       print(f"Filtrage des données entre {year_ago} et {current_date}")

       current_date = pd.to_datetime(current_date)
       year_ago = pd.to_datetime(year_ago)
       # Appliquer le filtre sur l'index de dates
       mask = (df.index >= year_ago) & (df.index <= current_date)
       print(mask)
       filtered_data = df.loc[mask, index_code]

       # Afficher les données filtrées pour déboguer
       print(f"Filtrage complet, résultats : {filtered_data.head()}")

       # Retourner les données sous forme de dictionnaire
       return filtered_data.to_dict()

