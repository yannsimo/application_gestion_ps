from decimal import Decimal
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
import pandas as pd
from django.db import transaction
from django.http import JsonResponse

from .data_loader import MarketDataLoader
from .calculators.holding_calculator import HoldingCalculator
from ..models import Holding,FinancialInstrument

class HoldingService:
    def __init__(self):
        self.loader = MarketDataLoader()
        self.calculator = HoldingCalculator()

    def update_holdings(self):
        prices = self.loader.load_prices()

        holdings_to_update = []

        logger.info("Début de la mise à jour des holdings")

        with transaction.atomic():
            for holding in Holding.objects.select_for_update():
                if holding.instrument.id in prices:
                    holding.current_price = Decimal(str(prices[holding.instrument.id]))
                    holding.total_value = self.calculator.calculate_total_value(
                        holding.quantity,
                        holding.current_price
                    )
                    holdings_to_update.append(holding)

            if holdings_to_update:
                logger.info(f"Nombre de holdings à mettre à jour : {len(holdings_to_update)}")
                Holding.objects.bulk_update(holdings_to_update, ['current_price', 'total_value'])

        logger.info("Mise à jour des holdings terminée")

    def update_holdings_for_date(self, date):
        """Met à jour les holdings avec les données du jour spécifié"""
        try:
            df = pd.read_excel(self.loader.file_path)

            # Filtrer les données pour la date spécifiée
            day_data = df[df['Date'] == date]

            if day_data.empty:
                raise ValueError(f"Pas de données pour la date {date}")

            # Préparation de la liste des holdings à mettre à jour
            holdings_to_update = []

            for holding in Holding.objects.select_related('instrument').all():
                if holding.instrument.code in day_data.columns:
                    holding.current_price = Decimal(str(day_data[holding.instrument.code].iloc[0]))
                    holding.total_value = self.calculator.calculate_total_value(
                        holding.quantity,
                        holding.current_price
                    )
                    holdings_to_update.append(holding)

            # Mise à jour des holdings en lot
            if holdings_to_update:
                with transaction.atomic():
                    Holding.objects.bulk_update(holdings_to_update, ['current_price', 'total_value'])

        except Exception as e:
            print(f"Erreur lors de la mise à jour des holdings pour la date {date} : {e}")