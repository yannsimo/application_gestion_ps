from ..MarketIndex import MarketIndex
import pandas as pd

class MarketIndexFactory:
    @staticmethod
    def create_market_index(row: pd.Series) -> MarketIndex:
        currency = row['Monnaie']
        currency_foreign = f"X{currency}" if currency != 'EUR' else 'EUR'
        return MarketIndex(
            code=row['Code'],
            ric=row['RIC'],
            name=row['Nom'],
            country=row['Pays'],
            currency=currency,
            foreign_currency=currency_foreign,
            rate_interest=f"R{currency}",
            prices={},
            returns={}
        )