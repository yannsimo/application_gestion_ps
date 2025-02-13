import pandas as pd

from .base_strategy import MarketDataStrategy
from datetime import datetime

class PriceStrategy(MarketDataStrategy):
    def load_data(self, file_path: str):
        self._data_df = self.data_loader.load_data(file_path, "ClosePrice")

    def get_data(self, code: str, date: datetime) -> float:
        if self._data_df is None:
            raise ValueError("Data not loaded")
        if code in self._data_df.columns:
            try:
                pd_date = pd.Timestamp(date)
                return self._data_df.loc[pd_date, code]
            except KeyError:
                return None
        return None