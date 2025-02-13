from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict
import pandas as pd

class MarketDataStrategy(ABC):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._data_df = None

    @abstractmethod
    def load_data(self, file_path: str):
        pass

    @abstractmethod
    def get_data(self, code: str, date: datetime) -> float:
        pass

    def get_dates(self) -> List[datetime]:
        return self._data_df.index.tolist()

    def get_available_codes(self) -> List[str]:
        return list(self._data_df.columns)

    def get_year_data(self, code: str, current_date: datetime) -> Dict[datetime, float]:
        year_ago = current_date - pd.Timedelta(days=365)
        mask = (self._data_df.index >= year_ago) & (self._data_df.index <= current_date)
        filtered_data = self._data_df.loc[mask, code]
        return filtered_data.to_dict()