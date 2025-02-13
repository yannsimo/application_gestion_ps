from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    @abstractmethod
    def load_data(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        pass

class ExcelDataLoader(DataLoader):
    def load_data(self, file_path: str, sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if sheet_name != 'Infos':
            df.set_index('Date', inplace=True)
            df = df.interpolate()
        return df