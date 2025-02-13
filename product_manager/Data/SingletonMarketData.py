
from .Market_data import MarketData
from ..Product.parameter.date.structured_product_dates import KEY_DATES_AUTO, KEY_DATES_CUSTOM
class SingletonMarketData:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MarketData()
            cls._instance.load_from_excel()
            cls._instance.current_date = KEY_DATES_AUTO.get_Ti(0)
        return cls._instance