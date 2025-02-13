from dataclasses import dataclass
from typing import Dict
from datetime import datetime

@dataclass
class MarketIndex:
    code: str
    ric: str
    name: str
    country: str
    currency: str
    foreign_currency: str
    rate_interest: str
    prices: Dict[datetime, float] = None
    returns: Dict[datetime, float] = None