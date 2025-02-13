from datetime import datetime
from typing import List


class KeyDates:
    def __init__(self, dates: List[datetime]):
        if len(dates) < 2:
            raise ValueError("Il faut au moins une date initiale et une date finale")

        self.all_dates = sorted(dates)
        self.initial_date = self.all_dates[0]
        self.final_date = self.all_dates[-1]
        self.intermediate_dates = self.all_dates[1:-1]

    @property
    def T0(self) -> datetime:
        return self.initial_date

    @property
    def Tc(self) -> datetime:
        return self.final_date

    def get_Ti(self, i: int) -> datetime:
        if 0 <= i < len(self.all_dates):
            return self.all_dates[i]
        raise ValueError(f"i doit être entre 0 et {len(self.all_dates) - 1}")

    def __str__(self):
        return "\n".join([f"T{i}: {date.strftime('%Y-%m-%d')}" for i, date in enumerate(self.all_dates)])

    def get_all_dates(self) -> List[datetime]:
        return self.all_dates

    def __len__(self):
        return len(self.all_dates)