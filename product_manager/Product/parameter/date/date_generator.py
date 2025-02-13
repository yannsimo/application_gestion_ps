from datetime import datetime, timedelta
from typing import List


class StructuredProductDateGenerator:
    @staticmethod
    def generate_dates(start_year: int, duration_years: int, num_intermediate_dates: int) -> List[datetime]:
        start_date = datetime(start_year, 1, 1)
        end_date = start_date + timedelta(days=365 * duration_years)

        dates = [start_date]

        for i in range(1, num_intermediate_dates + 1):
            intermediate_date = start_date + timedelta(days=(365 * duration_years * i) // (num_intermediate_dates + 1))
            dates.append(intermediate_date)

        dates.append(end_date)

        return dates

    @staticmethod
    def generate_custom_dates(start_date: datetime, intermediate_dates: List[datetime], end_date: datetime) -> List[
        datetime]:
        all_dates = [start_date] + intermediate_dates + [end_date]
        return sorted(all_dates)