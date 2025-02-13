from datetime import datetime
from .date_generator import StructuredProductDateGenerator
from .KeyDates import KeyDates

# Configuration
AUTO_START_YEAR = 2009
AUTO_DURATION_YEARS = 5
AUTO_NUM_INTERMEDIATE_DATES = 4

CUSTOM_START_DATE = datetime(2009, 1, 5)
CUSTOM_END_DATE = datetime(2014, 1, 6)
CUSTOM_INTERMEDIATE_DATES = [
    datetime(2010, 1, 4),
    datetime(2011, 1, 4),
    datetime(2012, 1, 4),
    datetime(2013, 1, 4)
]

# Initialisation des dates
date_generator = StructuredProductDateGenerator()

auto_dates = date_generator.generate_dates(
    start_year=AUTO_START_YEAR,
    duration_years=AUTO_DURATION_YEARS,
    num_intermediate_dates=AUTO_NUM_INTERMEDIATE_DATES
)
KEY_DATES_AUTO = KeyDates(auto_dates)

custom_dates = date_generator.generate_custom_dates(
    start_date=CUSTOM_START_DATE,
    intermediate_dates=CUSTOM_INTERMEDIATE_DATES,
    end_date=CUSTOM_END_DATE
)
KEY_DATES_CUSTOM = KeyDates(custom_dates)