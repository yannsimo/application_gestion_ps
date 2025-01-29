from decimal import Decimal


class HoldingCalculator:
    def calculate_total_value(self, quantity, price):
        return quantity * price

    def calculate_optimal_quantity(self, target_value, current_price):
        return int(target_value / current_price)