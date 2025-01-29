from datetime import datetime, timedelta
from decimal import Decimal

class Product11:
    """
    Modélisation du Produit 11 : Gestion des indices, performances et dividendes.
    Permet de calculer la valeur liquidative du produit à une date donnée.
    """

    def __init__(self, reference_nav, start_date, end_date):
        """
        Initialise le produit structuré avec sa valeur de référence et ses dates.
        """
        self.reference_nav = Decimal(reference_nav)  # NAV de référence
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")  # T0
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")  # Tc
        self.indices = {}  # Stocke les indices sous forme {symbol: {"name": name, "prices": {}, "dividends": {}}}
        self.dividends_paid = []  # Historique des dividendes payés
        self.min_return_cap = Decimal("-0.15")  # Perte max -15%
        self.max_return_cap = Decimal("0.50")  # Gain max 50%
        self.guaranteed_return = Decimal("0.20")  # Garantie si seuil 20% atteint
    
    def add_index(self, name, symbol):
        """
        Ajoute un nouvel indice au produit.
        """
        if symbol not in self.indices:
            self.indices[symbol] = {
                "name": name,
                "prices": {},  # {date: close_price}
                "dividends": {}  # {date: amount}
            }

    def add_price(self, symbol, date, close_price):
        """
        Ajoute un prix de clôture pour un indice à une date donnée.
        """
        if symbol in self.indices:
            self.indices[symbol]["prices"][date] = Decimal(close_price)
        else:
            raise ValueError(f"Indice {symbol} non trouvé. Ajoutez-le d'abord.")

    def add_dividend(self, symbol, date, amount):
        """
        Ajoute un dividende versé par un indice à une date donnée.
        """
        if symbol in self.indices:
            self.indices[symbol]["dividends"][date] = Decimal(amount)
        else:
            raise ValueError(f"Indice {symbol} non trouvé. Ajoutez-le d'abord.")

    def get_price_by_date(self, symbol, date):
        """
        Récupère le prix de clôture d'un indice à une date donnée.
        """
        return self.indices.get(symbol, {}).get("prices", {}).get(date, None)

    def get_total_dividends(self, symbol, until_date):
        """
        Calcule la somme des dividendes versés jusqu'à une date donnée.
        """
        return sum(amount for d, amount in self.indices.get(symbol, {}).get("dividends", {}).items() if d <= until_date)

    def calculate_performance(self, date):
        """
        Calcule la performance moyenne du panier d'indices à une date donnée.
        """
        total_performance = Decimal("0.0")
        count = 0

        for symbol in self.indices:
            price = self.get_price_by_date(symbol, date)
            if price is not None:
                initial_price = self.get_price_by_date(symbol, self.start_date)
                if initial_price:
                    total_performance += (price - initial_price) / initial_price
                    count += 1
        
        if count > 0:
            return total_performance / count
        return Decimal("0.0")

    def calculate_net_asset_value(self, date):
        """
        Calcule la valeur liquidative du Produit 11 à une date donnée.
        Prend en compte les performances des indices et les dividendes versés.
        """
        performance = self.calculate_performance(date)

        # Appliquer les limites de performance
        if performance < self.min_return_cap:
            performance = self.min_return_cap
        elif performance > self.max_return_cap:
            performance = self.max_return_cap
        
        # Vérification de la garantie
        if performance >= self.guaranteed_return:
            performance = max(performance, self.guaranteed_return)

        # Calcul de la valeur liquidative
        nav = self.reference_nav * (1 + performance)

        # Ajouter les dividendes versés
        total_dividends = sum(self.get_total_dividends(symbol, date) for symbol in self.indices)
        nav += total_dividends

        return round(nav, 2)