import numpy as np
from typing import Dict, List
from datetime import datetime
from model import BlackScholesSimulation
from ...Product.Index import Index
from ...Product.parameter import ProductParameters
from structured_product.product_manager.Product.parameter.Volatility.VolatilityCalculator import VolatilityCalculator
import numpy as np
from numba import jit, prange


def get_singleton_market_data():
    from ...views import SingletonMarketData
    return SingletonMarketData


@jit(nopython=True, parallel=True)
def simulate_price_numba(S0, r, sigma, time_to_maturity, correlated_Z):
    num_indices, num_simulations = correlated_Z.shape
    ST = np.empty((num_indices, num_simulations))
    for i in prange(num_indices):
        ST[i] = S0[i] * np.exp((r[i] - 0.5 * sigma[i]**2) * time_to_maturity + sigma[i] * np.sqrt(time_to_maturity) * correlated_Z[i])
    return ST

@jit(nopython=True, parallel=True)
def analyze_results_numba(simulated_prices):
    num_indices = simulated_prices.shape[0]
    mean_prices = np.empty(num_indices)
    std_prices = np.empty(num_indices)
    for i in prange(num_indices):
        mean_prices[i] = np.mean(simulated_prices[i])
        std_prices[i] = np.std(simulated_prices[i])
    return mean_prices, std_prices

class StructuredProductPricer:
    def __init__(self):
        self.market_data = get_singleton_market_data().get_instance()
        self.product_parameter = ProductParameters(self.market_data, self.market_data.current_date)

        # Exportation des attributs de ProductParameters
        self.underlying_indices = self.product_parameter.underlying_indices  # Liste des indices sous-jacents
        self.key_dates = self.product_parameter.key_dates  # Objet contenant les dates clés du produit
        self.initial_date = self.product_parameter.initial_date  # Date initiale du produit (T0)
        self.final_date = self.product_parameter.final_date  # Date finale du produit (Tc)
        self.observation_dates = self.product_parameter.observation_dates  # Liste des dates d'observation (T1 à T4)
        self.num_simulations = self.product_parameter.num_simulations  # Nombre de simulations pour Monte Carlo
        self.initial_value = self.product_parameter.initial_value  # Valeur initiale du produit
        self.participation_rate = self.product_parameter.participation_rate  # Taux de participation (40%)
        self.cap = self.product_parameter.cap  # Plafond de performance (50%)
        self.floor = self.product_parameter.floor  # Plancher de performance (-15%)
        self.minimum_guarantee = self.product_parameter.minimum_guarantee  # Garantie minimale (20%)
        self.dividend_multiplier = self.product_parameter.dividend_multiplier  # Multiplicateur pour le calcul du dividende

        # Paramètres de marché
        self.volatilities = self.product_parameter.volatilities  # Dictionnaire des volatilités par indice
        self.risk_free_rates = self.product_parameter.risk_free_rates  # Dictionnaire des taux sans risque par indice
        self.cholesky_matrix = self.product_parameter.cholesky_matrix  # Matrice de Cholesky

    def get_current_price(self, index_code):
        return self.market_data.get_price(index_code, self.market_data.current_date)

    def get_risk_free_rate_euro(self, index_code):
        return self.market_data.get_index_interest_rate(index_code,
                                                        self.market_data.current_date) * self.market_data.get_index_exchange_rate(index_code, self.market_data.current_date)



    def get_cholesky(self, index_codes):
        return VolatilityCalculator.calculate_vol_cholesky(index_codes, self.market_data.current_date)

    def price_product(self) -> float:
        """
        Calcule le prix du produit structuré.
        """
        # TODO: Implémenter la logique de pricing
        # Exemple d'utilisation des paramètres :
        # time_to_maturity = self.product_parameter.get_time_to_maturity()
        # current_prices = {index: self.market_data.get_price(index) for index in self.underlying_indices}

        pass

    def calculate_deltas(self) -> Dict[str, float]:
        """
        Calcule les deltas pour chaque indice sous-jacent.
        """
        deltas = {}
        for index_code in self.underlying_indices:
            delta = self._calculate_single_delta(index_code)
            deltas[index_code] = delta
        return deltas

    def _calculate_single_delta(self, index_code: str) -> float:
        """
        Calcule le delta pour un seul indice.
        """
        # TODO: Implémenter le calcul du delta
        # Exemple d'utilisation des paramètres :
        # volatility = self.volatilities[index_code]
        # risk_free_rate = self.risk_free_rates[index_code]
        pass

    from typing import Dict
    import numpy as np

    from typing import Dict
    import numpy as np

    def run_monte_carlo(self) -> Dict[str, Dict[str, float]]:
        """
        Exécute une simulation Monte Carlo pour le pricing du produit en prenant en compte la corrélation.
        ceci est un exemple.
        """

        current_date = self.market_data.current_date
        maturity_date = self.final_date
        date_constation_1 = self.observation_dates[0]  # Première date de constatation

        # Récupération des paramètres de marché pour chaque sous-jacent
        vectors_S0 = np.array([self.market_data.get_price(code, current_date) for code in self.underlying_indices])
        vector_r = np.array(
            [self.market_data.get_index_interest_rate(code, current_date) for code in self.underlying_indices])
        sigma = np.array([self.volatilities[code] for code in self.underlying_indices])

        # Calcul de la matrice de Cholesky pour tenir compte des corrélations entre actifs
        Cholesky = self.get_cholesky(self.underlying_indices)

        # Simulation des facteurs de bruit corrélés
        Z = np.random.standard_normal((len(self.underlying_indices), self.num_simulations))
        correlated_Z = np.dot(Cholesky, Z)

        # Instanciation de la classe de simulation Monte Carlo avec Cholesky


        # Exécution de la simulation avec variables corrélées
        dt = 1 / 252  # Pas de temps (trading journalier)
        simulated_paths = BlackScholesSimulation.simulate_multiple_indices(
            vectors_S0,
            vector_r,
            sigma,
            maturity_date - current_date, #à modifier
            dt,
            self.num_simulations,
            correlated_Z  # Passer les chocs corrélés
        )

        # Traitement des résultats pour obtenir les prix finaux à maturité
        final_prices = {code: simulated_paths[code][:, -1].mean() for code in self.underlying_indices}

        # Retourne un dictionnaire contenant les prix simulés à maturité
        return {"final_prices": final_prices}


# Exemple d'utilisation
if __name__ == "__main__":
    pricer = StructuredProductPricer()
    print(f"Indices sous-jacents : {pricer.underlying_indices}")
    print(f"Volatilités : {pricer.volatilities}")
    print(f"Taux sans risque : {pricer.risk_free_rates}")
    print(f"Matrice de Cholesky :\n{pricer.cholesky_matrix}")