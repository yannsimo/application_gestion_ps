from ...Product.parameter.date.structured_product_dates import KEY_DATES_AUTO
from ...Product.parameter.VolatilityCalculator import VolatilityCalculator
from ...Product.Index import Index
import numpy as np
from scipy.stats import norm
from numba import jit, prange

index_codes = [index.value for index in Index]

def get_singleton_market_data():
    from ...views import SingletonMarketData
    return SingletonMarketData

@jit(nopython=True)
def simulate_price_numba(S0, r, sigma, time_to_maturity, correlated_Z):
    num_indices, num_simulations = correlated_Z.shape
    ST = np.empty((num_indices, num_simulations))
    for i in range(num_indices):
        ST[i] = S0[i] * np.exp((r[i] - 0.5 * sigma[i]**2) * time_to_maturity
                               + sigma[i] * np.sqrt(time_to_maturity) * correlated_Z[i])
    return ST
@jit(nopython=True)
def analyze_results_numba(simulated_prices):
    mean_prices = np.empty(simulated_prices.shape[0])
    std_prices = np.empty(simulated_prices.shape[0])
    for i in range(simulated_prices.shape[0]):
        mean_prices[i] = np.mean(simulated_prices[i])
        std_prices[i] = np.std(simulated_prices[i])
    return mean_prices, std_prices
class BlackScholesSimulation:
    def __init__(self):
        self.market_data = get_singleton_market_data().get_instance()
        self.volatility_calculator = VolatilityCalculator(self.market_data)
        self.key_dates = KEY_DATES_AUTO

    def get_current_price(self, index_code):
        return self.market_data.get_price(index_code, self.market_data.current_date)

    def get_risk_free_rate_euro(self, index_code):
        return self.market_data.get_index_interest_rate(index_code, self.market_data.current_date) * self.market_data.get_index_exchange_rate(index_code, self.market_data.current_date)

    def get_volatility(self, index_code):
        return self.volatility_calculator.calculate_volatility(index_code, self.market_data.current_date)

    def get_cholesky(self, index_codes):
        return self.volatility_calculator.calculate_vol_cholesky(index_codes, self.market_data.current_date)

    def simulate_price(self, index_codes, time_to_maturity, num_simulations=10000):
        S0 = np.array([self.get_current_price(code) for code in index_codes])
        r = np.array([self.get_risk_free_rate_euro(code) for code in index_codes])
        sigma = np.array([self.get_volatility(code) for code in index_codes])
        Cholesky = self.get_cholesky(index_codes)

        Z = np.random.standard_normal((len(index_codes), num_simulations))
        correlated_Z = np.dot(Cholesky, Z)

        return simulate_price_numba(S0, r, sigma, time_to_maturity, correlated_Z)

    def run_simulation(self):

        """
        Cette fonction est juste un exemple
        :return:
        """
        maturity_date = self.key_dates.Tc #ici c'est juste un exemple
        date_constatation_1 = self.key_dates.get(1)

        time_to_maturity = (maturity_date - self.market_data.current_date).days / 365.0

        simulated_prices = self.simulate_price(index_codes, time_to_maturity)
        mean_prices, std_prices = analyze_results_numba(simulated_prices)

        results = {index: {'mean': mean, 'std': std}
                   for index, mean, std in zip(index_codes, mean_prices, std_prices)}

        return results

if __name__ == "__main__":
    simulator = BlackScholesSimulation()
    results = simulator.run_simulation()
    print(f"Résultats de la simulation: {results}")