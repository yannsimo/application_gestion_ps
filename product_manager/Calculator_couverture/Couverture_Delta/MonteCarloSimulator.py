from datetime import datetime
import numpy as np
from typing import Dict, List, Tuple
from ...Product.Index import Index
from ...Product.StructuredProduct import StructuredProduct
from ...Product.PayoffCalculator import PayoffCalculator


class MonteCarloSimulator:
    def __init__(self, structured_product: StructuredProduct, num_simulations: int = 10000):
        """
        Initialise le simulateur Monte Carlo pour un produit structuré.

        Args:
            structured_product: Le produit structuré à simuler
            num_simulations: Nombre de simulations à exécuter
        """
        self.product = structured_product
        self.market_data = structured_product.market_data
        self.product_parameter = structured_product.product_parameter
        self.num_simulations = num_simulations
        # Récupérer les indices
        self.indices = [idx.value for idx in Index]

    def run_monte_carlo(self) -> float:
        """
        Exécute une simulation Monte Carlo pour calculer le prix théorique du produit.

        Returns:
            Le prix théorique du produit à la date initiale
        """
        current_date = self.market_data.current_date

        # Créer les trajectoires de prix
        simulated_paths = self._generate_price_paths()

        # Calculer les payoffs pour chaque simulation
        payoffs = np.zeros(self.num_simulations)

        for sim in range(self.num_simulations):
            payoffs[sim] = self._calculate_simulation_payoff(simulated_paths, sim)

        # Retourner la moyenne des payoffs comme prix théorique
        return np.mean(payoffs)

    def _generate_price_paths(self) -> Dict:
        """
        Génère les trajectoires de prix pour tous les indices.

        Returns:
            Dictionnaire contenant les trajectoires simulées pour chaque indice
        """
        current_date = self.market_data.current_date
        indices = self.indices

        # Paramètres de marché
        initial_prices = {}
        rates = {}
        volatilities = {}

        for idx in indices:
            initial_prices[idx] = self.market_data.get_price(idx, current_date) * \
                                  self.market_data.get_index_exchange_rate(idx, current_date)
            rates[idx] = self.market_data.get_index_interest_rate(idx, current_date)
            volatilities[idx] = self.product_parameter.volatilities[idx]

        # Matrice de corrélation/Cholesky
        cholesky_matrix = self.product_parameter.cholesky_matrix

        # Dates des observations
        observation_dates = [self.product.initial_date] + self.product.observation_dates + [self.product.final_date]
        observation_dates = sorted(list(set(observation_dates)))  # Éliminer les doublons et trier

        # Créer un dictionnaire pour stocker les prix simulés à chaque date d'observation
        simulated_prices = {idx: {date: np.zeros(self.num_simulations) for date in observation_dates}
                            for idx in indices}

        # Simuler les prix à chaque date d'observation
        for i, idx in enumerate(indices):
            # Prix initial
            simulated_prices[idx][observation_dates[0]] = np.ones(self.num_simulations) * initial_prices[idx]

            # Générer des nombres aléatoires corrélés
            Z = np.random.standard_normal((len(indices), self.num_simulations))
            correlated_Z = np.dot(cholesky_matrix, Z)

            # Simuler les prix aux dates d'observation
            for j in range(1, len(observation_dates)):
                prev_date = observation_dates[j - 1]
                curr_date = observation_dates[j]
                dt = (curr_date - prev_date).days / 365.0  # Temps en années

                drift = (rates[idx] - 0.5 * volatilities[idx] ** 2) * dt
                diffusion = volatilities[idx] * np.sqrt(dt) * correlated_Z[i]

                prev_prices = simulated_prices[idx][prev_date]
                simulated_prices[idx][curr_date] = prev_prices * np.exp(drift + diffusion)

        return simulated_prices

    def _calculate_simulation_payoff(self, simulated_prices: Dict, sim_index: int) -> float:
        """
        Calcule le payoff pour une simulation donnée.

        Args:
            simulated_prices: Prix simulés pour chaque indice à chaque date
            sim_index: Indice de la simulation

        Returns:
            Payoff total actualisé pour cette simulation
        """
        # Dates clés
        initial_date = self.product.initial_date
        final_date = self.product.final_date
        observation_dates = self.product.observation_dates

        # Taux d'actualisation
        payoff_calculator = PayoffCalculator(self.product)
        r_EUR = payoff_calculator._get_risk_free_rate(initial_date)

        # Ensemble d'indices exclus (vide au départ)
        excluded_indices = set()
        guarantee_activated = False

        # Calculer les dividendes actualisés
        total_dividends = 0

        for i, obs_date in enumerate(observation_dates):
            # Date précédente
            prev_date = initial_date if i == 0 else observation_dates[i - 1]

            # Calculer les rendements pour chaque indice
            returns = self._get_simulated_returns(simulated_prices, prev_date, obs_date, sim_index, excluded_indices)

            if returns:
                # Trouver l'indice avec le meilleur rendement
                best_index = max(returns, key=returns.get)
                best_return = returns[best_index]

                # Exclure cet indice pour les prochaines dates
                excluded_indices.add(best_index)

                # Calculer le dividende
                dividend = self.product.dividend_multiplier * best_return

                # Actualiser le dividende
                discount_factor = np.exp(-r_EUR * (obs_date - initial_date).days / 365.0)
                total_dividends += dividend * discount_factor

                # Vérifier si la garantie est activée
                basket_perf = sum(returns.values()) / len(returns)
                if basket_perf >= 0.20:
                    guarantee_activated = True

        # Calculer la performance finale du panier
        final_returns = self._get_simulated_returns(simulated_prices, initial_date, final_date, sim_index)

        if final_returns:
            basket_perf = sum(final_returns.values()) / len(final_returns)

            # Appliquer les règles du produit
            if basket_perf < 0:
                basket_perf = max(basket_perf, -0.15)  # Protection à -15%
            else:
                basket_perf = min(basket_perf, 0.50)  # Plafond à +50%

            # Appliquer la garantie si activée
            if guarantee_activated:
                basket_perf = max(basket_perf, 0.20)

            # Calculer la valeur finale
            final_value = self.product.initial_value * (1 + 0.40 * basket_perf)

            # Actualiser la valeur finale
            discount_factor = np.exp(-r_EUR * (final_date - initial_date).days / 365.0)
            discounted_final_value = final_value * discount_factor

            # Payoff total = dividendes actualisés + valeur finale actualisée
            return total_dividends + discounted_final_value

        return 0.0

    def _get_simulated_returns(self, simulated_prices: Dict, start_date: datetime,
                               end_date: datetime, sim_index: int, excluded_indices: set = None) -> Dict:
        """
        Calcule les rendements simulés entre deux dates pour chaque indice.

        Args:
            simulated_prices: Prix simulés
            start_date: Date de début
            end_date: Date de fin
            sim_index: Indice de la simulation
            excluded_indices: Indices à exclure (optionnel)

        Returns:
            Dictionnaire des rendements par indice
        """
        if excluded_indices is None:
            excluded_indices = set()

        returns = {}

        for idx in self.indices:
            if idx in excluded_indices:
                continue

            start_price = simulated_prices[idx][start_date][sim_index]
            end_price = simulated_prices[idx][end_date][sim_index]

            if start_price > 0:
                returns[idx] = (end_price / start_price) - 1

        return returns