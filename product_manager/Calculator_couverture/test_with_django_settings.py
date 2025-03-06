# test_with_django_settings.py
"""
Ce script configure les paramètres Django avant d'importer vos modules
"""
import os
import sys
import numpy as np
from datetime import datetime, timedelta

# Ajouter le chemin racine du projet au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

print(f"Chemin racine ajouté: {project_root}")

# Configurer l'environnement Django
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pythonProject.settings')
try:
    django.setup()
    print("Configuration Django réussie")
except Exception as e:
    print(f"Erreur lors de la configuration Django: {e}")

    # Configuration manuelle des paramètres Django minimaux
    from django.conf import settings

    if not settings.configured:
        settings.configure(
            DEBUG=True,
            BASE_DIR=project_root,
            INSTALLED_APPS=[
                'django.contrib.contenttypes',
                'django.contrib.auth',
                'pythonProject',
            ],
        )
        print("Configuration manuelle des paramètres Django")

# Maintenant essayez d'importer avec le chemin complet
try:
    from pythonProject.structured_product.product_manager.Calculator_couverture.testr import SimpleSimulator

    print("Import réussi via le chemin absolu complet")
except ImportError as e:
    print(f"Erreur lors de l'import absolu: {e}")
    sys.exit(1)


def create_mock_data():
    """Crée un mock du fichier Excel si nécessaire"""
    try:
        from django.conf import settings
        import pandas as pd

        # Vérifier si le répertoire data existe
        data_dir = os.path.join(settings.BASE_DIR, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Répertoire data créé: {data_dir}")

        # Vérifier si le fichier existe
        excel_path = os.path.join(data_dir, 'DonneesGPS2025.xlsx')
        if not os.path.exists(excel_path):
            # Créer un fichier Excel factice avec des données minimales
            print(f"Création d'un fichier Excel factice: {excel_path}")

            # Données minimales pour les indices
            indices_data = {
                'Code': ['INDEX1', 'INDEX2', 'INDEX3'],
                'Name': ['Index 1', 'Index 2', 'Index 3'],
                'Price': [100.0, 150.0, 200.0],
                'Volatility': [0.2, 0.25, 0.3],
                'Date': [datetime.now(), datetime.now(), datetime.now()]
            }

            # Créer un DataFrame et l'exporter vers Excel
            df = pd.DataFrame(indices_data)

            # Créer un writer Excel avec plusieurs feuilles
            with pd.ExcelWriter(excel_path) as writer:
                df.to_excel(writer, sheet_name='Indices', index=False)
                df.to_excel(writer, sheet_name='Correlations', index=False)
                df.to_excel(writer, sheet_name='Parameters', index=False)

            print(f"Fichier Excel factice créé avec succès")
            return True
        else:
            print(f"Le fichier Excel existe déjà: {excel_path}")
            return True
    except Exception as e:
        print(f"Erreur lors de la création des données factices: {e}")
        return False


def run_tests_with_mock():
    """Exécute les tests pour SimpleSimulator avec données factices"""
    print("\n" + "=" * 60)
    print("TESTS POUR SIMPLESIMULATOR AVEC MOCKES")
    print("=" * 60)

    # Remplacer les méthodes qui accèdent aux données externes
    from unittest.mock import patch

    # Créer des mocks pour les méthodes qui accèdent aux données
    def mock_get_price(self, code, date):
        return 100.0

    def mock_get_index_exchange_rate(self, code, date):
        return 1.0

    def mock_get_volatility(self, code, date):
        return 0.2

    def mock_get_correlation(self, code1, code2, date):
        return 0.5 if code1 != code2 else 1.0

    # Appliquer les mocks
    from pythonProject.structured_product.product_manager.Data.SingletonMarketData import SingletonMarketData

    # Sauvegarder les méthodes originales
    original_get_price = SingletonMarketData.get_price
    original_get_index_exchange_rate = SingletonMarketData.get_index_exchange_rate

    # Remplacer par les mocks
    SingletonMarketData.get_price = mock_get_price
    SingletonMarketData.get_index_exchange_rate = mock_get_index_exchange_rate

    try:
        # Créer le simulateur
        simulator = SimpleSimulator()
        print("✓ Simulateur créé avec succès")

        # Restaurer les méthodes originales
        SingletonMarketData.get_price = original_get_price
        SingletonMarketData.get_index_exchange_rate = original_get_index_exchange_rate

        # Continuer les tests
        print("\n[TEST] Simulation des chemins...")
        num_paths = 5
        paths = simulator.simulate_paths(num_paths=num_paths, seed=42, daily_steps=False)

        print("✓ Simulation réussie")
        print(f"- Dimensions des chemins: {paths.shape}")

        print("\n[TEST] Calcul du payoff...")
        payoff_results = simulator.calculate_payoff(paths)

        print("✓ Calcul du payoff réussi")
        print(f"- Payoff final moyen: {payoff_results['final_payoff']:.2f}")
        print(f"- Payoff total moyen: {payoff_results['total_payoff']:.2f}")

        print("\n" + "=" * 60)
        print("TESTS RÉUSSIS AVEC DONNÉES MOCKÉES!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ ERREUR LORS DES TESTS: {e}")
        import traceback
        traceback.print_exc()

        # Restaurer les méthodes originales en cas d'erreur
        SingletonMarketData.get_price = original_get_price
        SingletonMarketData.get_index_exchange_rate = original_get_index_exchange_rate
        return False


if __name__ == "__main__":
    # D'abord essayer de créer des données factices
    if create_mock_data():
        try:
            # Essayer de créer et tester le simulateur normalement
            simulator = SimpleSimulator()
            print("Simulateur créé avec succès sans mocks")

            # Exécuter des tests sur le simulateur
            num_paths = 5
            paths = simulator.simulate_paths(num_paths=num_paths, seed=42, daily_steps=False)
            print(f"Simulation réussie, dimensions: {paths.shape}")

            payoff_results = simulator.calculate_payoff(paths)
            print(f"Calcul du payoff réussi: {payoff_results['total_payoff']:.2f}")

            print("\nTOUS LES TESTS ONT RÉUSSI SANS MOCKS!")

        except Exception as e:
            print(f"Erreur avec le simulateur réel: {e}")
            print("Passage aux tests avec données mockées...")
            run_tests_with_mock()
    else:
        print("Impossible de créer des données factices, passage aux tests avec mocks complets...")
        run_tests_with_mock()