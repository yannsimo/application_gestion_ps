# test_numba_performance.py
"""
Test de performance avec Numba pour la classe SimpleSimulator
"""
import os
import sys
import time
import numpy as np
from datetime import datetime

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
        )
        print("Configuration manuelle des paramètres Django")

# Importer avec le chemin complet
try:
    from pythonProject.structured_product.product_manager.Calculator_couverture.testr import SimpleSimulator

    print("Import réussi via le chemin absolu complet")
except ImportError as e:
    print(f"Erreur lors de l'import absolu: {e}")
    sys.exit(1)


def run_numba_performance_test():
    """Test de performance avec Numba pour SimpleSimulator"""
    print("\n" + "=" * 60)
    print("TEST DE PERFORMANCE AVEC NUMBA")
    print("=" * 60)

    try:
        # Créer l'instance
        print("\nInitialisation du simulateur...")
        simulator = SimpleSimulator()
        print("✓ Simulateur initialisé")

        # Test 1: Premier appel avec compilation JIT
        print("\nTest 1: Premier appel avec 100 chemins (compilation JIT)...")
        start_time = time.time()
        paths_small = simulator.simulate_paths(num_paths=100, seed=42, daily_steps=False)
        compile_time = time.time() - start_time
        print(f"✓ Temps avec compilation JIT: {compile_time:.4f} secondes")
        print(f"✓ Dimensions des chemins: {paths_small.shape}")

        # Test 2: Deuxième appel (post-compilation)
        print("\nTest 2: Deuxième appel avec 100 chemins (post-compilation)...")
        start_time = time.time()
        paths_small = simulator.simulate_paths(num_paths=100, seed=42, daily_steps=False)
        optimized_time_small = time.time() - start_time
        print(f"✓ Temps post-compilation: {optimized_time_small:.4f} secondes")
        print(f"✓ Accélération: {compile_time / optimized_time_small:.2f}x plus rapide")

        # Test 3: Grand nombre de chemins
        print("\nTest 3: Simulation avec 5000 chemins...")
        start_time = time.time()
        paths_large = simulator.simulate_paths(num_paths=5000, seed=42, daily_steps=False)
        large_time = time.time() - start_time
        print(f"✓ Temps pour 5000 chemins: {large_time:.4f} secondes")
        print(f"✓ Dimensions des chemins: {paths_large.shape}")

        # Calculer l'extrapolation
        extrapolated_time = large_time * (10000 / 5000)
        print(f"✓ Temps extrapolé pour 10000 chemins: {extrapolated_time:.4f} secondes")

        # Test 4: Calcul du payoff
        print("\nTest 4: Calcul du payoff...")
        start_time = time.time()
        payoff_results = simulator.calculate_payoff(paths_small)
        payoff_time = time.time() - start_time
        print(f"✓ Temps pour le calcul du payoff: {payoff_time:.4f} secondes")
        print(f"✓ Payoff final moyen: {payoff_results['final_payoff']:.2f}")
        print(f"✓ Payoff total moyen: {payoff_results['total_payoff']:.2f}")

        # Test 5: Simulation quotidienne si possible
        try:
            print("\nTest 5: Simulation avec pas quotidien (100 chemins)...")
            start_time = time.time()
            daily_paths = simulator.simulate_paths(num_paths=100, seed=42, daily_steps=True)
            daily_time = time.time() - start_time
            print(f"✓ Temps pour simulation quotidienne: {daily_time:.4f} secondes")
            print(f"✓ Dimensions des chemins: {daily_paths.shape}")
        except Exception as e:
            print(f"✗ Erreur lors de la simulation quotidienne: {e}")

        print("\n" + "=" * 60)
        print("TEST DE PERFORMANCE TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ ERREUR LORS DES TESTS: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_numba_performance_test()