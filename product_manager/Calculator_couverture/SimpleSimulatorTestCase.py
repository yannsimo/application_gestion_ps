# test_with_root_path.py
"""
Ce script ajoute le chemin racine du projet au PYTHONPATH avant d'importer
"""
import os
import sys
import numpy as np

# Déterminer le chemin racine du projet (4 niveaux au-dessus du répertoire courant)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

print(f"Chemin racine ajouté: {project_root}")
print(f"Chemins Python: {sys.path}")

# Maintenant essayez d'importer avec le chemin complet
try:
    from pythonProject.structured_product.product_manager.Calculator_couverture.testr import SimpleSimulator

    print("Import réussi via le chemin absolu complet")
except ImportError as e:
    print(f"Erreur lors de l'import absolu: {e}")
    sys.exit(1)


def run_tests():
    """Exécute les tests pour SimpleSimulator"""
    print("\n" + "=" * 60)
    print("TESTS POUR SIMPLESIMULATOR")
    print("=" * 60)

    try:
        # 1. Création de l'instance
        print("\n[TEST 1] Création du simulateur...")
        simulator = SimpleSimulator()
        print("✓ Simulateur créé avec succès")
        print(f"- Indices sous-jacents: {simulator.product_parameter.underlying_indices}")
        print(f"- Date de début: {simulator.start_date}")
        print(f"- Date de fin: {simulator.end_date}")

        # 2. Test de simulation
        print("\n[TEST 2] Simulation des chemins...")
        num_paths = 10
        paths = simulator.simulate_paths(num_paths=num_paths, seed=42, daily_steps=False)

        print(f"✓ Simulation réussie")
        print(f"- Dimensions des chemins: {paths.shape}")
        num_indices = len(simulator.product_parameter.underlying_indices)
        num_dates = len(simulator.product_parameter.observation_dates)
        assert paths.shape == (num_indices, num_paths, num_dates), "Dimensions incorrectes"

        # Afficher quelques statistiques
        print(f"- Prix initiaux moyens: {np.mean(paths[:, :, 0], axis=1)}")
        print(f"- Prix finaux moyens: {np.mean(paths[:, :, -1], axis=1)}")

        # 3. Test de calcul du payoff
        print("\n[TEST 3] Calcul du payoff...")
        payoff_results = simulator.calculate_payoff(paths)

        assert 'final_payoff' in payoff_results, "Clé 'final_payoff' manquante"
        assert 'dividends' in payoff_results, "Clé 'dividends' manquante"
        assert 'total_payoff' in payoff_results, "Clé 'total_payoff' manquante"

        print("✓ Calcul du payoff réussi")
        print(f"- Payoff final moyen: {payoff_results['final_payoff']:.2f}")
        print(f"- Dividendes moyens: {', '.join([f'{d:.2f}' for d in payoff_results['dividends']])}")
        print(f"- Payoff total moyen: {payoff_results['total_payoff']:.2f}")

        print("\n" + "=" * 60)
        print("TOUS LES TESTS ONT RÉUSSI !")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_tests()