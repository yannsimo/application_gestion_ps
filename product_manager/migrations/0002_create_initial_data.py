from django.db import transaction, migrations
from django.utils import timezone
from ..models import Index, StructuredProduct, FinancialInstrument, Portfolio, Holding, IndexBasket

def create_initial_data(apps, schema_editor):
   try:
       # Utiliser une transaction atomique pour garantir la cohérence des données
       with transaction.atomic():
           # Créer les indices
           indices = {
               'ASX200': ('ASX 200', 'AUD'),
               'DAX': ('DAX', 'EUR'),
               'FTSE100': ('FTSE 100', 'GBP'),
               'NASDAQ100': ('NASDAQ 100', 'USD'),
               'SMI': ('SMI', 'CHF')
           }

           for code, (name, currency) in indices.items():
               Index.objects.create(
                   code=code,
                   name=name,
                   currency=currency
               )

           # Créer produit structuré
           product = StructuredProduct.objects.create(
               reference_nav=1000,
               start_date=timezone.now(),
               end_date=timezone.now() + timezone.timedelta(days=365),
               min_return_cap=-15.00,
               max_return_cap=50.00,
               guaranteed_return=20.00
           )

           # Créer panier d'indices
           basket = IndexBasket.objects.create(
               product=product,
               basket_performance=0,
               annual_performance=0,
               daily_performance=0,
               six_month_performance=0
           )
           basket.active_indices.add(*Index.objects.all())  # Ajoute tous les indices créés

           # Créer portfolio et instruments
           portfolio = Portfolio.objects.create(
               structured_product=product,
               current_value=1000,
               target_value=1000000,
               last_rebalance=timezone.now()
           )

           for code, (name, currency) in indices.items():
               instrument = FinancialInstrument.objects.create(
                   name=name,
                   currency=currency,
                   type='INDEX',
                   current_price=0.028300
               )

               Holding.objects.create(
                   portfolio=portfolio,
                   instrument=instrument,
                   quantity=1000 if code == 'ASX200' else 0,
                   purchase_price=0.028300,
                   current_price=0.028300,
                   total_value=1000 if code == 'ASX200' else 0
               )
   except Exception as e:
       print(f"Erreur lors de la création des données initiales: {str(e)}")
       raise e  # Raiser l'exception pour que la migration échoue si nécessaire

class Migration(migrations.Migration):
   dependencies = [
       ('product_manager', '0001_initial'),
   ]
   operations = [
       migrations.RunPython(create_initial_data),
   ]
