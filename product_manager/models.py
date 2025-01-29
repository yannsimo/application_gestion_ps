from django.db import models
import uuid
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
from decimal import Decimal
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import uuid
from datetime import date
class StructuredProduct(models.Model):
    """
    Représente un produit structuré basé sur la performance des indices mondiaux.
    Ce produit offre une protection à la baisse de -15% et un plafond à la hausse de 50%.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique du produit"
    )
    reference_nav = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Valeur liquidative de référence utilisée pour les calculs"
    )
    start_date = models.DateField(
        help_text="Date de lancement du produit (T0)",
        default=date(2009, 1, 5)  # Utilisation de datetime.date
    )
    courant_date = models.DateField(
        help_text="Date courante pour le produit",
        default=date(2009, 1, 5)  # Utilisation de datetime.date
    )
    end_date = models.DateField(
        help_text="Date d'échéance du produit (Tc)",
        default=date(2014, 1, 6)  # Utilisation de datetime.date
    )
    min_return_cap = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=-0.15,
        validators=[MinValueValidator(-1.0), MaxValueValidator(0.0)],
        help_text="Limite de perte maximale (-15%)"
    )
    max_return_cap = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0.50,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Plafond de gain maximal (50%)"
    )
    guaranteed_return = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0.20,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Performance minimale garantie si seuil de 20% atteint"
    )

    def __str__(self):
        return f"Produit structuré #{self.id}"

class Index(models.Model):
    """
    Représente un indice boursier faisant partie du panier d'indices.
    Les 5 indices sont : ASX200, DAX, FTSE100, NASDAQ100, SMI.
    """
    code = models.CharField(
        max_length=10,
        primary_key=True,
        help_text="Code unique de l'indice (ex: ASX200)"
    )
    ric = models.CharField(
        max_length=10,
        help_text="Code RIC Reuters de l'indice (ex: .AXJO)"
    )
    name = models.CharField(
        max_length=100,
        help_text="Nom complet de l'indice"
    )
    country = models.CharField(
        max_length=50,
        help_text="Pays de l'indice"
    )
    currency = models.CharField(
        max_length=3,
        help_text="Devise de cotation de l'indice"
    )

    excluded_from_dividends = models.BooleanField(
        default=False,
        help_text="True si l'indice est exclu du calcul des dividendes après avoir eu la meilleure performance"
    )


class IndexBasket(models.Model):
    """
    Représente le panier des 5 indices et leurs performances.
    La performance est calculée comme la moyenne des performances individuelles.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique du panier"
    )
    product = models.ForeignKey(
        StructuredProduct,
        on_delete=models.CASCADE,
        help_text="Produit structuré associé à ce panier"
    )
    basket_performance = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="Performance globale du panier (moyenne des performances)"
    )
    annual_performance = models.DecimalField(
        max_digits=10,
        decimal_places=4, default=0,
        help_text="Performance annuelle du panier entre Ti-1 et Ti"
    )
    daily_performance = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        default=0,
        help_text="Performance annuelle journalière  du panier "
    )
    six_month_performance = models.DecimalField(
        max_digits=10,
        decimal_places=4, default=0,
        help_text="Performance sur six mois   du panier "
    )
    active_indices = models.ManyToManyField(
        Index,
        help_text="Liste des indices actifs dans le panier"
    )

class MarketData(models.Model):
    """
    Stocke les données de marché historiques pour chaque indice.
    Inclut les prix de clôture et les rendements.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique de la donnée de marché"
    )
    index = models.ForeignKey(
        Index,
        on_delete=models.CASCADE,
        help_text="Indice auquel se rapporte la donnée"
    )
    timestamp = models.DateTimeField(
        help_text="Date et heure de la donnée"
    )
    value = models.DecimalField(
        max_digits=15,
        decimal_places=4,
        help_text="Valeur de la donnée (prix ou rendement)"
    )

    data_type = models.CharField(
        max_length=20,
        help_text="Type de donnée (ClosePrice, CloseRet, etc.)"
    )

    class Meta:
        indexes = [
            models.Index(fields=['timestamp', 'data_type']),
        ]

class PerformanceCheck(models.Model):
    """
    Représente les constatations de performance aux dates Ti.
    Vérifie si le seuil de performance de 20% est atteint.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique de la constatation"
    )
    check_date = models.DateField(
        help_text="Date de constatation Ti"
    )
    basket = models.ForeignKey(
        IndexBasket,
        on_delete=models.CASCADE,
        help_text="Panier d'indices évalué"
    )
    basket_performance = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="Performance du panier à la date Ti"
    )
    max_annual_return = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        help_text="Meilleure performance annuelle parmi les indices"
    )
    threshold_reached = models.BooleanField(
        default=False,
        help_text="True si la performance >= 20%, déclenchant la garantie"
    )

class DividendPayment(models.Model):
    """
    Gère les paiements de dividendes aux dates Ti.
    Le montant est 50 fois la meilleure performance annuelle.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique du paiement"
    )
    performance_check = models.ForeignKey(
        PerformanceCheck,
        on_delete=models.CASCADE,
        help_text="Constatation ayant déclenché le dividende"
    )
    payment_date = models.DateField(
        help_text="Date de paiement du dividende"
    )
    amount = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Montant du dividende (50 × max performance)"
    )
    excluded_index = models.ForeignKey(
        Index,
        on_delete=models.SET_NULL,
        null=True,
        help_text="Indice exclu après ce paiement (meilleure performance)"
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'En attente'),
            ('PAID', 'Payé'),
            ('CANCELLED', 'Annulé')
        ],
        default='PENDING',
        help_text="Statut du paiement du dividende"
    )

class Portfolio(models.Model):
    """
    Représente le portefeuille de couverture du produit structuré.
    Permet de répliquer les payoffs promis aux clients.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique du portefeuille"
    )
    structured_product = models.OneToOneField(
        StructuredProduct,
        on_delete=models.CASCADE,
        help_text="Produit structuré couvert par ce portefeuille"
    )
    current_value = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Valeur actuelle du portefeuille"
    )
    target_value = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Valeur cible du portefeuille"
    )
    last_rebalance = models.DateTimeField(
        help_text="Date du dernier rebalancement du portefeuille"
    )

class Holding(models.Model):
    """
    Représente une position sur un instrument financier dans le portefeuille.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique de la position"
    )
    portfolio = models.ForeignKey(
        Portfolio,
        on_delete=models.CASCADE,
        help_text="Portefeuille contenant cette position"
    )
    instrument = models.ForeignKey(
        'FinancialInstrument',
        on_delete=models.CASCADE,
        help_text="Instrument financier détenu"
    )
    quantity = models.IntegerField(
        help_text="Nombre de titres détenus"
    )
    purchase_price = models.DecimalField(
        max_digits=15,
        decimal_places=4,
        help_text="Prix moyen d'achat"
    )
    current_price = models.DecimalField(
        max_digits=15,
        decimal_places=4,
        help_text="Prix actuel de l'instrument"
    )
    total_value = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        help_text="Valeur totale de la position (quantité × prix)"
    )

class FinancialInstrument(models.Model):
    """
    Représente les instruments financiers disponibles pour la couverture.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        help_text="Identifiant unique de l'instrument"
    )
    name = models.CharField(
        max_length=100,
        help_text="Nom de l'instrument"
    )
    type = models.CharField(
        max_length=50,
        help_text="Type d'instrument (Action, Obligation, Option, etc.)"
    )

    currency = models.CharField(
        max_length=3,
        help_text="Devise de cotation de l'instrument"
    )
    current_price = models.DecimalField(
        max_digits=15,
        decimal_places=4,
        help_text="Prix actuel de l'instrument"
    )
