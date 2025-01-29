from celery import shared_task
from celery.loaders import app
from celery.schedules import crontab

from .services.holding_service import HoldingService

@shared_task
def update_holdings_daily():
    """Mise à jour quotidienne des holdings"""
    service = HoldingService()
    service.update_holdings()

@shared_task
def check_portfolio_balance():
    """Vérification du rebalancement"""
    service = HoldingService()
    imbalanced = service.check_portfolio_balance()
    if imbalanced:
        # Notification au gérant
        pass

# Configuration Celery (celery.py)
app.conf.beat_schedule = {
    'update-holdings': {
        'task': 'product_manager.tasks.update_holdings_daily',
        'schedule': crontab(hour=18),  # Mise à jour à 18h
    },
    'check-balance': {
        'task': 'product_manager.tasks.check_portfolio_balance',
        'schedule': crontab(hour='9,13,17'),  # 3 vérifications par jour
    },
}