from django.contrib import admin
from .models import *

@admin.register(StructuredProduct)
class StructuredProductAdmin(admin.ModelAdmin):
    list_display = ('id', 'reference_nav', 'start_date', 'end_date')
    search_fields = ('id',)

@admin.register(Index)
class IndexAdmin(admin.ModelAdmin):
    list_display = ('code', 'name', 'country', 'currency')
    search_fields = ('code', 'name')
    list_filter = ('country', 'currency')

@admin.register(IndexBasket)
class IndexBasketAdmin(admin.ModelAdmin):
    list_display = ('id', 'product', 'basket_performance', 'annual_performance')
    search_fields = ('id', 'product__id')

@admin.register(MarketData)
class MarketDataAdmin(admin.ModelAdmin):
    list_display = ('index', 'timestamp', 'value', 'data_type')
    list_filter = ('data_type', 'index')
    date_hierarchy = 'timestamp'

@admin.register(PerformanceCheck)
class PerformanceCheckAdmin(admin.ModelAdmin):
    list_display = ('check_date', 'basket_performance', 'max_annual_return', 'threshold_reached')
    list_filter = ('threshold_reached', 'check_date')

@admin.register(DividendPayment)
class DividendPaymentAdmin(admin.ModelAdmin):
    list_display = ('payment_date', 'amount', 'status', 'excluded_index')
    list_filter = ('status', 'payment_date')
    search_fields = ('excluded_index__code',)

@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ('id', 'current_value', 'target_value', 'last_rebalance')
    search_fields = ('id', 'structured_product__id')

@admin.register(FinancialInstrument)
class FinancialInstrumentAdmin(admin.ModelAdmin):
    list_display = ('name', 'type', 'currency', 'current_price')
    list_filter = ('type', 'currency')
    search_fields = ('name',)

@admin.register(Holding)
class HoldingAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'instrument', 'quantity', 'total_value')
    list_filter = ('instrument__type',)
    search_fields = ('portfolio__id', 'instrument__name')