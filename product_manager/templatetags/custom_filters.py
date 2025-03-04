# product_manager/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Get an item from a dictionary using a key.
    Usage: {{ my_dict|get_item:key_var }}
    """
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    elif hasattr(dictionary, '__getitem__'):
        try:
            return dictionary[key]
        except (KeyError, IndexError, TypeError):
            return None
    return None