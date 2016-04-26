from django import template

register = template.Library()

@register.filter
def divide(value, arg):
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return None

@register.filter
def fadd(value, arg):
    try:
        return float(value) + float(arg)
    except (ValueError, ZeroDivisionError):
        return None

from django.utils.safestring import mark_safe

import json

@register.filter(is_safe=True)
def js(obj):
    return mark_safe(json.dumps(obj))
