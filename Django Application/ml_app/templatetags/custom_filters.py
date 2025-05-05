from django import template
from django.template.defaultfilters import stringfilter
from datetime import datetime

register = template.Library()

@register.filter
def divided_by(value, arg):
    """Divides the value by the argument"""
    try:
        return float(value) / float(arg)
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def times(value, arg):
    """Multiplies the value by the argument"""
    try:
        return float(value) * float(arg)
    except ValueError:
        return 0

@register.filter
def percentage(value, total):
    """Calculate percentage"""
    try:
        return float(value) / float(total) * 100
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
@stringfilter
def format_sqlite_date(value):
    """Format an SQLite date string (YYYY-MM-DD) into a more readable format"""
    try:
        date_obj = datetime.strptime(value, '%Y-%m-%d')
        return date_obj.strftime('%b %d, %Y')
    except (ValueError, TypeError):
        return value 