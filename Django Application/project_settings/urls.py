"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static
from ml_app.admin_views import admin_dashboard, admin_combined_dashboard

# Customize admin site
admin.site.index_template = 'admin/custom_index.html'

urlpatterns = [
    # Custom admin dashboard URLs first (before admin site URLs)
    path('admin/dashboard/', admin_dashboard, name='admin_dashboard'),
    path('admin-dashboard/', admin_combined_dashboard, name='admin_combined_dashboard'),
    # Main admin site URLs
    path('admin/', admin.site.urls),
    path('', include('ml_app.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
