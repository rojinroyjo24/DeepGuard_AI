"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import about, home, index, predict_page, cuda_full, login_view, logout_view, register_view, profile_view, dashboard, view_detection, generate_report

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', home, name='home'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('register/', register_view, name='register'),
    path('profile/', profile_view, name='profile'),
    path('upload/', index, name='upload'),
    path('about/', about, name='about'),
    path('predict/', predict_page, name='predict'),
    path('cuda_full/',cuda_full,name='cuda_full'),
    path('dashboard/', dashboard, name='dashboard'),
    path('detection/<int:detection_id>/', view_detection, name='view_detection'),
    path('detection/<int:detection_id>/report/', generate_report, name='generate_report'),
]
