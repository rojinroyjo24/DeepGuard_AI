from django.contrib.admin.views.decorators import staff_member_required
from django.shortcuts import render
from django.db.models import Count, Avg, Sum, Q, F, Min, Max
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta
from .models import DetectionHistory
import json

@staff_member_required
def admin_dashboard(request):
    """Custom admin dashboard with statistics and charts"""
    
    # Get overall stats
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    total_detections = DetectionHistory.objects.count()
    
    # Detection results breakdown
    detection_results = DetectionHistory.objects.values('result').annotate(
        count=Count('id')
    ).order_by('result')
    
    # Prepare data for pie chart
    results_labels = [item['result'].capitalize() for item in detection_results]
    results_data = [item['count'] for item in detection_results]
    results_colors = ['#28a745', '#dc3545', '#6c757d']  # Green for real, red for fake, gray for unknown
    
    # Get recent activity (last 7 days)
    last_week = timezone.now() - timedelta(days=7)
    recent_detections = DetectionHistory.objects.filter(
        upload_date__gte=last_week
    ).count()
    
    new_users = User.objects.filter(
        date_joined__gte=last_week
    ).count()
    
    # Statistics by day for the last 30 days
    last_month = timezone.now() - timedelta(days=30)
    daily_stats = DetectionHistory.objects.filter(
        upload_date__gte=last_month
    ).extra(
        select={'day': "strftime('%Y-%m-%d', upload_date)"}
    ).values('day').annotate(
        total=Count('id'),
        real_count=Count('id', filter=Q(result='real')),
        fake_count=Count('id', filter=Q(result='fake'))
    ).order_by('day')
    
    # Prepare data for time series chart - handle string dates properly for SQLite
    time_labels = [item['day'] for item in daily_stats]
    real_series = [item['real_count'] for item in daily_stats]
    fake_series = [item['fake_count'] for item in daily_stats]
    
    # Most active users
    active_users_data = DetectionHistory.objects.values(
        'user__username'
    ).annotate(
        detection_count=Count('id')
    ).order_by('-detection_count')[:10]
    
    # Confidence score distribution
    confidence_distribution = DetectionHistory.objects.values('result').annotate(
        avg_confidence=Avg('confidence_score')
    ).order_by('result')
    
    # Average detection time
    avg_detection_time = DetectionHistory.objects.aggregate(
        avg_time=Avg('detection_time')
    )['avg_time'] or 0
    
    # Context with all data for the template
    context = {
        'total_users': total_users,
        'active_users': active_users,
        'total_detections': total_detections,
        'recent_detections': recent_detections,
        'new_users': new_users,
        'avg_detection_time': round(avg_detection_time, 2),
        
        # Chart data (serialized for JavaScript)
        'results_chart_data': json.dumps({
            'labels': results_labels,
            'datasets': [{
                'data': results_data,
                'backgroundColor': results_colors
            }]
        }),
        
        'time_series_data': json.dumps({
            'labels': time_labels,
            'datasets': [
                {
                    'label': 'Real Videos',
                    'data': real_series,
                    'borderColor': '#28a745',
                    'backgroundColor': 'rgba(40, 167, 69, 0.1)'
                },
                {
                    'label': 'Fake Videos',
                    'data': fake_series,
                    'borderColor': '#dc3545',
                    'backgroundColor': 'rgba(220, 53, 69, 0.1)'
                }
            ]
        }),
        
        'active_users_data': active_users_data,
        'confidence_distribution': confidence_distribution
    }
    
    return render(request, 'admin/dashboard.html', context)

@staff_member_required
def admin_combined_dashboard(request):
    """Combined dashboard view for admin users with all project details"""
    
    # Get statistics
    total_users = User.objects.count()
    active_users = User.objects.filter(is_active=True).count()
    total_detections = DetectionHistory.objects.count()
    
    # Get user data
    users = User.objects.all().order_by('-date_joined')
    
    # User detection counts
    user_detection_counts = DetectionHistory.objects.values('user__username').annotate(
        count=Count('id'),
        real_count=Count('id', filter=Q(result='real')),
        fake_count=Count('id', filter=Q(result='fake'))
    ).order_by('-count')
    
    # Recent detections
    recent_detections = DetectionHistory.objects.select_related('user').order_by('-upload_date')[:20]
    
    # Result distribution
    result_distribution = DetectionHistory.objects.values('result').annotate(
        count=Count('id')
    ).order_by('result')
    
    # Detection time stats
    detection_time_stats = {
        'avg': DetectionHistory.objects.aggregate(Avg('detection_time'))['detection_time__avg'] or 0,
        'min': DetectionHistory.objects.aggregate(Min('detection_time'))['detection_time__min'] or 0,
        'max': DetectionHistory.objects.aggregate(Max('detection_time'))['detection_time__max'] or 0,
    }
    
    # Real vs Fake distribution over time (last 90 days)
    last_90_days = timezone.now() - timedelta(days=90)
    
    # SQLite-compatible version - using strftime to format dates
    time_distribution = DetectionHistory.objects.filter(
        upload_date__gte=last_90_days
    ).extra(
        select={'week': "strftime('%Y-%m-%d', upload_date, 'weekday 0', '-6 days')"}
    ).values('week').annotate(
        total=Count('id'),
        real=Count('id', filter=Q(result='real')),
        fake=Count('id', filter=Q(result='fake'))
    ).order_by('week')
    
    context = {
        # Stats
        'total_users': total_users,
        'active_users': active_users,
        'total_detections': total_detections,
        
        # Lists
        'users': users,
        'user_detection_counts': user_detection_counts,
        'recent_detections': recent_detections,
        'result_distribution': result_distribution,
        
        # Time data
        'detection_time_stats': detection_time_stats,
        'time_distribution': time_distribution,
        
        # Additional UI flags
        'is_combined_dashboard': True
    }
    
    return render(request, 'admin_dashboard.html', context) 