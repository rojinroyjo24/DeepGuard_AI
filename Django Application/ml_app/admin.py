from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import DetectionHistory
from django.utils.html import format_html
from django.db.models import Count, Avg, Sum, Case, When, IntegerField
from django.urls import reverse

# Register your models here.

# Customize User admin
class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_active', 'date_joined', 'detection_count')
    actions = ['delete_selected']
    
    def detection_count(self, obj):
        """Count the number of detections for this user"""
        count = DetectionHistory.objects.filter(user=obj).count()
        if count > 0:
            # Link to filtered view of detections
            url = reverse('admin:ml_app_detectionhistory_changelist') + f'?user__id__exact={obj.id}'
            return format_html('<a href="{}">{} detections</a>', url, count)
        return '0 detections'
    
    detection_count.short_description = 'Detections'

# Unregister the default User admin and register custom one
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)

# Custom admin for DetectionHistory
class DetectionHistoryAdmin(admin.ModelAdmin):
    list_display = ('video_name', 'user_link', 'result', 'confidence_score', 'frames_analyzed', 
                   'detection_time', 'upload_date', 'view_video')
    list_filter = ('result', 'upload_date', 'user')
    search_fields = ('video_name', 'original_filename', 'user__username')
    readonly_fields = ('video_preview', 'thumbnail_preview')
    
    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        # Prefetch related user to avoid N+1 queries
        return queryset.select_related('user')
    
    def user_link(self, obj):
        """Create a link to the user's change form"""
        url = reverse('admin:auth_user_change', args=[obj.user.id])
        return format_html('<a href="{}">{}</a>', url, obj.user.username)
    
    user_link.short_description = 'User'
    
    def view_video(self, obj):
        """Create a link to view the detection details"""
        url = reverse('ml_app:view_detection', args=[obj.id])
        return format_html('<a href="{}" target="_blank">View</a>', url)
    
    view_video.short_description = 'View'
    
    def video_preview(self, obj):
        """Show video preview in the detail view"""
        if obj.video_path:
            # Return an HTML video tag
            return format_html(
                '<video width="320" height="240" controls><source src="/media/{}" type="video/mp4">Your browser does not support the video tag.</video>',
                obj.video_path
            )
        return "No video available"
    
    video_preview.short_description = 'Video Preview'
    
    def thumbnail_preview(self, obj):
        """Show thumbnail preview in the detail view"""
        if obj.thumbnail_path:
            return format_html('<img src="/static/{}" width="150" height="auto" />', obj.thumbnail_path)
        return "No thumbnail available"
    
    thumbnail_preview.short_description = 'Thumbnail'
    
    # Custom Admin Actions
    actions = ['mark_as_real', 'mark_as_fake']
    
    def mark_as_real(self, request, queryset):
        queryset.update(result='real')
    mark_as_real.short_description = "Mark selected detections as real"
    
    def mark_as_fake(self, request, queryset):
        queryset.update(result='fake')
    mark_as_fake.short_description = "Mark selected detections as fake"

# Register the DetectionHistory model
admin.site.register(DetectionHistory, DetectionHistoryAdmin)

# Custom Admin Site Title and Header
admin.site.site_header = "DeepGuard AI Admin"
admin.site.site_title = "DeepGuard AI Admin Portal"
admin.site.index_title = "Welcome to DeepGuard AI Admin Portal"
