from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import os

# Create your models here.

class DetectionHistory(models.Model):
    RESULT_CHOICES = (
        ('real', 'Real'),
        ('fake', 'Fake'),
        ('unknown', 'Unknown'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detection_history')
    video_name = models.CharField(max_length=255)
    original_filename = models.CharField(max_length=255)
    upload_date = models.DateTimeField(default=timezone.now)
    result = models.CharField(max_length=10, choices=RESULT_CHOICES)
    confidence_score = models.FloatField(default=0.0)
    frames_analyzed = models.IntegerField(default=0)
    detection_time = models.FloatField(help_text="Time taken for detection in seconds", default=0.0)
    
    # Store paths to relevant files
    video_path = models.CharField(max_length=500)
    thumbnail_path = models.CharField(max_length=500, null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.video_name} - {self.upload_date.strftime('%Y-%m-%d')}"
    
    class Meta:
        ordering = ['-upload_date']
        verbose_name_plural = "Detection Histories"
