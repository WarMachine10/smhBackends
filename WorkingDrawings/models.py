from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.conf import settings
from project.models import *
# from Subscriptions.models import 
# import os
class WorkingDrawingProject(models.Model):
    DRAWING_TYPES = [
        ('electrical', 'Electrical'),
        ('plumbing', 'Plumbing'),
        ('structural', 'Structural')
    ]
    
    subproject = models.OneToOneField(
        SubProject,
        on_delete=models.CASCADE,
        related_name='working_drawing'
    )
    drawing_type = models.CharField(max_length=20, choices=DRAWING_TYPES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

class ElectricalFixtures(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='electrical_fixtures')
    input_file = models.FileField(upload_to='electrical/fixtures/inputs/')
    output_file = models.FileField(upload_to='electrical/fixtures/outputs/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']

    def get_s3_url(self, file_field):
        if file_field and hasattr(file_field, 'url'):
            return file_field.url
        return None

    @property
    def input_file_url(self):
        return self.get_s3_url(self.input_file)

    @property
    def output_file_url(self):
        return self.get_s3_url(self.output_file)
    
class ElectricalWiring(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='electrical_wiring')
    input_file = models.FileField(upload_to='electrical/wiring/inputs/')
    output_file = models.FileField(upload_to='electrical/wiring/outputs/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def input_file_url(self):
        if self.input_file:
            return self.input_file.url
        return None

    @property
    def output_file_url(self):
        if self.output_file:
            return self.output_file.url
        return None