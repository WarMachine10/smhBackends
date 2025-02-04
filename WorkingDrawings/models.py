from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.conf import settings
from project.models import *
# from Subscriptions.models import 
# import os

class WorkingDrawingSection(models.Model):
    """Base class for section-specific data"""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')

    class Meta:
        abstract = True

class WorkingDrawingProject(models.Model):
    """Main container for all working drawing sections"""
    subproject = models.OneToOneField(
        SubProject,
        on_delete=models.CASCADE,
        related_name='working_drawing'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def initialize_sections(self):
        """Initialize all section states in subproject"""
        default_state = {
            'status': 'in_progress',
            'sections': {
                'electrical': {'status': 'pending', 'last_updated': None},
                'plumbing': {'status': 'pending', 'last_updated': None},
                'structural': {'status': 'pending', 'last_updated': None}
            }
        }
        if not self.subproject.state:
            self.subproject.state = default_state
            self.subproject.save()

    def save(self, *args, **kwargs):
        is_new = not self.pk
        super().save(*args, **kwargs)
        if is_new:
            self.initialize_sections()

    def update_section_status(self, section_name, status):
        """Update status for a specific section"""
        state = self.subproject.state
        state['sections'][section_name]['status'] = status
        state['sections'][section_name]['last_updated'] = timezone.now().isoformat()
        
        # Update overall status
        all_completed = all(
            section['status'] == 'completed' 
            for section in state['sections'].values()
        )
        state['status'] = 'completed' if all_completed else 'in_progress'
        
        self.subproject.state = state
        self.subproject.save()

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
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Update working drawing section status
        if self.status in ['completed', 'failed']:
            working_drawing = self.subproject.working_drawing
            if working_drawing:
                working_drawing.update_section_status('electrical', self.status)

    
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
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Update working drawing section status
        if self.status in ['completed', 'failed']:
            working_drawing = self.subproject.working_drawing
            if working_drawing:
                working_drawing.update_section_status('electrical', self.status)

class WaterSupply(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='water_supply')
    input_file = models.FileField(upload_to='plumbing/watersupply/inputs/')
    output_file = models.FileField(upload_to='plumbing/watersupply/outputs/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def input_file_url(self):
        return self.input_file.url if self.input_file else None

    @property
    def output_file_url(self):
        return self.output_file.url if self.output_file else None

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.status in ['completed', 'failed']:
            working_drawing = self.subproject.working_drawing
            if working_drawing:
                working_drawing.update_section_status('plumbing', self.status)
                
class PlumbingComplete(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='plumbing_complete')
    input_file = models.FileField(upload_to='plumbing/complete/inputs/')
    output_file = models.FileField(upload_to='plumbing/complete/outputs/', null=True, blank=True)
    boq_output_file = models.FileField(upload_to='plumbing/complete/boq/', null=True, blank=True)  # New field
    input1_option = models.CharField(max_length=3, default='yes')  # For user_input1
    input2_option = models.CharField(max_length=3, default='yes')  # For user_input2
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def input_file_url(self):
        return self.input_file.url if self.input_file else None

    @property
    def output_file_url(self):
        return self.output_file.url if self.output_file else None

    @property
    def boq_output_file_url(self):  # New property
        return self.boq_output_file.url if self.boq_output_file else None
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        if self.status in ['completed', 'failed']:
            working_drawing = self.subproject.working_drawing
            if working_drawing:
                working_drawing.update_section_status('plumbing', self.status)


class StructuralMain(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='structural_main')
    input_file = models.FileField(upload_to='structural/main/inputs/')
    output_file = models.FileField(upload_to='structural/main/outputs/', null=True, blank=True)
    column_info = models.JSONField(null=True, blank=True)  # Store column data directly as JSON
    beam_info = models.JSONField(null=True, blank=True)    # Store beam data directly as JSON
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def input_file_url(self):
        return self.input_file.url if self.input_file else None

    @property
    def output_file_url(self):
        return self.output_file.url if self.output_file else None

class StructuralCenterLine(BaseFileModel):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='structural_centerline')
    input_file = models.FileField(upload_to='structural/centerline/inputs/')
    output_file = models.FileField(upload_to='structural/centerline/outputs/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ], default='pending')
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def input_file_url(self):
        return self.input_file.url if self.input_file else None

    @property
    def output_file_url(self):
        return self.output_file.url if self.output_file else None
    
    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        # Update working drawing section status
        if self.status in ['completed', 'failed']:
            working_drawing = self.subproject.working_drawing
            if working_drawing:
                working_drawing.update_section_status('structural', self.status)



