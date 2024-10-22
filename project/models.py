from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.conf import settings
# Model to store basic project details
class Project(models.Model):
    project_name = models.CharField(max_length=255)
    description = models.TextField()
    location = models.CharField(max_length=255)
    created_by = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.project_name

# Model for Working Drawings
class WorkingDrawing(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="working_drawings")
    uploaded_by = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # file = models.FileField(upload_to="media/working_drawings/")
    sheet_size = models.CharField(max_length=10)
    scale = models.CharField(max_length=50)

# Model for Plan
class Plan(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="plans")
    uploaded_by = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # file = models.FileField(upload_to="media/plans/")
    plan_type = models.CharField(max_length=50)
    scale = models.CharField(max_length=50)

# Model for 3D Model
class ThreeDModel(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="three_d_models")
    uploaded_by = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # file = models.FileField(upload_to="media/3d_models/")
    model_type = models.CharField(max_length=50)
    render_quality = models.CharField(max_length=50)

# Model for Concept
class Concept(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="concepts")
    uploaded_by = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # file = models.FileField(upload_to="media/concepts/")
    concept_type = models.CharField(max_length=50)