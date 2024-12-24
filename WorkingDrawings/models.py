from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.conf import settings
import os

# Model to store basic project details
class Project(models.Model):
    project_name = models.CharField(max_length=100)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)
    input_file_url = models.URLField(null=True, blank=True)
    output_file_url = models.URLField(null=True, blank=True)
    plumbing = models.JSONField(null=True, blank=True)
    electrical = models.JSONField(null=True, blank=True)
    structural = models.JSONField(null=True, blank=True)

    def __str__(self):
        return self.project_name