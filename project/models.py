from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model

class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='projects')

class SubProject(models.Model):
    project = models.ForeignKey(Project, related_name='subprojects', on_delete=models.CASCADE)
    type = models.CharField(max_length=20)
    state = models.JSONField(default=dict, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

class BaseFileModel(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    size = models.PositiveIntegerField()  # Size of the file in bytes
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        abstract = True

    def __str__(self):
        return f"{self.user.username} - {self.size} bytes"