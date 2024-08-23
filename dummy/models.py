from django.db import models
from django.contrib.auth import get_user_model
from .models import *
from django.conf import settings

class Project(models.Model):
    project_name = models.CharField(max_length=255)
    width = models.IntegerField()
    length = models.IntegerField()
    bedroom = models.IntegerField()
    bathroom = models.IntegerField()
    car = models.IntegerField()
    temple = models.IntegerField()
    garden = models.IntegerField()
    living_room = models.IntegerField()
    store_room = models.IntegerField()
    def __str__(self):
        return self.project_name


class UserFile(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    png_image = models.URLField(max_length=500, null=True, blank=True)
    dxf_file = models.URLField(max_length=500, null=True, blank=True)
    gif_file = models.URLField(max_length=500, null=True, blank=True)
    info = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"UserFile(user={self.user}, png_image={self.png_image}, dxf_file={self.dxf_file},, gif_file={self.gif_file})"
    

class SoilData(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    soil_type = models.CharField(max_length=255)
    ground_water_depth = models.CharField(max_length=255)
    foundation_type = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"user={self.user}, created_at={self.created_at}"

class MapFile(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    map_path = models.CharField(max_length=255)  # Store only the path, not the full URL
    created_at = models.DateTimeField(auto_now_add=True)
    @property
    def map_url(self):
        return f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{self.map_path}"


