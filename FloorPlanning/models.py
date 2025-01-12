from django.db import models
from django.conf import settings
from django.contrib.auth import get_user_model
from project.models import BaseFileModel, SubProject

class FloorplanningProject(models.Model):
    subproject = models.ForeignKey(SubProject, on_delete=models.CASCADE, related_name='floorplanning')
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

class UserFile(BaseFileModel):
    floorplanning = models.ForeignKey(FloorplanningProject, on_delete=models.CASCADE, related_name='user_files')
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE)
    png_image = models.URLField(max_length=500, null=True, blank=True)
    dxf_file = models.URLField(max_length=500, null=True, blank=True)
    gif_file = models.URLField(max_length=500, null=True, blank=True)
    info = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class SoilData(models.Model):
    floorplanning = models.ForeignKey(FloorplanningProject, on_delete=models.CASCADE, related_name='soil_data')
    soil_type = models.CharField(max_length=255)
    ground_water_depth = models.CharField(max_length=255)
    foundation_type = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

class MapFile(models.Model):
    floorplanning = models.ForeignKey(FloorplanningProject, on_delete=models.CASCADE, related_name='map_files')
    map_path = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def map_url(self):
        return f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{self.map_path}"