from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(Project)
admin.site.register(UserFile)
admin.site.register(SoilData)
admin.site.register(MapFile)