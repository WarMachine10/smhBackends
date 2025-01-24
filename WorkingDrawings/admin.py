from django.contrib import admin
from .models import *
# Register your models here.
# admin.site.register(ElectricalFixtures)

@admin.register(WorkingDrawingProject)
class WorkingDrawingProjectAdmin(admin.ModelAdmin):
    list_display = ['id', 'subproject', 'created_at']
    list_filter = ['created_at']
    search_fields = ['subproject__project__name']

@admin.register(ElectricalFixtures)
class ElectricalFixturesAdmin(admin.ModelAdmin):
    list_display = ['id', 'subproject', 'status', 'created_at', 'updated_at']
    list_filter = ['status', 'created_at']
    search_fields = ['subproject__project__name']

@admin.register(ElectricalWiring)
class ElectricalWiringAdmin(admin.ModelAdmin):
    list_display = ['id', 'subproject', 'status', 'created_at', 'updated_at']
    list_filter = ['status', 'created_at']
    search_fields = ['subproject__project__name']


