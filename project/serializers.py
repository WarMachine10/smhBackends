from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Project, SubProject
from loguru import logger
from FloorPlanning.serializers import CreateFloorplanningProjectSerializer, UserFileSerializer
from FloorPlanning.models import FloorplanningProject, UserFile
# from threed.serializers import ThreeDProjectSerializer
# from conceptualisation.serializers import ConceptualisationProjectSerializer
# from working_drawing.serializers import WorkingDrawingProjectSerializer


class SubProjectSerializer(serializers.ModelSerializer):
    floorplanning = serializers.SerializerMethodField()
    type = serializers.ChoiceField(choices=['floorplanning'])  # This enforces correct type value

    class Meta:
        model = SubProject
        fields = ['id', 'project', 'type', 'state', 'created_at', 'floorplanning']

    def get_floorplanning(self, obj):
        # Add logging to debug
        logger.debug(f"Getting floorplanning for subproject {obj.id} of type {obj.type}")
        
        # Check both 'floorplanning' and 'floorplann' for backward compatibility
        if obj.type in ['floorplanning', 'floorplann']:
            try:
                floorplanning = FloorplanningProject.objects.select_related('subproject').get(subproject=obj)
                return CreateFloorplanningProjectSerializer(floorplanning).data
            except FloorplanningProject.DoesNotExist:
                logger.warning(f"No FloorplanningProject found for subproject {obj.id}")
                return None
            except Exception as e:
                logger.error(f"Error getting floorplanning data: {str(e)}")
                return None
        return None



class ProjectSerializer(serializers.ModelSerializer):
    user = serializers.PrimaryKeyRelatedField(queryset=get_user_model().objects.all())
    
    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'created_at', 'user']

class ProjectWithSubProjectsSerializer(serializers.ModelSerializer):
    subprojects = SubProjectSerializer(many=True, read_only=True)

    class Meta:
        model = Project
        fields = ['id', 'name', 'description', 'created_at', 'user', 'subprojects']



    