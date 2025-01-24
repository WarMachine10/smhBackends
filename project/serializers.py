from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import Project, SubProject
from loguru import logger
from FloorPlanning.serializers import CreateFloorplanningProjectSerializer, UserFileSerializer
from FloorPlanning.models import FloorplanningProject, UserFile
from WorkingDrawings.serializers import *
from WorkingDrawings.models import *
# from threed.serializers import ThreeDProjectSerializer
# from conceptualisation.serializers import ConceptualisationProjectSerializer
# from working_drawing.serializers import WorkingDrawingProjectSerializer


class ElectricalFixturesSerializer(serializers.ModelSerializer):
    input_file_url = serializers.SerializerMethodField()
    output_file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = ElectricalFixtures
        fields = [
            'id', 'user', 'subproject', 'input_file', 'output_file',
            'input_file_url', 'output_file_url', 'size',
            'status', 'created_at', 'updated_at'
        ]
        read_only_fields = ['output_file', 'status', 'created_at', 'updated_at',
                           'input_file_url', 'output_file_url']

    def get_input_file_url(self, obj):
        return obj.input_file_url if obj.input_file else None

    def get_output_file_url(self, obj):
        return obj.output_file_url if obj.output_file else None

class SubProjectSerializer(serializers.ModelSerializer):
    floorplanning = serializers.SerializerMethodField()
    working_drawing = serializers.SerializerMethodField()
    # electrical_fixtures = ElectricalFixturesSerializer(many=True, read_only=True)
    
    type = serializers.ChoiceField(choices=[
        ('floorplanning', 'Floorplanning'),
        ('working_drawing', 'Working Drawing')
    ])

    class Meta:
        model = SubProject
        fields = ['id', 'project', 'type', 'state', 'created_at', 
                 'floorplanning', 'working_drawing']

    def get_floorplanning(self, obj):
        if obj.type == 'floorplanning':
            try:
                floorplanning = obj.floorplanning
                return FloorplanningProjectSerializer(floorplanning).data
            except:
                return None
        return None

    def get_working_drawing(self, obj):
        if obj.type == 'working_drawing':
            try:
                working_drawing = obj.working_drawing
                working_drawing_data = WorkingDrawingProjectSerializer(working_drawing).data
                return working_drawing_data
            except:
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



    