from rest_framework import serializers
from .models import *
from django.contrib.auth import get_user_model

User = get_user_model()

# working_drawings/serializers.py

class FileSerializer(serializers.Serializer):
    url = serializers.URLField()
    size = serializers.IntegerField()

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

    def create(self, validated_data):
        # Set size from input_file before creating instance
        input_file = validated_data.get('input_file')
        if input_file:
            validated_data['size'] = input_file.size
        return super().create(validated_data)

class ElectricalWiringSerializer(serializers.ModelSerializer):
    created_by = serializers.PrimaryKeyRelatedField(source='user', read_only=True)
    input_file = serializers.SerializerMethodField()
    output_file = serializers.SerializerMethodField()

    class Meta:
        model = ElectricalWiring
        fields = [
            'id', 'created_by', 'input_file', 'output_file',
            'status', 'created_at', 'updated_at'
        ]

    def get_input_file(self, obj):
        if obj.input_file:
            return {
                'url': obj.input_file.url,
                'size': obj.size
            }
        return None

    def get_output_file(self, obj):
        if obj.output_file:
            return {
                'url': obj.output_file.url,
                'size': obj.size
            }
        return None

class ElectricalSectionSerializer(serializers.Serializer):
    fixtures = ElectricalFixturesSerializer(many=True, read_only=True)
    wiring = ElectricalWiringSerializer(many=True, read_only=True)

class SectionsSerializer(serializers.Serializer):
    electrical = serializers.SerializerMethodField()
    plumbing = serializers.SerializerMethodField()
    structural = serializers.SerializerMethodField()

    def get_electrical(self, obj):
        electrical_fixtures = ElectricalFixtures.objects.filter(subproject=obj.subproject)
        electrical_wiring = ElectricalWiring.objects.filter(subproject=obj.subproject)
        return {
            'fixtures': ElectricalFixturesSerializer(electrical_fixtures, many=True).data,
            'wiring': ElectricalWiringSerializer(electrical_wiring, many=True).data
        }

    def get_plumbing(self, obj):
        return {
            'pipes': [],
            'fixtures': []
        }

    def get_structural(self, obj):
        return {
            'beams': [],
            'columns': []
        }

class WorkingDrawingProjectSerializer(serializers.ModelSerializer):
    sections = SectionsSerializer(source='*', read_only=True)

    class Meta:
        model = WorkingDrawingProject
        fields = ['id', 'sections', 'created_at']

class SubProjectSerializer(serializers.ModelSerializer):
    working_drawing = WorkingDrawingProjectSerializer(read_only=True)
    floorplanning = serializers.SerializerMethodField()

    class Meta:
        model = SubProject
        fields = [
            'id', 'project', 'type', 'state', 'created_at',
            'floorplanning', 'working_drawing'
        ]
        read_only_fields = ['state']

    def get_floorplanning(self, obj):
        if obj.type == 'floorplanning':
            try:
                return FloorplanningProjectSerializer(obj.floorplanning).data
            except:
                return None
        return None