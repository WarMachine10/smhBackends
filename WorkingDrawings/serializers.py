from rest_framework import serializers
from .models import *
from django.contrib.auth import get_user_model

User = get_user_model()


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
        return obj.input_file_url

    def get_output_file_url(self, obj):
        return obj.output_file_url

    def validate(self, data):
        # Ensure the subproject is of type working_drawing
        if data['subproject'].type != 'working_drawing':
            raise serializers.ValidationError(
                "Fixtures can only be added to working drawing subprojects"
            )
        return data

class ElectricalWiringSerializer(serializers.ModelSerializer):
    input_file_url = serializers.SerializerMethodField()
    output_file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = ElectricalWiring
        fields = [
            'id', 'user', 'subproject', 'input_file', 'output_file',
            'input_file_url', 'output_file_url', 'size',
            'status', 'created_at', 'updated_at'
        ]
        read_only_fields = ['output_file', 'status', 'created_at', 'updated_at',
                           'input_file_url', 'output_file_url']

    def get_input_file_url(self, obj):
        return obj.input_file_url

    def get_output_file_url(self, obj):
        return obj.output_file_url

class SubProjectSerializer(serializers.ModelSerializer):
    electrical_fixtures = ElectricalFixturesSerializer(many=True, read_only=True)
    
    class Meta:
        model = SubProject
        fields = ['id', 'project', 'type', 'state', 'created_at', 'electrical_fixtures']

class WorkingDrawingProjectSerializer(serializers.ModelSerializer):
    electrical_fixtures = serializers.SerializerMethodField()
    electrical_wiring = serializers.SerializerMethodField()

    class Meta:
        model = WorkingDrawingProject
        fields = [
            'id',
            'drawing_type',
            'created_at',
            'electrical_fixtures',
            'electrical_wiring'
        ]

    def get_electrical_fixtures(self, obj):
        if obj.drawing_type == 'electrical':
            fixtures = obj.subproject.electrical_fixtures.all()
            return ElectricalFixturesSerializer(fixtures, many=True).data
        return []

    def get_electrical_wiring(self, obj):
        if obj.drawing_type == 'electrical':
            wiring = obj.subproject.electrical_wiring.all()
            return ElectricalWiringSerializer(wiring, many=True).data
        return []


    # def get_plumbing_details(self, obj):
    #     if obj.drawing_type == 'plumbing':
    #         try:
    #             return PlumbingDetailsSerializer(obj.plumbing_details).data
    #         except PlumbingDrawing.DoesNotExist:
    #             return None
    #     return None

    # def get_structural_details(self, obj):
    #     if obj.drawing_type == 'structural':
    #         try:
    #             return StructuralDetailsSerializer(obj.structural_details).data
    #         except StructuralDrawing.DoesNotExist:
    #             return None
    #     return None

    def to_representation(self, instance):
        data = super().to_representation(instance)
        # Only include relevant details based on drawing type
        if instance.drawing_type == 'electrical':
            data.pop('plumbing_details', None)
            data.pop('structural_details', None)
        elif instance.drawing_type == 'plumbing':
            data.pop('electrical_details', None)
            data.pop('structural_details', None)
            data.pop('electrical_fixtures', None)  # Remove fixtures for non-electrical
        elif instance.drawing_type == 'structural':
            data.pop('electrical_details', None)
            data.pop('plumbing_details', None)
            data.pop('electrical_fixtures', None)  # Remove fixtures for non-electrical
        return data
    
