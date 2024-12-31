from rest_framework import serializers
import os
from .models import FloorplanningProject, UserFile, SoilData, MapFile
from loguru import logger

class UserFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserFile
        fields = ['id', 'png_image', 'dxf_file', 'gif_file', 'info', 'created_at']

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        if 'info' in representation and representation['info']:
            info_with_s3_urls = {}
            for key, value in representation['info'].items():
                if key.startswith('/media/'):
                    # Extract filename from the path
                    filename = os.path.basename(key)
                    # Construct S3 URL
                    s3_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/media/pngs/{filename}"
                    info_with_s3_urls[s3_url] = value
                else:
                    info_with_s3_urls[key] = value
            representation['info'] = info_with_s3_urls
        return representation

class SoilDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilData
        fields = ['id', 'soil_type', 'ground_water_depth', 'foundation_type', 'created_at']

class MapFileSerializer(serializers.ModelSerializer):
    map_html = serializers.SerializerMethodField()

    class Meta:
        model = MapFile
        fields = ['id', 'map_html', 'created_at']

    def get_map_html(self, obj):
        return obj.map_url

class CreateFloorplanningProjectSerializer(serializers.ModelSerializer):
    user_files = UserFileSerializer(many=True, read_only=True)

    class Meta:
        model = FloorplanningProject
        fields = [
            'id', 'project_name', 'width', 'length', 'bedroom', 'bathroom',
            'car', 'temple', 'garden', 'living_room', 'store_room', 'user_files'
        ]

    def create(self, validated_data):
        subproject = self.context.get('subproject')
        if not subproject:
            raise serializers.ValidationError("SubProject is required")
        
        try:
            floorplanning_project = FloorplanningProject.objects.create(
                subproject=subproject,
                **validated_data
            )
            return floorplanning_project
        except Exception as e:
            logger.error(f"Error in serializer create: {str(e)}")
            raise