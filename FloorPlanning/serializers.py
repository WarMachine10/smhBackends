from rest_framework import serializers
from account.models import *
from .models import *
import os
class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ['project_name', 'width', 'length', 'bedroom', 'bathroom', 'car', 'temple', 'garden', 'living_room', 'store_room']

class UserFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserFile
        fields = '__all__'
        read_only_fields = ('user',)

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        
        # Modify the 'info' field to include PNG paths
        if 'info' in representation and representation['info']:
            info_with_paths = {}
            base_media_url = '/media/pngs/'  # Adjust this to your media URL

            for key, value in representation['info'].items():
                # If the key already contains the base_media_url, skip this transformation
                if not key.startswith(base_media_url):
                    file_name = os.path.basename(key)
                    full_path = f"{base_media_url}{file_name}"
                else:
                    full_path = key
                info_with_paths[full_path] = value

            representation['info'] = info_with_paths
        
        return representation

class SoilDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilData
        fields = ['id', 'user', 'soil_type', 'ground_water_depth', 'foundation_type', 'created_at']


class MapFileSerializer(serializers.ModelSerializer):
    map_html = serializers.SerializerMethodField()

    class Meta:
        model = MapFile
        fields = ['id', 'user', 'map_html', 'created_at']

    def get_map_html(self, obj):
        return obj.map_url