from rest_framework import serializers
from .models import *
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ('id', 'project_name', 'created_by', 'created_at', 'modified_at',
                  'input_file_url', 'output_file_url', 'plumbing', 'electrical', 'structural')

class WorkingDrawingSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = Project
        fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'sheet_size', 'scale']

# class PlanSerializer(serializers.ModelSerializer):
#     uploaded_by = UserSerializer(read_only=True)

#     class Meta:
#         model = Plan
#         fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'plan_type', 'scale']

# class ThreeDModelSerializer(serializers.ModelSerializer):
#     uploaded_by = UserSerializer(read_only=True)

#     class Meta:
#         model = ThreeDModel
#         fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'model_type', 'render_quality']

# class ConceptSerializer(serializers.ModelSerializer):
#     uploaded_by = UserSerializer(read_only=True)

#     class Meta:
#         model = Concept
#         fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'concept_type']
