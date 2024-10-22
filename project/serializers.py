from rest_framework import serializers
from .models import Project, WorkingDrawing, Plan, ThreeDModel, Concept
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class ProjectSerializer(serializers.ModelSerializer):
    created_by = UserSerializer(read_only=True)
    
    class Meta:
        model = Project
        fields = ['id', 'project_name', 'description', 'location', 'created_by', 'created_at', 'updated_at']

class WorkingDrawingSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = WorkingDrawing
        fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'sheet_size', 'scale']

class PlanSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = Plan
        fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'plan_type', 'scale']

class ThreeDModelSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = ThreeDModel
        fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'model_type', 'render_quality']

class ConceptSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = Concept
        fields = ['id', 'project', 'uploaded_by', 'uploaded_at', 'concept_type']
