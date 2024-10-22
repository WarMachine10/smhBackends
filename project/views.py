from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from .models import Project, WorkingDrawing, Plan, ThreeDModel, Concept
from .serializers import (
    ProjectSerializer, WorkingDrawingSerializer,
    PlanSerializer, ThreeDModelSerializer, ConceptSerializer
)

# Project Views

class ProjectCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ProjectSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(created_by=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ProjectListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        projects = Project.objects.filter(created_by=request.user)
        serializer = ProjectSerializer(projects, many=True)
        return Response(serializer.data)

# Working Drawing Views

class WorkingDrawingCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        serializer = WorkingDrawingSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(uploaded_by=request.user, project=project)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class WorkingDrawingListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        working_drawings = WorkingDrawing.objects.filter(project=project)
        serializer = WorkingDrawingSerializer(working_drawings, many=True)
        return Response(serializer.data)

# Plan Views

class PlanCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        serializer = PlanSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(uploaded_by=request.user, project=project)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class PlanListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        plans = Plan.objects.filter(project=project)
        serializer = PlanSerializer(plans, many=True)
        return Response(serializer.data)

# 3D Model Views

class ThreeDModelCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        serializer = ThreeDModelSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(uploaded_by=request.user, project=project)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ThreeDModelListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        models_3d = ThreeDModel.objects.filter(project=project)
        serializer = ThreeDModelSerializer(models_3d, many=True)
        return Response(serializer.data)

# Concept Views

class ConceptCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        serializer = ConceptSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(uploaded_by=request.user, project=project)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ConceptListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        project = get_object_or_404(Project, id=project_id, created_by=request.user)
        concepts = Concept.objects.filter(project=project)
        serializer = ConceptSerializer(concepts, many=True)
        return Response(serializer.data)
