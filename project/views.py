from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from rest_framework_simplejwt.authentication import JWTAuthentication
from .models import *
import boto3,shutil
from loguru import logger
import tempfile,re
from io import StringIO
from pathlib import Path
from django.conf import settings,Settings
import tempfile
from .serializers import *
from .Scripts.WorkingDrawings.AutoSOP.SopGenerator import process_dxf_file
from .Scripts.WorkingDrawings.Plumbing.Plumb import complete_pipeline
import pandas as pd
import numpy as np
import json
import ezdxf

# Project Views

TEMP_FOLDER = os.path.join(settings.BASE_DIR,'Temp','crap')  # Replace with a real folder path, e.g., '/tmp/dxf_processing'

# Set up a StringIO log capture stream
log_stream = StringIO()
# logger.basicConfig(stream=log_stream, level=logger.INFO)

class ProcessDXFView(APIView):
    parser_classes = [MultiPartParser]
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        dxf_file = request.FILES.get('file')
        project_name = request.data.get('project_name')
        process_plumbing = request.data.get('process_plumbing', False)
        process_electrical = request.data.get('process_electrical', False)
        process_structural = request.data.get('process_structural', False)

        if not dxf_file:
            return Response(
                {"error": "No file provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not project_name:
            return Response(
                {"error": "Project name is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Initialize the S3 client
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_S3_REGION_NAME
            )
            bucket_name = settings.AWS_STORAGE_BUCKET_NAME
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            return Response(
                {"error": "Failed to initialize S3 connection"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Ensure the temp directory exists
        temp_dir = os.path.join(TEMP_FOLDER, project_name)
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save input file to the temp directory
            input_temp_path = os.path.join(temp_dir, 'input.dxf')
            with open(input_temp_path, 'wb') as temp_file:
                for chunk in dxf_file.chunks():
                    temp_file.write(chunk)

            # Define output path in the temp directory
            output_temp_path = os.path.join(temp_dir, 'output.dxf')

            # Upload input file to S3
            input_file_key = f"uploads/{project_name}/{dxf_file.name}"
            with open(input_temp_path, 'rb') as file_obj:
                s3_client.upload_fileobj(file_obj, bucket_name, input_file_key)
            input_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{input_file_key}"

            logger.info(f"Processing DXF file for project: {project_name}")
            # Process the file and get material_counts
            material_counts = complete_pipeline(
                input_temp_path,
                output_temp_path,
                block_file=os.path.join(settings.BASE_DIR,'assets','SMH-Blocks.dxf'),
                excel_file_path=os.path.join(settings.BASE_DIR,'assets','PlumbingPrices.xlsx')
            )

            if not material_counts:
                logger.error(f"DXF processing failed for project: {project_name}")
                return Response(
                    {"error": "DXF processing failed"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # Prepare plumbing processing if requested
            plumbing_result = None
            if process_plumbing:
                logger.info(f"Processing plumbing for project: {project_name}")
                try:
                    plumbing_output_filename = 'Final_Floor_Plan_with_Drainage_Pipes.dxf'
                    blocksDxf = os.path.join(settings.BASE_DIR, 'assets', 'SMH-Blocks.dxf')
                    plumbExcel = os.path.join(settings.BASE_DIR, 'assets', 'PlumbingPrices.xlsx')
                    material=complete_pipeline(input_temp_path, output_temp_path, blocksDxf, plumbExcel)

                    # Upload plumbing result to S3
                    plumbing_file_key = f"processed/{project_name}/{plumbing_output_filename}"
                    with open(plumbing_output_filename, 'rb') as file_obj:
                        s3_client.upload_fileobj(file_obj, bucket_name, plumbing_file_key)
                    plumbing_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{plumbing_file_key}"
                    
                    plumbing_result = {
                        "output_file_url": plumbing_file_url,
                        "output_file_name": plumbing_output_filename,
                        "additional_result": {
                            "MaterialList": material
                        }
                    }
                except Exception as e:
                    logger.error(f"Plumbing processing failed for project: {project_name}: {str(e)}")

            # Create or update the project instance
            project, created = Project.objects.update_or_create(
                project_name=project_name,
                defaults={
                    'created_by': request.user,
                    'input_file_url': input_file_url,
                    'output_file_url': f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/processed/{project_name}/Final_output.dxf",
                    'plumbing': plumbing_result,  # Store plumbing result here
                    'electrical': None,  # Set default values
                    'structural': None,  # Set default values
                }
            )

            # Prepare response data
            project_data = ProjectSerializer(project).data
            response_data = {
                "project": {
                    **project_data,
                    "plumbing": plumbing_result if plumbing_result else {"output_file_url": None, "required": process_plumbing},
                    "electrical": {"output_file_url": None, "required": process_electrical},
                    "structural": {"output_file_url": None, "required": process_structural},
                    "layer_names": material_counts.get('Layer_names', []),
                    "number_overlapped_lines": material_counts.get('Number_overlapped_lines', 0),
                }
            }

            logger.info(f"Successfully processed DXF file for project: {project_name}")
            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Error processing project {project_name}: {str(e)}")
            return Response(
                {"error": f"Processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            # Clean up the temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

class ProjectListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        projects = Project.objects.filter(created_by=request.user)
        serializer = ProjectSerializer(projects, many=True)
        return Response(serializer.data)

# Working Drawing Views

# class WorkingDrawingCreateView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         serializer = WorkingDrawingSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(uploaded_by=request.user, project=project)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# class WorkingDrawingListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         working_drawings = WorkingDrawing.objects.filter(project=project)
#         serializer = WorkingDrawingSerializer(working_drawings, many=True)
#         return Response(serializer.data)

# # Plan Views

# class PlanCreateView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         serializer = PlanSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(uploaded_by=request.user, project=project)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# class PlanListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         plans = Plan.objects.filter(project=project)
#         serializer = PlanSerializer(plans, many=True)
#         return Response(serializer.data)

# # 3D Model Views

# class ThreeDModelCreateView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         serializer = ThreeDModelSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(uploaded_by=request.user, project=project)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# class ThreeDModelListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         models_3d = ThreeDModel.objects.filter(project=project)
#         serializer = ThreeDModelSerializer(models_3d, many=True)
#         return Response(serializer.data)

# # Concept Views

# class ConceptCreateView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         serializer = ConceptSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(uploaded_by=request.user, project=project)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# class ConceptListView(APIView):
#     permission_classes = [IsAuthenticated]

#     def get(self, request, project_id):
#         project = get_object_or_404(Project, id=project_id, created_by=request.user)
#         concepts = Concept.objects.filter(project=project)
#         serializer = ConceptSerializer(concepts, many=True)
#         return Response(serializer.data)












