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
from .Scripts.WorkingDrawings.Electrical.Lights import main_final
from .Scripts.WorkingDrawings.Structural.Struct import pipeline_main
from .Scripts.WorkingDrawings.BOQs.boq import main_electrical,main_plumbing,main_structure
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
        project_name = request.data.get('project_name', '').strip()
        process_plumbing = request.data.get('process_plumbing', 'false').lower() == 'true'
        process_electrical = request.data.get('process_electrical', 'false').lower() == 'true'
        process_structural = request.data.get('process_structural', 'false').lower() == 'true'

        if not dxf_file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        if not project_name:
            return Response({"error": "Project name is required"}, status=status.HTTP_400_BAD_REQUEST)

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
            return Response({"error": "Failed to initialize S3 connection"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        temp_dir = os.path.join(TEMP_FOLDER)
        os.makedirs(temp_dir, exist_ok=True)
        try:
            # Save the uploaded file temporarily
            input_temp_path = os.path.join(temp_dir, 'input.dxf')
            with open(input_temp_path, 'wb') as temp_file:
                for chunk in dxf_file.chunks():
                    temp_file.write(chunk)

            # Process DXF file
            output_temp_path = os.path.join(temp_dir, 'output.dxf')
            processing_result, final_output_path = process_dxf_file(input_temp_path, output_temp_path, temp_dir)
            
            if not processing_result:
                return Response({"error": "DXF processing failed"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Upload original file to S3
            input_file_key = f"uploads/{project_name}/{dxf_file.name}"
            with open(input_temp_path, 'rb') as file_obj:
                s3_client.upload_fileobj(file_obj, bucket_name, input_file_key)
            input_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{input_file_key}"

            # Upload processed file to S3
            output_file_key = f"processed/{project_name}/Final_output.dxf"
            with open(final_output_path, 'rb') as file_obj:
                s3_client.upload_fileobj(file_obj, bucket_name, output_file_key)
            output_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{output_file_key}"

            # Prepare plumbing processing if requested
            plumbing_result = None
            if process_plumbing:
                logger.info(f"Processing plumbing for project: {project_name}")
                try:
                    plumbing_output_filename = 'Final_Floor_Plan_with_Drainage_Pipes.dxf'
                    blocksDxf = os.path.join(settings.BASE_DIR, 'assets', 'SMH-Blocks.dxf')
                    plumbExcel = os.path.join(settings.BASE_DIR, 'assets', 'price_of_plumbing_material.xlsx')
                    material = complete_pipeline(final_output_path, output_temp_path, blocksDxf, plumbExcel)

                    # Upload plumbing result to S3
                    plumbing_file_key = f"processed/{project_name}/{plumbing_output_filename}"
                    with open(plumbing_output_filename, 'rb') as file_obj:
                        s3_client.upload_fileobj(file_obj, bucket_name, plumbing_file_key)
                    plumbing_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{plumbing_file_key}"
                    
                    boq_filename = "Plumbing_BOQ.xlsx"
                    main_plumbing(material)

                    boq_file_key = f"processed/{project_name}/{boq_filename}"
                    with open(boq_filename, 'rb') as boq_file:
                        s3_client.upload_fileobj(boq_file, bucket_name, boq_file_key)
                    boq_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{boq_file_key}"

                    plumbing_result = {
                        "output_file_url": plumbing_file_url,
                        "output_file_name": plumbing_output_filename,
                        "boq_file_url": boq_file_url,
                        "additional_result": {
                            "MaterialList": material
                        }
                    }
                    
                except Exception as e:
                    logger.error(f"Plumbing processing failed for project: {project_name}: {str(e)}")

            # Prepare electrical processing if requested
            electrical_result = None
            if process_electrical:
                logger.info(f"Processing electrical for project: {project_name}")
                try:
                    # Define paths for input DXF, source DXF, SW DXF, electrical DXF, and Excel file
                    input_dxf_file = input_temp_path
                    source_dxf_path = os.path.join(settings.BASE_DIR, 'assets', '15W.dxf')
                    sw_dxf_path = os.path.join(settings.BASE_DIR, 'assets', 'SW.dxf')
                    dxf_path_electrical = os.path.join(settings.BASE_DIR, 'Electrical.dxf')
                    excel_file_path = os.path.join(settings.BASE_DIR, 'assets', 'price_of_electrical.xlsx')
                    output_path_final = os.path.join(settings.BASE_DIR, 'Electrical_drawing.dxf')

                    # Call main_final with all required parameters
                    electrical_output_path, material_result = main_final(
                        input_dxf_file=input_dxf_file,
                        source_dxf_path=source_dxf_path,
                        sw_dxf_path=sw_dxf_path,
                        dxf_path_electrical=dxf_path_electrical,
                        excel_file_path=excel_file_path,
                        output_path_final=output_path_final,
                        offset_distance=24
                    )

                    # Upload electrical output to S3
                    electrical_file_key = f"processed/{project_name}/Final_Electrical_Plan.dxf"
                    with open(electrical_output_path, 'rb') as file_obj:
                        s3_client.upload_fileobj(file_obj, bucket_name, electrical_file_key)
                    electrical_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{electrical_file_key}"

                    # Generate BOQ for electrical
                    boq_filename = "Electrical_BOQ.xlsx"
                    main_electrical(material_result)

                    # Upload the BOQ file to S3
                    boq_file_key = f"processed/{project_name}/{boq_filename}"
                    with open(boq_filename, 'rb') as boq_file:
                        s3_client.upload_fileobj(boq_file, bucket_name, boq_file_key)
                    boq_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{boq_file_key}"

                    electrical_result = {
                        "output_file_url": electrical_file_url,
                        "output_file_name": os.path.basename(electrical_output_path),
                        "boq_file_url": boq_file_url,  # URL for the BOQ file
                        "material_result": material_result
                    }

                except Exception as e:
                    logger.error(f"Electrical processing failed for project: {project_name}: {str(e)}")
                    return Response(
                        {"error": "Electrical processing failed", "details": str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )

            structural_result = None
            if process_structural:
                logger.info(f"Processing structural components for project: {project_name}")
                try:
                    xl = os.path.join(settings.BASE_DIR, 'assets', 'price_of_material_new.xlsx')

                    # Call pipeline_main and verify if data is returned correctly
                    try:
                        column_info_df, beam_info_df = pipeline_main(final_output_path, output_temp_path)
                        logger.info("Successfully retrieved column and beam info.")
                    except Exception as e:
                        logger.error(f"Error in pipeline_main: {str(e)}")
                        return Response(
                            {"error": "Failed to process structural components", "details": str(e)},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )

                    # Upload the output DXF file to S3
                    output_file_key = f"processed/{project_name}/structural_output.dxf"
                    with open(output_temp_path, 'rb') as file_obj:
                        s3_client.upload_fileobj(file_obj, bucket_name, output_file_key)
                    output_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{output_file_key}"

                    # Generate and upload BOQ file for structural components
                    boq_filename = "Structural_BOQ.xlsx"
                    main_structure(column_info_df, beam_info_df, xl)
                    boq_file_key = f"processed/{project_name}/{boq_filename}"
                    with open(boq_filename, 'rb') as boq_file:
                        s3_client.upload_fileobj(boq_file, bucket_name, boq_file_key)
                    boq_file_url = f"https://{bucket_name}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{boq_file_key}"

                    # Prepare structural result response data
                    structural_result = {
                        "output_file_url": output_file_url,
                        "boq_file_url": boq_file_url,
                        "column_info": column_info_df.to_dict(),
                        "beam_info": beam_info_df.to_dict()
                    }

                except Exception as e:
                    logger.error(f"Structural processing failed: {str(e)}")
                    return Response(
                        {"error": "Structural processing failed", "details": str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
            # Create or update the project instance
            project, created = Project.objects.update_or_create(
                project_name=project_name,
                defaults={
                    'created_by': request.user,
                    'input_file_url': input_file_url,
                    'output_file_url': output_file_url,
                    'plumbing': plumbing_result,
                    'electrical': electrical_result,
                    'structural': structural_result,
                }
            )

            # Prepare response data
            project_data = ProjectSerializer(project).data
            response_data = {
                "project": {
                    **project_data,
                    "plumbing": plumbing_result if plumbing_result else {"output_file_url": None, "required": process_plumbing},
                    # "electrical": electrical_result if electrical_result else {"output_file_url": None, "required": process_electrical},
                    "electrical": electrical_result if electrical_result else {"output_file_url": None, "required": process_electrical},
                    "structural": structural_result if structural_result else {"output_file_url": None, "required": process_structural},
                    "layer_names": processing_result['Layer_names'],
                    "number_overlapped_lines": processing_result['Number_overlapped_lines'],
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
        # finally:
        #     # Clean up the temp directory
        #     if os.path.exists(temp_dir):
        #         shutil.rmtree(temp_dir)



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












