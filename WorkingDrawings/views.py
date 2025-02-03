import threading
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from loguru import logger
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from rest_framework_simplejwt.authentication import JWTAuthentication

from WorkingDrawings.Scripts.WorkingDrawings.Plumbing.PlumbingBOQs import main_plumbing_boq
from WorkingDrawings.Scripts.WorkingDrawings.Structural.Struct import pipeline_main_final
from .models import *
import nest_asyncio 
nest_asyncio.apply()
import tempfile
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import boto3,shutil
from loguru import logger
import tempfile,re
from io import StringIO
from pathlib import Path
from django.conf import settings,Settings
import tempfile
from .serializers import *
from WorkingDrawings.Scripts.WorkingDrawings.Electrical.Fixtures import main_process
from WorkingDrawings.Scripts.WorkingDrawings.Electrical.Wiring import main_process_wiring
import pandas as pd
import numpy as np
import json
import ezdxf
import os
import asyncio
from WorkingDrawings.Scripts.WorkingDrawings.Plumbing.WaterSupply import main_final_water
from WorkingDrawings.Scripts.WorkingDrawings.Plumbing.PlumbingComplete import main_final_plumbing_complete
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from project.models import SubProject
from .models import ElectricalFixtures
from .serializers import ElectricalFixturesSerializer
import asyncio
from rest_framework.exceptions import ValidationError   
from django.db import transaction
from django.conf import settings
from io import StringIO

TEMP_FOLDER = os.path.join(settings.BASE_DIR,'Temp','crap')  # Replace with a real folder path, e.g., '/tmp/dxf_processing'

# Set up a StringIO log capture stream
log_stream = StringIO()
# logger.basicConfig(stream=log_stream, level=logger.INFO)
# working_drawings/views.py

class BaseWorkingDrawingView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def validate_dxf_file(self, file):
        """Validate uploaded DXF file"""
        if not file:
            raise ValidationError('Input file is required.')
        
        if not file.name.lower().endswith('.dxf'):
            raise ValidationError('Only DXF files are allowed.')
        
        return file

    def get_working_drawing(self, project_id, user):
        """Get or create working drawing subproject"""
        project = get_object_or_404(Project, id=project_id, user=user)
        
        # First try to find existing working drawing subproject
        subproject = SubProject.objects.filter(
            project=project,
            type='working_drawing',
            state__drawing_type='electrical'  # Only get electrical ones
        ).first()
        
        if not subproject:
            # Create new working drawing subproject if none exists
            subproject = SubProject.objects.create(
                project=project,
                type='working_drawing',
                state={
                    'drawing_type': 'electrical',
                    'status': 'in_progress',
                    'sections': {
                        'electrical': {'status': 'pending', 'last_updated': None},
                        'plumbing': {'status': 'pending', 'last_updated': None},
                        'structural': {'status': 'pending', 'last_updated': None}
                    }
                }
            )
            # Create working drawing project without drawing_type
            working_drawing = WorkingDrawingProject.objects.create(
                subproject=subproject  # Remove drawing_type parameter
            )
        else:
            working_drawing = getattr(subproject, 'working_drawing', None)
            
            if not working_drawing:
                # Create working drawing if missing, without drawing_type
                working_drawing = WorkingDrawingProject.objects.create(
                    subproject=subproject  # Remove drawing_type parameter
                )
        
        return working_drawing

class ElectricalFixturesView(BaseWorkingDrawingView):
    def process_fixtures_async(self, instance):
        """Process fixtures in background"""
        input_temp_path = None
        output_temp_path = None
        
        try:
            instance.status = 'processing'
            instance.save()
            os.makedirs(TEMP_FOLDER, exist_ok=True)

            # Setup paths
            input_temp_path = os.path.join(TEMP_FOLDER, f'input_{instance.id}.dxf')
            output_filename = f"electrical_drawing_{instance.id}.dxf"
            output_temp_path = os.path.join(TEMP_FOLDER, output_filename)

            # Save input file
            with open(input_temp_path, 'wb') as f:
                f.write(instance.input_file.read())

            try:
                # Process DXF
                main_process(
                    input_file=input_temp_path,
                    light_dxf=os.path.join(settings.BASE_DIR, 'assets', 'Ceiling_Lights.dxf'),
                    switch_dxf=os.path.join(settings.BASE_DIR, 'assets', 'M_SW.dxf'),
                    fan_dxf=os.path.join(settings.BASE_DIR, 'assets', 'Fan_C.dxf'),
                    wall_light=os.path.join(settings.BASE_DIR, 'assets', 'Wall_Lights.dxf'),
                    ac_dxf=os.path.join(settings.BASE_DIR, 'assets', 'AC.dxf'),
                    MBD_dxf=os.path.join(settings.BASE_DIR, 'assets', 'MBD.dxf'),
                    EvSwitch_dxf=os.path.join(settings.BASE_DIR, 'assets', 'EvSwitch.dxf'),
                    output_file_final=output_temp_path,
                    user_input="yes"
                )

                # Save output file
                with open(output_temp_path, 'rb') as f:
                    output_path = f'electrical/fixtures/outputs/{output_filename}'
                    instance.output_file.save(output_path, ContentFile(f.read()))

                instance.status = 'completed'
                instance.save()

            except Exception as e:
                raise RuntimeError(f"DXF processing error: {str(e)}")

        except Exception as e:
            instance.status = 'failed'
            instance.save()
            logger.error(f"Fixture processing failed: {str(e)}")
            raise

        finally:
            # Cleanup temp files
            for temp_file in [input_temp_path, output_temp_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Cleanup error for {temp_file}: {str(e)}")

    @transaction.atomic
    def post(self, request, project_id):
        """Create new electrical fixtures"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        try:
            input_file = self.validate_dxf_file(request.FILES.get('input_file'))
        except ValidationError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Calculate size from input file
        size = input_file.size

        data = {
            'subproject': working_drawing.subproject.id,
            'user': request.user.id,
            'input_file': input_file,
            'size': size  # Add size to the data
        }

        serializer = ElectricalFixturesSerializer(data=data, context={'request': request})
        if serializer.is_valid():
            instance = serializer.save(user=request.user)
            
            # Process in background
            thread = threading.Thread(
                target=self.process_fixtures_async,
                args=(instance,),
                name=f"ProcessFixtures-{instance.id}"
            )
            thread.daemon = True
            thread.start()
            
            return Response({
                'id': instance.id,
                'status': 'processing',
                'message': 'Processing started. Check status for completion.',
                'working_drawing': {
                    'id': working_drawing.id,
                },
                'files': {
                    'input': instance.input_file_url,
                    'output': None
                }
            }, status=status.HTTP_201_CREATED)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, project_id, fixture_id=None):
        """Get fixture(s) details"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        if fixture_id:
            fixture = get_object_or_404(
                ElectricalFixtures, 
                id=fixture_id, 
                subproject=working_drawing.subproject
            )
            serializer = ElectricalFixturesSerializer(fixture)
            return Response(serializer.data)
        
        fixtures = ElectricalFixtures.objects.filter(
            subproject=working_drawing.subproject
        ).order_by('-created_at')
        
        serializer = ElectricalFixturesSerializer(fixtures, many=True)
        return Response(serializer.data)

class ElectricalWiringView(BaseWorkingDrawingView):
    def process_wiring_async(self, instance):
        """Process wiring in background"""
        input_temp_path = None
        output_temp_path = None
        
        try:
            instance.status = 'processing'
            instance.save()
            os.makedirs(TEMP_FOLDER, exist_ok=True)

            # Setup paths
            input_temp_path = os.path.join(TEMP_FOLDER, f'input_{instance.id}.dxf')
            output_filename = f"electrical_wiring_{instance.id}.dxf"
            output_temp_path = os.path.join(TEMP_FOLDER, output_filename)

            # Save input file
            with open(input_temp_path, 'wb') as f:
                f.write(instance.input_file.read())

            try:
                # Process DXF
                main_process_wiring(
                    input_file=input_temp_path,
                    output_file_final=output_temp_path
                )

                # Save output file
                with open(output_temp_path, 'rb') as f:
                    output_path = f'electrical/wiring/outputs/{output_filename}'
                    instance.output_file.save(output_path, ContentFile(f.read()))

                instance.status = 'completed'
                instance.save()

            except Exception as e:
                raise RuntimeError(f"DXF processing error: {str(e)}")

        except Exception as e:
            instance.status = 'failed'
            instance.save()
            logger.error(f"Wiring processing failed: {str(e)}")
            raise

        finally:
            # Cleanup temp files
            for temp_file in [input_temp_path, output_temp_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Cleanup error for {temp_file}: {str(e)}")

    @transaction.atomic
    def post(self, request, project_id):
        """Create new electrical wiring"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        try:
            input_file = self.validate_dxf_file(request.FILES.get('input_file'))
        except ValidationError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        data = {
            'subproject': working_drawing.subproject.id,
            'user': request.user.id,
            'input_file': input_file,
            'size': input_file.size
        }

        serializer = ElectricalWiringSerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            
            # Process in background
            thread = threading.Thread(
                target=self.process_wiring_async,
                args=(instance,),
                name=f"ProcessWiring-{instance.id}"
            )
            thread.daemon = True
            thread.start()
            
            return Response({
                'id': instance.id,
                'status': 'processing',
                'message': 'Processing started. Check status for completion.',
                'working_drawing': {
                    'id': working_drawing.id,
                },
                'files': {
                    'input': instance.input_file_url,
                    'output': None
                }
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, project_id, wiring_id=None):
        """Get wiring details"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        if wiring_id:
            wiring = get_object_or_404(
                ElectricalWiring, 
                id=wiring_id, 
                subproject=working_drawing.subproject
            )
            serializer = ElectricalWiringSerializer(wiring)
            return Response(serializer.data)
        
        wiring_list = ElectricalWiring.objects.filter(
            subproject=working_drawing.subproject
        ).order_by('-created_at')
        
        serializer = ElectricalWiringSerializer(wiring_list, many=True)
        return Response(serializer.data)

class WorkingDrawingView(BaseWorkingDrawingView):
    def get(self, request, project_id):
        """Get complete working drawing details"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        serializer = WorkingDrawingProjectSerializer(working_drawing)
        return Response(serializer.data)
    
class WaterSupplyView(BaseWorkingDrawingView):
    def process_water_supply_async(self, instance):
        """Process water supply in background"""
        input_temp_path = None
        output_temp_path = None
        
        try:
            # Debug prints to verify asset files
            mwsp_block_path = os.path.join(settings.BASE_DIR, 'assets', 'MWSP Pipe block.dxf')
            inlet_block_path = os.path.join(settings.BASE_DIR, 'assets', 'Inlet_pipe.dxf')
            outlet_block_path = os.path.join(settings.BASE_DIR, 'assets', 'Outlet_pipe.dxf')
            
            print(f"MWSP block exists: {os.path.exists(mwsp_block_path)}")
            print(f"Inlet block exists: {os.path.exists(inlet_block_path)}")
            print(f"Outlet block exists: {os.path.exists(outlet_block_path)}")

            instance.status = 'processing'
            instance.save()
            os.makedirs(TEMP_FOLDER, exist_ok=True)

            input_temp_path = os.path.join(TEMP_FOLDER, f'input_{instance.id}.dxf')
            output_filename = f"water_supply_{instance.id}.dxf"
            output_temp_path = os.path.join(TEMP_FOLDER, output_filename)

            # Debug print input file
            print(f"Input file size: {instance.input_file.size}")

            with open(input_temp_path, 'wb') as f:
                f.write(instance.input_file.read())

            try:
                # Process DXF with more debug info
                print("Starting DXF processing...")
                main_final_water(
                    input_dxf1=input_temp_path,
                    transparency_percent=75,
                    mwsp_block=mwsp_block_path,
                    inlet_block=inlet_block_path,
                    outlet_block=outlet_block_path,
                    final_output=output_temp_path
                )
                print("DXF processing completed")

                # Verify output file was created
                print(f"Output file exists: {os.path.exists(output_temp_path)}")
                print(f"Output file size: {os.path.getsize(output_temp_path) if os.path.exists(output_temp_path) else 'N/A'}")

                with open(output_temp_path, 'rb') as f:
                    output_path = f'plumbing/watersupply/outputs/{output_filename}'
                    instance.output_file.save(output_path, ContentFile(f.read()))

                instance.status = 'completed'
                instance.save()

            except Exception as e:
                print(f"Processing error: {str(e)}")  # Debug print the actual error
                raise RuntimeError(f"DXF processing error: {str(e)}")

        except Exception as e:
            print(f"Overall error: {str(e)}")  # Debug print for outer exception
            instance.status = 'failed'
            instance.save()
            logger.error(f"Water supply processing failed: {str(e)}")
            raise

        finally:
            # Cleanup temp files
            for temp_file in [input_temp_path, output_temp_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Cleanup error for {temp_file}: {str(e)}")

    @transaction.atomic
    def post(self, request, project_id):
        """Create new water supply processing"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        try:
            input_file = self.validate_dxf_file(request.FILES.get('input_file'))
        except ValidationError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        data = {
            'subproject': working_drawing.subproject.id,
            'user': request.user.id,
            'input_file': input_file,
            'size': input_file.size
        }

        serializer = WaterSupplySerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            
            thread = threading.Thread(
                target=self.process_water_supply_async,
                args=(instance,),
                name=f"ProcessWaterSupply-{instance.id}"
            )
            thread.daemon = True
            thread.start()
            
            return Response({
                'id': instance.id,
                'status': 'processing',
                'message': 'Processing started. Check status for completion.',
                'working_drawing': {
                    'id': working_drawing.id,
                },
                'files': {
                    'input': instance.input_file_url,
                    'output': None
                }
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, project_id, supply_id=None):
        """Get water supply details"""
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        if supply_id:
            supply = get_object_or_404(
                WaterSupply, 
                id=supply_id, 
                subproject=working_drawing.subproject
            )
            serializer = WaterSupplySerializer(supply)
            return Response(serializer.data)
        
        supplies = WaterSupply.objects.filter(
            subproject=working_drawing.subproject
        ).order_by('-created_at')
        
        serializer = WaterSupplySerializer(supplies, many=True)
        return Response(serializer.data)

class PlumbingCompleteView(BaseWorkingDrawingView):
    def process_complete_async(self, instance):
        input_temp_path = None
        output_temp_path = None
        boq_temp_path = None
        
        try:
            instance.status = 'processing'
            instance.save()
            os.makedirs(TEMP_FOLDER, exist_ok=True)

            input_temp_path = os.path.join(TEMP_FOLDER, f'input_{instance.id}.dxf')
            output_filename = f"plumbing_complete_{instance.id}.dxf"
            output_temp_path = os.path.join(TEMP_FOLDER, output_filename)
            boq_filename = f"plumbing_boq_{instance.id}.xlsx"
            boq_temp_path = os.path.join(TEMP_FOLDER, boq_filename)

            with open(input_temp_path, 'wb') as f:
                f.write(instance.input_file.read())

            try:
                # Process DXF
                main_final_plumbing_complete(
                    input_dxf1=input_temp_path,
                    user_input1=instance.input1_option,
                    user_input2=instance.input2_option,
                    block_file_IC_and_GT=os.path.join(settings.BASE_DIR, 'assets', 'IC & GT_Blocks.dxf'),
                    block_file_FT=os.path.join(settings.BASE_DIR, 'assets', 'Floor_Trap.dxf'),
                    block_file_WPDT=os.path.join(settings.BASE_DIR, 'assets', 'Waste_Pipe.dxf'),
                    block_dxf_path=os.path.join(settings.BASE_DIR, 'assets', 'Rain_Water_Pipe.dxf'),
                    csv_file_path=os.path.join(settings.BASE_DIR, 'assets', 'indian_residential_layers.csv'),
                    annual_rainfall_mm=800,
                    output_dxf=output_temp_path,
                    block_mapping={
                        "Inlet_Pipe": ("Inlet Pipe", 10),
                        "Outlet_Pipe": ("Outlet Pipe",2),
                        "MWSP": ("Main Water\nSupply Pipe", 0.1),
                        "WPDT": ("Waste Pipe\nDown Take", 0.1),
                        "SPDT": ("Soil Pipe\nDown Take", 7),
                        "RPDT": ("Rain Water\nDown Take", 0.1),
                        "IC": ("Inspection Chamber", 0.8),
                        "GT": ("Gully Trap", 0.8),
                        "FT": ("Floor Trap", 0.05)
                    }
                )

                # Generate BOQ
                main_plumbing_boq(
                    dxf_file=output_temp_path,
                    excel_file_path=os.path.join(settings.BASE_DIR, 'assets', 'BOQs_Plumbing.xlsx'),
                    output_file=boq_temp_path
                )

                # Save DXF output
                with open(output_temp_path, 'rb') as f:
                    output_path = f'plumbing/complete/outputs/{output_filename}'
                    instance.output_file.save(output_path, ContentFile(f.read()))

                # Save BOQ output
                with open(boq_temp_path, 'rb') as f:
                    boq_path = f'plumbing/complete/boq/{boq_filename}'
                    instance.boq_output_file.save(boq_path, ContentFile(f.read()))

                instance.status = 'completed'
                instance.save()

            except Exception as e:
                raise RuntimeError(f"Processing error: {str(e)}")

        except Exception as e:
            instance.status = 'failed'
            instance.save()
            logger.error(f"Plumbing complete processing failed: {str(e)}")
            raise

        finally:
            for temp_file in [input_temp_path, output_temp_path, boq_temp_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Cleanup error for {temp_file}: {str(e)}")

    @transaction.atomic
    def post(self, request, project_id):
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        try:
            input_file = self.validate_dxf_file(request.FILES.get('input_file'))
        except ValidationError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        data = {
            'subproject': working_drawing.subproject.id,
            'user': request.user.id,
            'input_file': input_file,
            'size': input_file.size,
            'input1_option': request.data.get('input1_option', 'yes'),
            'input2_option': request.data.get('input2_option', 'yes')
        }

        serializer = PlumbingCompleteSerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            
            thread = threading.Thread(
                target=self.process_complete_async,
                args=(instance,),
                name=f"ProcessPlumbingComplete-{instance.id}"
            )
            thread.daemon = True
            thread.start()
            
            return Response({
                'id': instance.id,
                'status': 'processing',
                'message': 'Processing started. Check status for completion.',
                'working_drawing': {
                    'id': working_drawing.id,
                },
                'files': {
                    'input': instance.input_file_url,
                    'output': None
                }
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, project_id, complete_id=None):
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        if complete_id:
            complete = get_object_or_404(
                PlumbingComplete, 
                id=complete_id, 
                subproject=working_drawing.subproject
            )
            serializer = PlumbingCompleteSerializer(complete)
            return Response(serializer.data)
        
        completes = PlumbingComplete.objects.filter(
            subproject=working_drawing.subproject
        ).order_by('-created_at')
        
        serializer = PlumbingCompleteSerializer(completes, many=True)
        return Response(serializer.data)

class StructuralMainView(BaseWorkingDrawingView):
    def process_structural_async(self, instance):
        input_temp_path = None
        output_temp_path = None
        try:
            instance.status = 'processing'
            instance.save()
            os.makedirs(TEMP_FOLDER, exist_ok=True)

            # Setup temporary file paths
            input_temp_path = os.path.join(TEMP_FOLDER, f'input_{instance.id}.dxf')
            output_filename = f"structural_main_{instance.id}.dxf"
            output_temp_path = os.path.join(TEMP_FOLDER, output_filename)

            # Write input file to temp location
            with open(input_temp_path, 'wb') as f:
                f.write(instance.input_file.read())

            try:
                # Process DXF and get dataframes
                column_info_df, beam_info_df = pipeline_main_final(
                    input_temp_path,
                    output_temp_path
                )

                # Save output DXF
                with open(output_temp_path, 'rb') as f:
                    output_path = f'structural/main/outputs/{output_filename}'
                    instance.output_file.save(output_path, ContentFile(f.read()))

                # Convert DataFrames to JSON-compatible format and save
                instance.column_info = column_info_df.to_dict(orient='records')
                instance.beam_info = beam_info_df.to_dict(orient='records')
                
                instance.status = 'completed'
                instance.save()

            except Exception as e:
                raise RuntimeError(f"Processing error: {str(e)}")

        except Exception as e:
            instance.status = 'failed'
            instance.save()
            logger.error(f"Structural main processing failed: {str(e)}")
            raise

        finally:
            # Cleanup temporary files
            for temp_file in [input_temp_path, output_temp_path]:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except OSError as e:
                        logger.error(f"Cleanup error for {temp_file}: {str(e)}")

    @transaction.atomic
    def post(self, request, project_id):
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        try:
            input_file = self.validate_dxf_file(request.FILES.get('input_file'))
        except ValidationError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        data = {
            'subproject': working_drawing.subproject.id,
            'user': request.user.id,
            'input_file': input_file,
            'size': input_file.size
        }

        serializer = StructuralMainSerializer(data=data)
        if serializer.is_valid():
            instance = serializer.save()
            
            thread = threading.Thread(
                target=self.process_structural_async,
                args=(instance,),
                name=f"ProcessStructuralMain-{instance.id}"
            )
            thread.daemon = True
            thread.start()
            
            return Response({
                'id': instance.id,
                'status': 'processing',
                'message': 'Processing started. Check status for completion.',
                'working_drawing': {
                    'id': working_drawing.id,
                },
                'data': {
                    'input_file': instance.input_file_url,
                    'output_file': None,
                    'column_info': None,
                    'beam_info': None
                }
            }, status=status.HTTP_201_CREATED)
            
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, project_id, main_id=None):
        working_drawing = self.get_working_drawing(project_id, request.user)
        
        if main_id:
            main = get_object_or_404(
                StructuralMain, 
                id=main_id, 
                subproject=working_drawing.subproject
            )
            serializer = StructuralMainSerializer(main)
            return Response(serializer.data)
        
        mains = StructuralMain.objects.filter(
            subproject=working_drawing.subproject
        ).order_by('-created_at')
        
        serializer = StructuralMainSerializer(mains, many=True)
        return Response(serializer.data)