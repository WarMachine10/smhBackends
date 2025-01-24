import threading
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from loguru import logger
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from rest_framework_simplejwt.authentication import JWTAuthentication
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
            # Create working drawing project
            working_drawing = WorkingDrawingProject.objects.create(
                subproject=subproject,
                drawing_type='electrical'
            )
        else:
            working_drawing = subproject.working_drawing
            
            if not working_drawing:
                # Create working drawing if somehow missing
                working_drawing = WorkingDrawingProject.objects.create(
                    subproject=subproject,
                    drawing_type='electrical'
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