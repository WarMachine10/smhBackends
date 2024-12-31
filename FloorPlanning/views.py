from django.shortcuts import get_object_or_404
from rest_framework import status,generics
from rest_framework.response import Response
from rest_framework.generics import CreateAPIView
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.conf import settings
from django.core.files.base import ContentFile
from subprocess import run, PIPE
import json, os, uuid,ast, shutil,boto3,re,glob,mimetypes,pandas as pd,base64
from loguru import logger
from pathlib import Path
from urllib.parse import quote
from botocore.exceptions import ClientError
from rest_framework.permissions import IsAuthenticated
from .serializers import *
from helper.SiteAnalyzer import main, soil_type
from helper.uuidGenerator import generate_short_uuid
from helper.CacheCleaner import cleanup_temp_files
from drf_yasg.utils import swagger_auto_schema
from loguru import logger
from project.models import *
from project.serializers import *
from FloorPlanning.models import *

class CreateProjectView(CreateAPIView):
    serializer_class = CreateFloorplanningProjectSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    @swagger_auto_schema(request_body=FloorplanningProject)
    def create(self, request, project_id, *args, **kwargs):
        try:
            # Get the parent project and verify ownership
            project = get_object_or_404(Project, id=project_id, user=request.user)
            
            # Create SubProject first
            subproject = SubProject.objects.create(
                project=project,
                type='floorplanning',
                state={}
            )
            logger.info(f"Created SubProject: {subproject.id}")
            
            # Pass subproject through serializer context
            serializer = self.get_serializer(
                data=request.data,
                context={'subproject': subproject}
            )
            
            if not serializer.is_valid():
                logger.error(f"Serializer validation failed: {serializer.errors}")
                subproject.delete()
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            # Save the FloorplanningProject
            try:
                floorplanning_project = serializer.save()
                logger.info(f"Created FloorplanningProject: {floorplanning_project.id}")
            except Exception as e:
                logger.error(f"Error creating FloorplanningProject: {str(e)}")
                subproject.delete()
                raise
            
            try:
                # Run the external script and process files
                script_result = self.run_external_script(serializer.validated_data, request.user)
                
                if isinstance(script_result, Response):
                    logger.error(f"External script failed: {script_result.data}")
                    floorplanning_project.delete()
                    return script_result
                
                # Process the output and create UserFiles
                info_data_list = []
                # Check if script_result is already a list of dictionaries
                if isinstance(script_result, list):
                    info_data_list = script_result
                else:
                    # If it's a string output, process it line by line
                    for line in script_result.split('\n'):
                        if line.startswith("INFO:"):
                            try:
                                dict_str = line.split("INFO:", 1)[1].strip()
                                info_data = ast.literal_eval(dict_str)
                                info_data_list.append(info_data)
                            except (IndexError, ValueError, SyntaxError) as e:
                                logger.error(f"Error parsing INFO line: {str(e)}")
                
                if not info_data_list:
                    logger.error("No valid INFO data found in script output")
                    floorplanning_project.delete()
                    raise ValueError("No valid data returned from external script")
                
                processed_files = []
                for info_data in info_data_list:
                    files = self.process_files(info_data, floorplanning_project)
                    if files:  # Only extend if files were actually processed
                        processed_files.extend(files)
                
                if not processed_files:
                    logger.error("No files were processed successfully")
                    floorplanning_project.delete()
                    raise ValueError("Failed to process any files")
                
                response_data = {
                    'message': 'Project created successfully',
                    'floorplanning_project': serializer.data,
                    'processed_files': processed_files
                }
                
                return Response(response_data, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                logger.error(f"Error in file processing: {str(e)}")
                floorplanning_project.delete()  # This will cascade delete the subproject
                raise
            
        except Exception as e:
            logger.error(f"Error in create: {str(e)}")
            return Response(
                {'error': f'Failed to create project: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        finally:
            project_name = request.data.get('project_name', '')
            cleanup_temp_files(project_name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_delete = []
        
    def run_external_script(self, data, user):
        script_path = os.path.join(settings.BASE_DIR, 'FloorPlanning', 'PrototypeScript.py')
        python_executable='/home/ubuntu/SketchMyHome-Complete/venv/bin/python'
        if not os.path.exists(script_path):
            return self.error_response('Script path does not exist')
        result = run(['python', script_path, json.dumps(data)], 
                     stdout=PIPE, stderr=PIPE, text=True, cwd=os.path.join(settings.BASE_DIR, 'FloorPlanning'))
        if result.returncode != 0:
            logger.critical("External script execution failed")
            return self.error_response('External script execution failed', result.stderr)
        return self.process_output(result.stdout.strip(), user)

    def process_output(self, output, user):
        if isinstance(output, list):
            return output
            
        info_data_list = []
        for line in output.split('\n'):
            if line.startswith("INFO:"):
                try:
                    dict_str = line.split("INFO:", 1)[1].strip()
                    logger.debug(f"Attempting to parse dict: {dict_str}")
                    info_data = ast.literal_eval(dict_str)
                    info_data_list.append(info_data)
                    logger.success(f"Successfully parsed INFO: {info_data}")
                except (IndexError, ValueError, SyntaxError) as e:
                    logger.error(f"Error when parsing INFO line: {line}. Error: {str(e)}")
        
        if not info_data_list:
            return self.error_response('No valid INFO data returned', output)
        
        return info_data_list

    def process_files(self, info_data, floorplanning_project):
        if not isinstance(floorplanning_project, FloorplanningProject):
            logger.error(f"Invalid floorplanning_project type: {type(floorplanning_project)}")
            logger.error(f"FloorplanningProject ID: {getattr(floorplanning_project, 'id', 'No ID')}")
            logger.error(f"FloorplanningProject attributes: {vars(floorplanning_project)}")
            raise ValueError("floorplanning_project must be an instance of FloorplanningProject")

        processed_files = []
        for png_filename, floor_data in info_data.items():
            try:
                # Create UserFile instance
                user_file = UserFile.objects.create(
                    floorplanning=floorplanning_project
                )
                logger.info(f"Created UserFile: {user_file.id} for FloorplanningProject: {floorplanning_project.id}")
                
                # Process the main files
                png_saved, png_url = self.save_file(png_filename, user_file, 'png_image', subfolder='pngs')
                dxf_filename = png_filename.replace('.png', '.dxf')
                dxf_saved, dxf_url = self.save_file(dxf_filename, user_file, 'dxf_file', subfolder='dxfs')
                gif_filename = png_filename.replace('.png', '.html')
                gif_saved, gif_url = self.save_file(gif_filename, user_file, 'gif_file', subfolder='gifs')
                
                # Update UserFile with URLs
                user_file.png_image = png_url
                user_file.dxf_file = dxf_url
                user_file.gif_file = gif_url
                
                # Process floor files and store S3 URLs in info
                updated_floor_data = {}
                floor_files_saved = []
                
                for floor_file in list(floor_data.keys()):
                    floor_saved, floor_url = self.save_file(floor_file, user_file, 'floor_file', subfolder='pngs')
                    if floor_saved:
                        # Store the S3 URL as the key instead of local path
                        updated_floor_data[floor_url] = floor_data[floor_file]
                        floor_files_saved.append(floor_url)
                        # Clean up local file
                        local_floor_path = os.path.join(settings.MEDIA_ROOT, 'pngs', floor_file)
                        if os.path.exists(local_floor_path):
                            os.remove(local_floor_path)
                
                # Update user_file with the new info containing S3 URLs
                user_file.info = updated_floor_data
                user_file.save()
                
                if png_saved or dxf_saved or gif_saved or floor_files_saved:
                    logger.info(f"Saved files for UserFile: {user_file.id}")
                    processed_files.append({
                        'png': png_url,
                        'dxf': dxf_url,
                        'gif': gif_url,
                        'floors': floor_files_saved
                    })
                else:
                    logger.error(f"No files were saved for {png_filename}")
                    user_file.delete()
                    
            except Exception as e:
                logger.error(f"Error processing file {png_filename}: {str(e)}")
                continue
                
        return processed_files

    def save_file(self, filename, user_file, file_type, subfolder):
        if not filename:
            return False, None    
        source_path = os.path.join(settings.MEDIA_ROOT, subfolder, filename)       
        if not os.path.exists(source_path):
            logger.warning(f"File not found: {source_path}")
            return False, None
        try:
            short_id = generate_short_uuid()
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{short_id}{ext}"
            s3_key = f"media/{subfolder}/{unique_filename}"
            content_type, _ = mimetypes.guess_type(filename)
            if content_type is None:
                content_type = 'application/octet-stream'
            extra_args = {
                'ContentType': content_type,
                'ACL': 'public-read'
            }
            if ext.lower() in ['.png', '.gif']:
                extra_args['ContentDisposition'] = 'inline'
            elif ext.lower() == '.html':
                extra_args['ContentType'] = 'text/html'
                extra_args['ContentDisposition'] = 'inline'    
            elif ext.lower() == '.dxf':
                extra_args['ContentDisposition'] = f'attachment; filename="{quote(unique_filename)}"'
            s3_client = settings.S3_CLIENT 
            with open(source_path, 'rb') as file:
                s3_client.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, s3_key, ExtraArgs=extra_args)
            s3_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"  
            if file_type == 'floor_file':
                if user_file.info is None:
                    user_file.info = {}
                user_file.info[s3_url] = user_file.info.pop(filename, {})
                user_file.save(update_fields=['info'])
            else:
                setattr(user_file, file_type, s3_url)        
            logger.success(f"Successfully saved {file_type} to S3: {s3_url}") 
            # Delete the local file after successful S3 upload
            os.remove(source_path)
            logger.info(f"Deleted local file after S3 upload: {source_path}")     
            return True, s3_url
        except ClientError as e:
            logger.error(f"Error uploading {file_type} {filename} to S3: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"Error processing {file_type} {filename}: {str(e)}")
            return False, None

    def error_response(self, message, details=None):
        response = {'message': message}
        if details:
            response['details'] = details
        return Response(response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class UserFileListView(APIView):
    permission_classes = [IsAuthenticated]
    authentication_classes = [JWTAuthentication]

    def get(self, request, project_id=None):
        if not project_id:
            return Response(
                {"error": "Project ID is required"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Get the FloorplanningProject through SubProject
        try:
            subproject = SubProject.objects.get(
                project_id=project_id,
                project__user=request.user,
                type='floorplanning'
            )
            floorplanning_project = FloorplanningProject.objects.get(
                subproject=subproject
            )
        except (SubProject.DoesNotExist, FloorplanningProject.DoesNotExist):
            return Response(
                {"error": "FloorplanningProject not found"},
                status=status.HTTP_404_NOT_FOUND
            )
            
        user_files = UserFile.objects.filter(floorplanning=floorplanning_project)
        serialized_data = []

        for user_file in user_files:
            file_data = {
                "id": user_file.id,
                "png_image": user_file.png_image,
                "dxf_file": user_file.dxf_file,
                "gif_file": user_file.gif_file,
                "info": {},
                "created_at": user_file.created_at,
            }
            
            for key, value in user_file.info.items() if user_file.info else {}.items():
                file_data["info"][key] = value
                
            serialized_data.append(file_data)
            
        return Response(serialized_data)

#SiteMap Analysis Code
class GenerateMapAndSoilDataView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(request_body=MapFileSerializer)
    def post(self, request, *args, **kwargs):
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')
        front_of_house = request.data.get('front_of_house')
        boundary_coords = request.data.get('boundary_coords')
        if not latitude or not longitude or not front_of_house:
            return Response({'error': 'Latitude, longitude, and front of house direction are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Generate a unique filename
        unique_filename = f'map_{generate_short_uuid()}.html'
        
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            boundary_coords = [(float(coord['lat']), float(coord['lng'])) for coord in boundary_coords]
        except ValueError:
            return Response({'error': 'Invalid coordinate values.'}, status=status.HTTP_400_BAD_REQUEST)

        # Set the GIF path
        gif_path = settings.BASE_DIR / 'assets' / 'GIF.gif'

        # Run the external script to generate the map and get soil data
        map_file_rel_path = main(unique_filename, front_of_house, latitude, longitude, boundary_coords, str(gif_path))
        if not map_file_rel_path:
            logger.error("Map Generation Task Failed")
            return Response({'error': 'Failed to generate map.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


        # Define the local file path
        local_file_path = os.path.join(settings.BASE_DIR, 'media', map_file_rel_path)

        # Upload the file to S3
        s3_key = f'media/maps/{unique_filename}'
        s3_url = self.upload_to_s3(local_file_path, s3_key)

        if not s3_url:
            return Response({'error': 'Failed to upload map to S3.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save only the path in the database
        map_file = MapFile.objects.create(user=request.user, map_path=s3_key)
        map_file_serializer = MapFileSerializer(map_file)
        
        try:
            os.remove(local_file_path)
            print(f"Successfully deleted local file: {local_file_path}")
        except OSError as e:
            print(f"Error deleting local file {local_file_path}: {e}")
        
        base_dir = settings.BASE_DIR / 'assets'
        excel_path = base_dir / 'soil_type.xlsx'    

        # Fetch and save the soil data
        soil_data = soil_type(pd.read_excel(excel_path), latitude, longitude).iloc[0]
        soil_data_instance = SoilData.objects.create(
            user=request.user,
            soil_type=soil_data['Soil Type'],
            ground_water_depth=soil_data['Ground Water Depth'],
            foundation_type=soil_data['Foundation Type']
        )
        soil_data_serializer = SoilDataSerializer(soil_data_instance)
        
        # Return the serialized data
        response_data = {
            'map_file': map_file_serializer.data,
            'soil_data': soil_data_serializer.data
        }
        logger.info(f"Response data: {response_data}")
        return Response(response_data, status=status.HTTP_201_CREATED)

    def upload_to_s3(self, file_path, s3_key):
        s3_client = settings.S3_CLIENT
        try:
            with open(file_path, 'rb') as file:
                content_type = 'text/html'  # Since this is an HTML file
                s3_client.upload_fileobj(file, settings.AWS_STORAGE_BUCKET_NAME, s3_key,
                                        ExtraArgs={'ContentType': content_type})
            
            return s3_key  # Return only the S3 key, not the full URL
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
        




        