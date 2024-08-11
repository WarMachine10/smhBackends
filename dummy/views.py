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


class CreateProjectView(CreateAPIView):
    serializer_class = ProjectSerializer
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]
    
    @swagger_auto_schema(request_body=ProjectSerializer)
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        self.perform_create(serializer)
        try:
            response = self.run_external_script(serializer.validated_data, request.user)
            return response
        finally:
            project_name = serializer.validated_data.get('name', '')
            cleanup_temp_files(project_name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.files_to_delete = []
        
    def run_external_script(self, data, user):
        script_path = os.path.join(settings.BASE_DIR, 'dummy', 'PrototypeScript.py')
        python_executable='/home/ubuntu/SketchMyHome-Complete/venv/bin/python'
        if not os.path.exists(script_path):
            return self.error_response('Script path does not exist')
        result = run(['python', script_path, json.dumps(data)], 
                     stdout=PIPE, stderr=PIPE, text=True, cwd=os.path.join(settings.BASE_DIR, 'dummy'))
        if result.returncode != 0:
            logger.critical("External script execution failed")
            return self.error_response('External script execution failed', result.stderr)
        return self.process_output(result.stdout.strip(), user)

    def process_output(self, output, user):
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
        
        try:
            all_processed_files = []
            for info_data in info_data_list:
                processed_files = self.process_files(info_data, user)
                all_processed_files.extend(processed_files)
            
            return Response({
                'message': 'External script executed successfully and files processed',
                'processed_files': all_processed_files,
                'info_data': info_data_list,
            })
        except FileNotFoundError as e:
            return self.error_response('Error processing files', str(e))

    def process_files(self, info_data, user):
        processed_files = []
        for png_filename, floor_data in info_data.items():
            user_file = UserFile(
                user=user,
                info=floor_data
            )         
            png_saved, png_name = self.save_file(png_filename, user_file, 'png_image', subfolder='pngs') 
            dxf_filename = png_filename.replace('.png', '.dxf')
            dxf_saved, dxf_name = self.save_file(dxf_filename, user_file, 'dxf_file', subfolder='dxfs')        
            gif_filename = png_filename.replace('.png', '.gif')
            gif_saved, gif_name = self.save_file(gif_filename, user_file, 'gif_file', subfolder='gifs')         
            floor_files_saved = []
            floor_file_keys = list(floor_data.keys())
            for floor_file in floor_file_keys:
                floor_saved, floor_name = self.save_file(floor_file, user_file, 'floor_file', subfolder='pngs')
                if floor_saved:
                    floor_files_saved.append(floor_name)
                    # Delete the local floor file after successful S3 upload
                    local_floor_path = os.path.join(settings.MEDIA_ROOT, 'pngs', floor_file)
                    if os.path.exists(local_floor_path):
                        os.remove(local_floor_path)
                        logger.info(f"Deleted local floor file after S3 upload: {local_floor_path}")
            
            if png_saved or dxf_saved or gif_saved or floor_files_saved:
                user_file.save()
                logger.info(f"Saved UserFile: id={user_file.id}, png={png_name}, dxf={dxf_name}, gif={gif_name}, info={user_file.info}")
                processed_files.append({
                    'png': png_name,
                    'dxf': dxf_name,
                    'gif': gif_name,
                    'floors': floor_files_saved
                })
            else:
                logger.critical(f"Failed to save files for {png_filename}")
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

    def get(self, request):
        user_files = UserFile.objects.filter(user=request.user)
        serialized_data = []

        for user_file in user_files:
            file_data = {
                "id": user_file.id,
                "png_image": f"{user_file.png_image}",
                "dxf_file": f"{user_file.dxf_file}",
                "gif_file": f"{user_file.gif_file}" if user_file.gif_file else None,
                "info": {},
                "created_at": user_file.created_at,
                "user": user_file.user.id
            }
            # full_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.amazonaws.com/{key}"
            # Process the info dictionary
            for key, value in user_file.info.items():
                full_url = f"{key}"
                file_data["info"][full_url] = value
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
        boundary_coords = request.data.get('boundary_coords')

        if not latitude or not longitude or not boundary_coords:
            return Response({'error': 'Latitude, longitude, and boundary coordinates are required.'}, status=status.HTTP_400_BAD_REQUEST)
        if len(boundary_coords) != 4:
            return Response({'error': 'Exactly 4 sets of boundary coordinates are required.'}, status=status.HTTP_400_BAD_REQUEST)
        # Generate a unique filename
        unique_filename = f'map_{generate_short_uuid()}.html'
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            boundary_coords = [(float(coord['lat']), float(coord['lng'])) for coord in boundary_coords]
        except ValueError:
            return Response({'error': 'Invalid coordinate values.'}, status=status.HTTP_400_BAD_REQUEST)

        # Run the external script to generate the map and get soil data
        map_file_rel_path = main(unique_filename, latitude, longitude, boundary_coords)
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