import os
import boto3
from django.db.models.signals import post_delete
from django.dispatch import receiver
from django.conf import settings
from .models import UserFile, MapFile

@receiver(post_delete, sender=UserFile)
def delete_userfile_s3_files(sender, instance, **kwargs):
    s3_client = boto3.client('s3',
                             aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    
    # Delete png_image, dxf_file, and gif_file from S3
    files_to_delete = [instance.png_image, instance.dxf_file, instance.gif_file]
    
    # Add files from 'info' if they exist
    if instance.info:
        files_to_delete.extend(instance.info.keys())
    
    for file_url in files_to_delete:
        if file_url:
            s3_key = file_url.replace(f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/", "")
            try:
                s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
                print(f"Deleted {s3_key} from S3")
            except Exception as e:
                print(f"Error deleting {s3_key} from S3: {str(e)}")

@receiver(post_delete, sender=MapFile)
def delete_mapfile_s3_files(sender, instance, **kwargs):
    s3_client = boto3.client('s3',
                             aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
    bucket_name = settings.AWS_STORAGE_BUCKET_NAME
    
    # Delete map file from S3
    s3_key = instance.map_path
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        print(f"Deleted {s3_key} from S3")
    except Exception as e:
        print(f"Error deleting {s3_key} from S3: {str(e)}")
