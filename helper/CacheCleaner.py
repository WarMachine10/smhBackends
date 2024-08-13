from django.conf import settings
from pathlib import Path
import glob,os
from loguru import logger

def delete_specific_files(folder_path, project_name):
    # Use a wildcard pattern to match any files containing the project_name
    files_to_delete = glob.glob(os.path.join(folder_path, f"*{project_name}*"))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting temporary file {file_path}: {e}")

def cleanup_temp_files(project_name):
        base_dir = Path(settings.BASE_DIR)
        temp_folders = ['dxfCache', 'gifCache', 'trimCache']
        floor_png=base_dir/'media'/'pngs/'
        for folder in temp_folders:
            folder_path = base_dir / 'Temp' / folder
            delete_specific_files(str(folder_path), project_name)
        delete_specific_files(str(floor_png),project_name)#for that rubbish floor files in pngs

