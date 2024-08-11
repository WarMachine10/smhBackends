
import os
import sys
import glob
from pathlib import Path
from django.conf import settings
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SKM.settings')

def delete_specific_files(folder_path, file_pattern):
    files_to_delete = glob.glob(os.path.join(folder_path, file_pattern))
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


# Targetting the clutter


folder_dxf = os.path.join(settings.BASE_DIR,'Temp','dxfCache/')
patternDxf = '*.dxf'


folder_png = os.path.join(settings.BASE_DIR,'Temp','gifCache/')
patternPng = '*.png'


folder_trim = os.path.join(settings.BASE_DIR,'Temp','trimCache/')
patternTrim = '*.dxf'

folder_floor = os.path.join(settings.BASE_DIR,'media','pngs/')
patternFloor = '*.png'
# Hitting the function
delete_specific_files(folder_dxf, patternDxf)
delete_specific_files(folder_png, patternPng)
delete_specific_files(folder_trim, patternTrim)
delete_specific_files(folder_floor, patternFloor)