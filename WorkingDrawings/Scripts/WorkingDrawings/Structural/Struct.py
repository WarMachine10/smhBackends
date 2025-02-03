
import ezdxf
import math
import pandas as pd
import numpy as np
from typing import Tuple
import warnings
from collections import Counter
from math import sqrt
import logging
import re
from typing import List, Optional
from sklearn.decomposition import PCA
from collections import defaultdict

# Suppress SettingWithCopyWarning from pandas
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging to only show warnings and above
logging.getLogger('ezdxf').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

# %% [markdown]
# # Step 1- Adjust DXF to zero

# %%
def adjust_dxf_coordinates_to00(filename, output_filename):
    # Read the DXF file
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        logging.error(f"Cannot open file: {filename}")
        return
    except ezdxf.DXFStructureError:
        logging.error(f"Invalid or corrupt DXF file: {filename}")
        return
    
    msp = doc.modelspace()

    min_x, min_y = float('inf'), float('inf')

    # Define a function to get the minimum coordinates for each entity
    def update_min_coords(x, y):
        nonlocal min_x, min_y
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    # First pass: Find the minimum x and y coordinates
    for entity in msp:
        try:
            if entity.dxftype() == 'CIRCLE' or entity.dxftype() == 'ARC':
                center = entity.dxf.center
                update_min_coords(center.x, center.y)

            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                update_min_coords(start.x, start.y)
                update_min_coords(end.x, end.y)

            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                insert = entity.dxf.insert
                update_min_coords(insert.x, insert.y)

            elif entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points()
                for point in points:
                    update_min_coords(point[0], point[1])

            elif entity.dxftype() == 'POLYLINE':
                vertices = [v.dxf.location for v in entity.vertices]
                for vertex in vertices:
                    update_min_coords(vertex[0], vertex[1])

            elif entity.dxftype() == 'INSERT':
                insert = entity.dxf.insert
                update_min_coords(insert.x, insert.y)

        except AttributeError:
            # Skip any malformed entities that are missing coordinates
            continue

    # Calculate the offsets
    if min_x == float('inf') or min_y == float('inf'):
        return
    
    offset_x = -min_x
    offset_y = -min_y

    # Second pass: Update the coordinates of the entities
    for entity in msp:
        try:
            if entity.dxftype() == 'CIRCLE' or entity.dxftype() == 'ARC':
                entity.dxf.center = (
                    entity.dxf.center.x + offset_x,
                    entity.dxf.center.y + offset_y,
                    entity.dxf.center.z
                )

            elif entity.dxftype() == 'LINE':
                entity.dxf.start = (
                    entity.dxf.start.x + offset_x,
                    entity.dxf.start.y + offset_y,
                    entity.dxf.start.z
                )
                entity.dxf.end = (
                    entity.dxf.end.x + offset_x,
                    entity.dxf.end.y + offset_y,
                    entity.dxf.end.z
                )

            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                entity.dxf.insert = (
                    entity.dxf.insert.x + offset_x,
                    entity.dxf.insert.y + offset_y,
                    entity.dxf.insert.z
                )

            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0] + offset_x, p[1] + offset_y, *p[2:]) for p in entity.get_points()]
                entity.set_points(points)

            elif entity.dxftype() == 'POLYLINE':
                for vertex in entity.vertices:
                    vertex.dxf.location = (
                        vertex.dxf.location[0] + offset_x,
                        vertex.dxf.location[1] + offset_y,
                        vertex.dxf.location[2]
                    )
            
            elif entity.dxftype() == 'INSERT':
                entity.dxf.insert = (
                    entity.dxf.insert.x + offset_x,
                    entity.dxf.insert.y + offset_y,
                    entity.dxf.insert.z
                )

        except AttributeError:
            # Skip any entities that cannot be updated
            continue

    # Save the modified DXF file
    try:
        doc.saveas(output_filename)
    except IOError:
        logging.error(f"Failed to save file: {output_filename}")

# %% [markdown]
# # Step 2- Setting transparency

# %%
def set_transparency_for_all_entities(dxf_file_path, output_file_path, transparency_percent):
    """
    Sets the transparency for all entities in a DXF file.

    Args:
        dxf_file_path (str): Path to the input DXF file.
        output_file_path (str): Path to save the output DXF file.
        transparency_percent (float): Transparency value as a percentage (0 to 100).
    """
    try:
        # Convert transparency percentage to DXF transparency value (0 to 1)
        transparency_value = transparency_percent / 100.0

        # Read the DXF file
        doc = ezdxf.readfile(dxf_file_path)

        # Iterate through all modelspace entities
        for entity in doc.modelspace():
            try:
                # Set transparency to the specified value
                entity.transparency = transparency_value  # Transparency value must be between 0 and 1
            except AttributeError:
                # Some entities may not support transparency, skip them
                continue

        # Save the updated DXF file
        doc.saveas(output_file_path)

    except Exception as e:
        print(f"An error occurred: {e}")

# %% [markdown]
# # Step 3 - DXF to dataframe 

# %%
def calculate_length(start, end):
    """Calculates the 3D length between two points with error handling for missing coordinates."""
    try:
        return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2 + (end.z - start.z)**2)
    except AttributeError as e:
        logging.error(f"Error calculating length: {e}")
        # Return a default length of 0 if start or end points are malformed
        return 0.0

def get_transparency(entity):
    """Helper function to get transparency value with error handling."""
    try:
        # Get transparency value if it exists, otherwise return 0 (fully opaque)
        transparency = getattr(entity.dxf, 'transparency', 0)
        # Convert from DXF transparency format to percentage (0-100)
        if isinstance(transparency, int):
            # Convert from DXF integer format (0-255) to percentage
            return (255 - transparency) / 255.0 * 100 if transparency != 0 else 0
        return transparency * 100 if transparency else 0
    except AttributeError as e:
        logging.warning(f"Could not get transparency value: {e}")
        return 0

# DXF to PANDAS DATAFRAME
def Dxf_to_DF_1(filename):
    try:
        doc = ezdxf.readfile(filename)
        logging.info(f"Successfully opened DXF file: {filename}")
    except IOError as e:
        logging.error(f"Cannot open file: {filename} - {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
    except ezdxf.DXFStructureError as e:
        logging.error(f"Invalid or corrupt DXF file: {filename} - {e}")
        return pd.DataFrame()

    msp = doc.modelspace()
    entities_data = []

    for entity in msp:
        entity_data = {'Type': entity.dxftype()}

        try:
            entity_data['Layer'] = entity.dxf.layer
            # Add transparency for all entities
            entity_data['Transparency'] = get_transparency(entity)
            logging.info(f"Processing entity on layer '{entity.dxf.layer}' with transparency {entity_data['Transparency']}%")
        except AttributeError:
            entity_data['Layer'] = 'Unknown'
            entity_data['Transparency'] = 0
            logging.warning(f"Entity with unknown layer: {entity.dxftype()}")

        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                length = calculate_length(start, end)
                entity_data.update({
                    'X_start': start.x, 'Y_start': start.y, 'Z_start': start.z,
                    'X_end': end.x, 'Y_end': end.y, 'Z_end': end.z,
                    'Length': length
                })
                logging.info(f"Line entity with length {length} processed.")

            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                entity_data.update({
                    'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                    'Radius': radius
                })
                logging.info(f"Circle entity with radius {radius} processed.")

            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                entity_data.update({
                    'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                    'Radius': radius,
                    'Start Angle': start_angle,
                    'End Angle': end_angle
                })
                logging.info(f"Arc entity with radius {radius} and angles {start_angle}-{end_angle} processed.")

            elif entity.dxftype() == 'TEXT':
                insert = entity.dxf.insert
                text = entity.dxf.text
                entity_data.update({
                    'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                    'Text': text
                })
                logging.info(f"Text entity with content '{text}' processed.")

            elif entity.dxftype() == 'MTEXT':
                text = entity.plain_text()
                insertion_point = entity.dxf.insert
                entity_data.update({
                    'Text': text,
                    'X_insert': insertion_point.x,
                    'Y_insert': insertion_point.y,
                    'Z_insert': insertion_point.z
                })
                logging.info(f"MTEXT entity with content '{text}' processed.")
            
            elif entity.dxftype() == 'INSERT':
                insert = entity.dxf.insert
                name = entity.dxf.name
                entity_data.update({
                    'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                    'Block Name': name
                })
                logging.info(f"Insert entity with block name '{name}' processed.")

        except AttributeError as e:
            logging.error(f"Error processing entity {entity.dxftype()}: {e}")
            # Skip the entity if key attributes are missing
            continue

        entities_data.append(entity_data)

    # Return a DataFrame of all extracted entities
    return pd.DataFrame(entities_data)

# %% [markdown]
# # Step 4 - Floor distribution

# %%
def calculate_ranges(df: pd.DataFrame) -> List[Tuple[float, float]]:
    """
    Calculate floor ranges based on X coordinates in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with X_start and X_end coordinates
        
    Returns:
        List[Tuple[float, float]]: List of (start, end) coordinate ranges
    """
    try:
        if df.empty:
            logger.warning("Empty dataframe provided to calculate_ranges")
            return []
            
        if not all(col in df.columns for col in ['X_start', 'X_end']):
            logger.error("Required columns 'X_start' or 'X_end' missing")
            return []
            
        # Calculate lengths and handle potential NaN values
        df['Length'] = np.abs(df['X_end'] - df['X_start']).round(4)
        df = df.dropna(subset=['Length'])
        
        if df.empty:
            logger.warning("No valid lengths could be calculated")
            return []
            
        max_along_x = df['Length'].max()
        if pd.isna(max_along_x):
            logger.warning("Could not determine maximum length")
            return []
            
        max_x_df = df[df['Length'] == max_along_x]
        if max_x_df.empty:
            logger.warning("No entries found with maximum length")
            return []
            
        # Collect and sort all X coordinates
        coordinates = []
        coordinates.extend(max_x_df['X_start'].dropna().unique())
        coordinates.extend(max_x_df['X_end'].dropna().unique())
        
        if not coordinates:
            logger.warning("No valid coordinates found")
            return []
            
        sorted_coords = sorted(set(coordinates))
        
        # Create ranges with validation
        ranges = []
        for i in range(0, len(sorted_coords) - 1, 2):
            if i + 1 < len(sorted_coords):
                ranges.append((sorted_coords[i], sorted_coords[i + 1]))
        
        logger.info(f"Successfully calculated {len(ranges)} ranges")
        return ranges
        
    except Exception as e:
        logger.error(f"Error in calculate_ranges: {str(e)}")
        return []

def distribute_floors(df: pd.DataFrame, 
                     column_name: str, 
                     ranges: List[Tuple[float, float]], 
                     tolerance: float = 5.0) -> pd.DataFrame:
    """
    Distribute elements to floors based on their X coordinates.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_name (str): Column to use for X coordinates
        ranges (List[Tuple[float, float]]): Floor ranges
        tolerance (float): Tolerance for range matching
        
    Returns:
        pd.DataFrame: Dataframe with floor assignments
    """
    try:
        if df.empty or not ranges:
            logger.warning("Empty dataframe or ranges provided to distribute_floors")
            return pd.DataFrame()
            
        if column_name not in df.columns:
            logger.error(f"Column {column_name} not found in dataframe")
            return pd.DataFrame()
            
        floor_dfs = []
        for floor_idx, (start, end) in enumerate(ranges):
            mask = (
                df[column_name].notna() & 
                (df[column_name] >= (start - tolerance)) & 
                (df[column_name] <= (end + tolerance))
            )
            floor_df = df[mask].copy()
            floor_df['floor'] = floor_idx
            floor_dfs.append(floor_df)
            
        if floor_dfs:
            result = pd.concat(floor_dfs, ignore_index=True)
            logger.info(f"Successfully distributed {len(result)} elements to floors")
            return result
            
        logger.warning("No elements could be distributed to floors")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in distribute_floors: {str(e)}")
        return pd.DataFrame()

def spline_floor_text(df: pd.DataFrame, 
                     ranges: List[Tuple[float, float]], 
                     tolerance: float = 5.0) -> pd.DataFrame:
    """
    Assign floors to spline elements based on X coordinates.
    
    Args:
        df (pd.DataFrame): Input dataframe
        ranges (List[Tuple[float, float]]): Floor ranges
        tolerance (float): Tolerance for range matching
        
    Returns:
        pd.DataFrame: Dataframe with floor assignments
    """
    try:
        if df.empty:
            logger.warning("Empty dataframe provided to spline_floor_text")
            return df
            
        floor_values = []
        x_columns = [col for col in df.columns if col.startswith('X') and re.match(r'.*\d$', col)]
        
        if not x_columns:
            logger.warning("No X coordinate columns found for splines")
            return df
            
        for _, row in df.iterrows():
            floor_idx = -1
            for idx, (start, end) in enumerate(ranges):
                all_within_range = True
                valid_coordinates = False
                
                for column in x_columns:
                    if pd.notna(row[column]):
                        valid_coordinates = True
                        if not (start - tolerance <= row[column] <= end + tolerance):
                            all_within_range = False
                            break
                            
                if valid_coordinates and all_within_range:
                    floor_idx = idx
                    break
                    
            floor_values.append(floor_idx)
            
        df['floor'] = floor_values
        logger.info(f"Successfully processed {len(df)} spline elements")
        return df
        
    except Exception as e:
        logger.error(f"Error in spline_floor_text: {str(e)}")
        return df

def floor_main(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to process and distribute all elements to floors.
    
    Args:
        df (pd.DataFrame): Input dataframe containing all elements
        
    Returns:
        pd.DataFrame: Processed dataframe with floor assignments
    """
    try:
        if df.empty:
            logger.warning("Empty dataframe provided to floor_main")
            return pd.DataFrame()
            
        ranges = calculate_ranges(df)
        if not ranges:
            logger.warning("No valid ranges calculated")
            return pd.DataFrame()
            
        # Split dataframe by element types
        type_groups = {
            'lines': df[df['Type'].isin(['LINE', 'POINT'])],
            'texts': df[df['Type'].isin(['TEXT', 'MTEXT', 'INSERT'])],
            'curves': df[df['Type'].isin(['CIRCLE', 'ARC', 'ELLIPSE'])],
            'splines': df[df['Type'].isin(['SPLINE'])]
        }
        
        floor_dfs = []
        
        # Process each group
        for group_name, group_df in type_groups.items():
            if group_df.empty:
                continue
                
            if group_name == 'splines':
                result = spline_floor_text(group_df, ranges)
            elif group_name == 'texts':
                result = distribute_floors(group_df, 'X_insert', ranges)
            else:
                result = distribute_floors(group_df, 'X_start', ranges)
                
            if not result.empty:
                floor_dfs.append(result)
                
        if floor_dfs:
            final_df = pd.concat(floor_dfs, ignore_index=True)
            logger.info(f"Successfully processed {len(final_df)} total elements")
            return final_df
            
        logger.warning("No elements could be processed")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error in floor_main: {str(e)}")
        return pd.DataFrame()

# %% [markdown]
# # Step 5 - Dxf from dataframe

# %%
def create_dxf_from_dataframe(df):
    doc = ezdxf.new()
    layers = {}
    for index, row in df.iterrows():
        layer_name = str(row['Layer'])  
        if not layer_name or layer_name.lower() == 'nan': 
            continue
            
        # Create new layer if it doesn't exist
        if layer_name not in layers and layer_name != '0': 
            layers[layer_name] = doc.layers.new(name=layer_name)
            
        msp = doc.modelspace()
        
        # Get transparency value from dataframe, default to 0 if not present
        transparency = row.get('Transparency', 0)
        # Convert transparency to DXF format (0-255, where 0 is opaque)
        dxf_transparency = int((1 - transparency/100) * 255) if transparency is not None else 0
        
        # Base attributes dictionary with layer and transparency
        base_attrs = {
            'layer': layer_name,
            'transparency': dxf_transparency
        }
        
        if row['Type'] == 'LINE':
            start = (row['X_start'], row['Y_start'])
            end = (row['X_end'], row['Y_end'])
            msp.add_line(start, end, dxfattribs=base_attrs)
        elif row['Type'] == 'CIRCLE':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            msp.add_circle(center, radius, dxfattribs=base_attrs)
        elif row['Type'] == 'ARC':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            start_angle = row['Start Angle']
            end_angle = row['End Angle']
            msp.add_arc(center, radius, start_angle, end_angle, dxfattribs=base_attrs)
        elif row['Type'] == 'TEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row.get('Text Content', '')
            text_attrs = base_attrs.copy()
            text_attrs['insert'] = insert
            msp.add_text(text, dxfattribs=text_attrs)
        elif row['Type'] == 'MTEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row.get('Text Content', '')
            text_attrs = base_attrs.copy()
            text_attrs['insert'] = insert
            msp.add_mtext(text, dxfattribs=text_attrs)
        elif row['Type'] == 'SPLINE':
            fit_point_count = int(row['Fit Point Count']) if not pd.isna(row['Fit Point Count']) else 0
            fit_points = [(row[f'X{i}'], row[f'Y{i}'], row[f'Z{i}']) for i in range(fit_point_count)]
            msp.add_spline(fit_points, dxfattribs=base_attrs)
        elif row['Type'] == 'ELLIPSE':
            center = (row['X_center'], row['Y_center'])
            major_axis = (row['X_major_axis'], row['Y_major_axis'])
            ratio = row['Ratio']
            msp.add_ellipse(center, major_axis, ratio, dxfattribs=base_attrs)
        elif row['Type'] == 'POINT':
            location = (row['X_start'], row['Y_start'])
            msp.add_point(location, dxfattribs=base_attrs)
                                                
    return doc

# %% [markdown]
# # Step 6 - Max along x and Max along y

# %%
def calculate_max_along_x_y(df):
    # Filter the DataFrame for rows where 'Layer' is 'Boundary'
    boundary_df = df[df['Layer'] == 'Boundary']
    
    # Calculate max_along_x and max_along_y for the filtered DataFrame
    max_along_x = np.round(np.max(np.abs(boundary_df['X_end'] - boundary_df['X_start'])), 1)
    max_along_y = np.round(np.max(np.abs(boundary_df['Y_end'] - boundary_df['Y_start'])), 1)

    # Return the values for later use
    return max_along_x, max_along_y

# %% [markdown]
# # Step 7- Nested function

# %%
def shift_dxf_to_coordinates(doc, target_x=0, target_y=0):
    """
    Shifts the DXF entities such that the minimum coordinates align with the given target coordinates.
    
    Parameters:
    - doc: ezdxf.document.Drawing, the DXF document to process.
    - target_x: float, the target x-coordinate to align the minimum x-coordinate.
    - target_y: float, the target y-coordinate to align the minimum y-coordinate.
    
    Returns:
    - The modified DXF document.
    """
    if not doc:
        logging.error("Invalid DXF document.")
        return None

    msp = doc.modelspace()

    min_x, min_y = float('inf'), float('inf')

    # Define a function to get the minimum coordinates for each entity
    def update_min_coords(x, y):
        nonlocal min_x, min_y
        min_x = min(min_x, x)
        min_y = min(min_y, y)

    # First pass: Find the minimum x and y coordinates
    for entity in msp:
        try:
            if entity.dxftype() == 'CIRCLE' or entity.dxftype() == 'ARC':
                center = entity.dxf.center
                update_min_coords(center.x, center.y)

            elif entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                update_min_coords(start.x, start.y)
                update_min_coords(end.x, end.y)

            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                insert = entity.dxf.insert
                update_min_coords(insert.x, insert.y)

            elif entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points()
                for point in points:
                    update_min_coords(point[0], point[1])

            elif entity.dxftype() == 'POLYLINE':
                vertices = [v.dxf.location for v in entity.vertices]
                for vertex in vertices:
                    update_min_coords(vertex[0], vertex[1])

            elif entity.dxftype() == 'INSERT':
                insert = entity.dxf.insert
                update_min_coords(insert.x, insert.y)

        except AttributeError:
            # Skip any malformed entities that are missing coordinates
            continue

    # Calculate the offsets
    if min_x == float('inf') or min_y == float('inf'):
        return doc

    offset_x = target_x - min_x
    offset_y = target_y - min_y

    # Second pass: Update the coordinates of the entities
    for entity in msp:
        try:
            if entity.dxftype() == 'CIRCLE' or entity.dxftype() == 'ARC':
                entity.dxf.center = (
                    entity.dxf.center.x + offset_x,
                    entity.dxf.center.y + offset_y,
                    entity.dxf.center.z
                )

            elif entity.dxftype() == 'LINE':
                entity.dxf.start = (
                    entity.dxf.start.x + offset_x,
                    entity.dxf.start.y + offset_y,
                    entity.dxf.start.z
                )
                entity.dxf.end = (
                    entity.dxf.end.x + offset_x,
                    entity.dxf.end.y + offset_y,
                    entity.dxf.end.z
                )

            elif entity.dxftype() in ['TEXT', 'MTEXT']:
                entity.dxf.insert = (
                    entity.dxf.insert.x + offset_x,
                    entity.dxf.insert.y + offset_y,
                    entity.dxf.insert.z
                )

            elif entity.dxftype() == 'LWPOLYLINE':
                points = [(p[0] + offset_x, p[1] + offset_y, *p[2:]) for p in entity.get_points()]
                entity.set_points(points)

            elif entity.dxftype() == 'POLYLINE':
                for vertex in entity.vertices:
                    vertex.dxf.location = (
                        vertex.dxf.location[0] + offset_x,
                        vertex.dxf.location[1] + offset_y,
                        vertex.dxf.location[2]
                    )
            
            elif entity.dxftype() == 'INSERT':
                entity.dxf.insert = (
                    entity.dxf.insert.x + offset_x,
                    entity.dxf.insert.y + offset_y,
                    entity.dxf.insert.z
                )

        except AttributeError:
            # Skip any entities that cannot be updated
            continue

    return doc

# %% [markdown]
# # Step 8 - Four corners columns

# %%
def four_corners(dxf_data, max_along_x, max_along_y, width=9, height=12):
    """
    Adds four rectangular boxes to the corners of the drawing's bounding box in a DXF data.
    """
    msp = dxf_data.modelspace()

    # Calculate the bounding box of the drawing
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
    for entity in msp:
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                min_x = min(min_x, start.x, end.x)
                min_y = min(min_y, start.y, end.y)
                max_x = max(max_x, start.x, end.x)
                max_y = max(max_y, start.y, end.y)
        except AttributeError:
            continue  # Skip malformed entities

    # Ensure bounds are valid
    if min_x == float('inf') or max_x == -float('inf'):
        logging.error("Error: No valid entities found to calculate bounds.")
        return dxf_data  # Ensure to return the dxf_data even if there's an error

    # Add boxes to the corners
    create_box(msp, (min_x, min_y), width, height)                # Bottom-left
    create_box(msp, (min_x, max_y - height), width, height)       # Top-left
    create_box(msp, (max_x - width, max_y - height), width, height)  # Top-right
    create_box(msp, (max_x - width, min_y), width, height)        # Bottom-right

    return dxf_data  # Return the modified dxf_data

def create_box(msp, start_point, width, height):
    """
    Creates a rectangular box as a closed polyline with a red hatch in the DXF modelspace.
    Both the polyline and hatch will be placed in the 'column' layer.
    """
    # Define corners of the box
    p1 = start_point
    p2 = (p1[0] + width, p1[1])
    p3 = (p2[0], p2[1] + height)
    p4 = (p1[0], p1[1] + height)
    
    # Create a closed polyline (DXF color index 1 is red)
    points = [p1, p2, p3, p4, p1]
    polyline = msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
    
    # Add a red hatch inside the box
    hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})
    hatch.paths.add_polyline_path(points, is_closed=True)

# %% [markdown]
# # Step 9 -Boundary 1 columns

# %%
def Boundary_1(dxf_data, target_x=9, tolerance=1, width=9, height=12, max_along_y=None):
    """
    Function to add multiple boxes to a DXF file along vertical lines at a specified x-coordinate.
    
    Parameters:
        - dxf_data: In-memory DXF document object.
        - target_x: Target x-coordinate for vertical lines (default: 9).
        - tolerance: Tolerance for matching vertical lines (default: 1).
        - width: Width of the boxes to be added (default: 9 units).
        - height: Height of the boxes to be added (default: 12 units).
        - max_along_y: Maximum y-coordinate value (float, must be provided).
    
    Returns:
        - Modified in-memory DXF document object.
    """
    if max_along_y is None:
        raise ValueError("max_along_y must be provided and cannot be None")

    # Access modelspace from the in-memory DXF data
    msp = dxf_data.modelspace()

    # List to store y-coordinates
    y_coordinates = []

    # Query and process LINE entities
    for line in msp.query('LINE'):
        x_start, y_start, _ = line.dxf.start
        x_end, y_end, _ = line.dxf.end

        # Identify vertical lines within the tolerance range of target_x
        if abs(x_start - target_x) <= tolerance and abs(x_end - target_x) <= tolerance:
            y_coordinates.extend([int(y_start), int(y_end)])

    # Sort and filter out duplicates from y-coordinates
    y_coordinates_sorted = sorted(y_coordinates)
    counts = Counter(y_coordinates_sorted)
    y_coordinates_filtered = [y for y in y_coordinates_sorted if counts[y] == 1]

    # Remove first and last elements if conditions are met
    new_y_coordinates = remove_first_last_if_conditions_met(y_coordinates_filtered, max_along_y)

    # Check if new_y_coordinates is empty
    if not new_y_coordinates:
        # Calculate the number of boxes
        num_boxes = max(0, math.ceil((max_along_y) / 144) - 2)

        # Initial start point for the first box
        start_point = (0, 144)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1])
            p3 = (p2[0], p1[1] + height)
            p4 = (p1[0], p3[1])

            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0], start_point[1] + 144)
    else:
        # Create boxes and hatches along the vertical line based on new_y_coordinates
        for i in range(0, len(new_y_coordinates), 2):
            if i + 1 < len(new_y_coordinates):
                p1 = (target_x, new_y_coordinates[i])
                p2 = (target_x, new_y_coordinates[i] + height)
                p3 = (target_x - width, p2[1])
                p4 = (p3[0], p1[1])

                # Create a closed polyline (box)
                points = [p1, p2, p3, p4, p1]
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # DXF color index: 1 is red
                hatch.paths.add_polyline_path(points, is_closed=True)

    # Return the modified DXF data
    return dxf_data


def remove_first_last_if_conditions_met(y_coordinates_sorted, max_along_y):
    """
    Removes the first and last y-coordinate from the list if specific conditions are met.
    
    Parameters:
        - y_coordinates_sorted: Sorted list of y-coordinates.
        - max_along_y: Maximum y-coordinate in the drawing.
    
    Returns:
        - Updated list of y-coordinates.
    """
    if len(y_coordinates_sorted) < 2:
        return y_coordinates_sorted

    first_value = y_coordinates_sorted[0]
    last_value = y_coordinates_sorted[-1]

    if ((-1 <= first_value <= 1 or 8 <= first_value <= 10) and
        (max_along_y - 10 <= last_value <= max_along_y - 8)):
        return y_coordinates_sorted[1:-1]  # Remove first and last element

    return y_coordinates_sorted

# %% [markdown]
# # Step 10 -Boundary 2 columns

# %%
def Boundary_2(dxf_data, width=12, height=9, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Function Description:
    This function identifies horizontal lines near a specified y-coordinate within a tolerance 
    range and creates boxes at filtered x-coordinates along these lines. It adds these closed
    polylines (boxes) into the modelspace of a DXF document. The boxes are filled with red color 
    using hatches. The width and height of the boxes are customizable.

    Parameters:
    (1) dxf_data: The DXF data object (loaded DXF file).
    (2) width: The width of each box (float, default is 12 units).
    (3) height: The height of each box (float, default is 9 units).
    (4) tolerance: The tolerance for matching the y-coordinate of horizontal lines (float, default is 1 unit).
    (5) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (6) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    None: This function does not return any values directly. It modifies the provided DXF data.
    '''
    
    msp = dxf_data.modelspace()

    # Ensure max_along_x and max_along_y are provided
    if max_along_x is None or max_along_y is None:
        raise ValueError("Both max_along_x and max_along_y must be provided and cannot be None")

    # Define the target y position for horizontal lines
    target_y = max_along_y - 9

    # Step 1: Find horizontal lines near the target y position
    horizontal_lines = []
    for line in msp.query('LINE'):
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        
        # Check if the line is horizontal within the tolerance range
        if abs(y_start - target_y) <= tolerance and abs(y_end - target_y) <= tolerance:
            horizontal_lines.append(line)

    # Step 2: Extract x-coordinates from the found lines
    x_coordinates = []
    for line in horizontal_lines:
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        x_coordinates.extend([int(x_start), int(x_end)])

    # Step 3: Sort and filter the x-coordinates
    x_coordinates_sorted = sorted(x_coordinates)
    counts = Counter(x_coordinates_sorted)
    x_coordinates_filtered = [num for num in x_coordinates_sorted if counts[num] == 1]

    # Step 4: Trim the list if conditions are met
    def remove_first_last_if_conditions_met(x_coordinates_filtered, max_along_x):
        if not isinstance(x_coordinates_filtered, list) or len(x_coordinates_filtered) < 2:
            raise ValueError("Input must be a list with at least 2 elements")
        if not isinstance(max_along_x, (int, float)):
            raise ValueError("max_along_x must be a number")

        total_digits = sum(len(str(abs(num))) for num in x_coordinates_filtered)
        first_value = x_coordinates_filtered[0]
        last_value = x_coordinates_filtered[-1]
        
        if ((-1 <= first_value <= 1 or 8 <= first_value <= 10) and
            (max_along_x - 10 <= last_value <= max_along_x - 8) and
            total_digits % 2 == 0):
            updated_x_coordinates = x_coordinates_filtered[1:-1]
        else:
            updated_x_coordinates = x_coordinates_filtered
        
        return updated_x_coordinates

    # Apply the function to get new x-coordinates
    new_x_coordinates = remove_first_last_if_conditions_met(x_coordinates_filtered, max_along_x)

    # Step 5: Draw boxes based on x-coordinates or default layout if empty
    if not new_x_coordinates:
        # Calculate the number of boxes to draw
        num_boxes = max(0, math.ceil((max_along_x) / 144) - 2)

        # Initial start point for the first box
        start_point = (148, max_along_y)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1])
            p3 = (p2[0], p1[1] - height)
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0] + 148, start_point[1])

    else:
        # Draw boxes at the provided x-coordinates
        for i in range(0, len(new_x_coordinates), 2):
            start_point = (new_x_coordinates[i], target_y)
            
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1])
            p3 = (p2[0], p1[1] + height)
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1,  dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    return dxf_data  # Ensure dxf_data is returned after modification

# %% [markdown]
# # Step 11 -Boundary 3 columns

# %%
def Boundary_3(dxf_data, width=9, height=12, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Function Description:
    This function identifies vertical lines near a specified x-coordinate within a tolerance range 
    and creates boxes at filtered y-coordinates along these lines. It adds closed polylines (boxes) 
    into the modelspace of a DXF document. The boxes are filled with red color using hatches. The width
    and height of the boxes are customizable.

    Parameters:
    (1) dxf_data: The DXF data object (loaded DXF file).
    (2) width: The width of each box (float, default is 9 units).
    (3) height: The height of each box (float, default is 12 units).
    (4) tolerance: The tolerance for matching the x-coordinate of vertical lines (float, default is 1 unit).
    (5) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (6) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    dxf_data: The modified DXF data object.
    '''
    
    msp = dxf_data.modelspace()

    # Ensure max_along_x and max_along_y are provided
    if max_along_x is None or max_along_y is None:
        raise ValueError("Both max_along_x and max_along_y must be provided and cannot be None")

    # Define the target x position for vertical lines
    target_x = max_along_x - 9

    # Step 1: Find vertical lines near the target x position
    vertical_lines = []
    for line in msp.query('LINE'):
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        
        # Check if the line is vertical within the tolerance range
        if abs(x_start - target_x) <= tolerance and abs(x_end - target_x) <= tolerance:
            vertical_lines.append(line)

    # Step 2: Extract y-coordinates from the found lines
    y_coordinates = []
    for line in vertical_lines:
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        y_coordinates.extend([int(y_start), int(y_end)])

    # Step 3: Sort and filter the y-coordinates
    y_coordinates_sorted = sorted(y_coordinates)
    counts = Counter(y_coordinates_sorted)
    y_coordinates_filtered = [num for num in y_coordinates_sorted if counts[num] == 1]

    # Step 4: Trim the list if conditions are met
    def remove_first_last_if_conditions_met(y_coordinates_filtered, max_along_y):
        if not isinstance(y_coordinates_filtered, list) or len(y_coordinates_filtered) < 2:
            raise ValueError("Input must be a list with at least 2 elements")
        if not isinstance(max_along_y, (int, float)):
            raise ValueError("max_along_y must be a number")

        total_digits = sum(len(str(abs(num))) for num in y_coordinates_filtered)
        first_value = y_coordinates_filtered[0]
        last_value = y_coordinates_filtered[-1]
        
        if ((-1 <= first_value <= 1 or 8 <= first_value <= 10) and
            (max_along_y - 10 <= last_value <= max_along_y - 8) and
            total_digits % 2 == 0):
            updated_y_coordinates = y_coordinates_filtered[1:-1]
        else:
            updated_y_coordinates = y_coordinates_filtered
        
        return updated_y_coordinates

    # Apply the function to get new y-coordinates
    new_y_coordinates = remove_first_last_if_conditions_met(y_coordinates_filtered, max_along_y)

    # Step 5: Draw boxes based on y-coordinates or default layout if empty
    if not new_y_coordinates:
        # Calculate the number of boxes to draw
        num_boxes = math.ceil((max_along_y - 9) / 144) - 2

        # Initial start point for the first box
        start_point = (max_along_x, 148)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] - width, p1[1]) 
            p3 = (p2[0], p1[1] + height) 
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0], start_point[1] + 148)

    else:
        # Draw boxes at the provided y-coordinates
        for i in range(0, len(new_y_coordinates), 2):
            start_point = (target_x, new_y_coordinates[i])  # Box start point at target_x and filtered y-coordinate
            
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1]) 
            p3 = (p2[0], p1[1] + height) 
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1,  dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    # Return the modified dxf_data object
    return dxf_data

# %% [markdown]
# # Step 12-Boundary 4 columns

# %%
def Boundary_4(dxf_data, width=12, height=9, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Function Description:
    This function identifies horizontal lines near a specified y-coordinate within a tolerance range 
    and creates boxes at filtered x-coordinates along these lines. It adds closed polylines (boxes) 
    into the modelspace of a DXF object, filling them with red color using hatches. The width and height 
    of the boxes are customizable. The function also tracks if a trim condition is applied to the coordinates, 
    and handles special cases for box creation based on the provided `max_along_x`.

    Parameters:
    (1) dxf_data: The DXF data object (loaded DXF file in memory).
    (2) width: The width of each box (float, default is 12 units).
    (3) height: The height of each box (float, default is 9 units).
    (4) tolerance: The tolerance for matching the y-coordinate of horizontal lines (float, default is 1 unit).
    (5) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (6) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    dxf_data: The modified DXF data object (in-memory).
    '''
    
    msp = dxf_data.modelspace()

    # Ensure max_along_x and max_along_y are provided
    if max_along_x is None or max_along_y is None:
        raise ValueError("Both max_along_x and max_along_y must be provided and cannot be None")

    # Define the target y position for horizontal lines
    target_y = 9

    # Step 1: Find horizontal lines near the target y position
    horizontal_lines = []
    for line in msp.query('LINE'):
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        
        # Check if the line is horizontal within the tolerance range
        if abs(y_start - target_y) <= tolerance and abs(y_end - target_y) <= tolerance:
            horizontal_lines.append(line)

    # Step 2: Extract x-coordinates from the found lines
    x_coordinates = []
    for line in horizontal_lines:
        x_start, y_start, z_start = line.dxf.start
        x_end, y_end, z_end = line.dxf.end
        x_coordinates.extend([int(x_start), int(x_end)])

    # Step 3: Sort and filter the x-coordinates
    x_coordinates_sorted = sorted(x_coordinates)
    counts = Counter(x_coordinates_sorted)
    x_coordinates_filtered = [num for num in x_coordinates_sorted if counts[num] == 1]

    # Step 4: Trim the list if conditions are met and track whether trim is applied
    trim_applied = False  # Initialize trim applied boolean
    def remove_first_last_if_conditions_met(x_coordinates_filtered, max_along_x):
        nonlocal trim_applied
        if not isinstance(x_coordinates_filtered, list) or len(x_coordinates_filtered) < 2:
            raise ValueError("Input must be a list with at least 2 elements")
        if not isinstance(max_along_x, (int, float)):
            raise ValueError("max_along_x must be a number")

        total_digits = sum(len(str(abs(num))) for num in x_coordinates_filtered)
        first_value = x_coordinates_filtered[0]
        last_value = x_coordinates_filtered[-1]
        
        if ((-1 <= first_value <= 1 or 8 <= first_value <= 10) and
            (max_along_x - 10 <= last_value <= max_along_x - 8) and
            total_digits % 2 == 0):
            trim_applied = True  # Set to True if trimming is applied
            updated_x_coordinates = x_coordinates_filtered[1:-1]
        else:
            updated_x_coordinates = x_coordinates_filtered
        
        return updated_x_coordinates

    # Apply the function to get new x-coordinates
    new_x_coordinates = remove_first_last_if_conditions_met(x_coordinates_filtered, max_along_x)

    # Check if the new_x_coordinates list is empty
    new_x_coordinates_empty = len(new_x_coordinates) == 0  # True if list is empty, otherwise False

    # Step 5: Add boxes based on conditions
    if trim_applied:
        if new_x_coordinates_empty:
            pass  #"Trim applied but no x-coordinates left, skipping box addition."
        else:
            for i in range(0, len(new_x_coordinates), 2):
                start_point = (new_x_coordinates[i], target_y)
                
                # Calculate the corners of the box
                p1 = start_point
                p2 = (p1[0] + width, p1[1])
                p3 = (p2[0], p1[1] - height)
                p4 = (p1[0], p3[1])
                
                # Create a closed polyline for the box
                points = [p1, p2, p3, p4, p1]
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1,  dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
    else:
        # If trim_applied is False, check if the last element is (max_along_x - 9)
        if new_x_coordinates and new_x_coordinates[-1] == (max_along_x - 9):
            # If only two elements, create a single box with the given logic
            if len(new_x_coordinates) == 2:
                start_point = (new_x_coordinates[0], target_y)
                
                # Calculate the corners of the box
                p1 = start_point
                p2 = (p1[0] + width, p1[1])
                p3 = (p2[0], p1[1] - height)
                p4 = (p1[0], p3[1])
                
                # Create a closed polyline for the box
                points = [p1, p2, p3, p4, p1]
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
            else:
                # First create a box with the first element, using the same logic as above
                start_point = (new_x_coordinates[0], target_y)
                
                # Calculate the corners of the box
                p1 = start_point
                p2 = (p1[0] + width, p1[1])
                p3 = (p2[0], p1[1] - height)
                p4 = (p1[0], p3[1])
                
                # Create a closed polyline for the first box
                points = [p1, p2, p3, p4, p1]
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
                
                # Then create more boxes at odd indices (index 1, 3, etc.)
                for i in range(1, len(new_x_coordinates), 2):
                    start_point = (new_x_coordinates[i], target_y)
                    
                    # Calculate the corners of the box
                    p1 = start_point
                    p2 = (p1[0] + width, p1[1])
                    p3 = (p2[0], p1[1] - height)
                    p4 = (p1[0], p3[1])
                    
                    # Create a closed polyline for the additional boxes
                    points = [p1, p2, p3, p4, p1]
                    msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                    
                    # Add a hatch to fill the polyline with red color
                    hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                    hatch.paths.add_polyline_path(points, is_closed=True)
        else:
            # If trim_applied is False, check if the first value is 0 or 9 and add boxes in reverse
            if new_x_coordinates and (new_x_coordinates[0] == 0 or new_x_coordinates[0] == 9):
                # If there are only two elements, add one box using the specified logic
                if len(new_x_coordinates) == 2:
                    start_point = (new_x_coordinates[1], target_y)
                
                    # Calculate the corners of the box
                    p1 = start_point
                    p2 = (p1[0] - width, p1[1])
                    p3 = (p2[0], p1[1] - height)
                    p4 = (p1[0], p3[1])

                    # Create a closed polyline for the box
                    points = [p1, p2, p3, p4, p1]
                    msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

                    # Add a hatch to fill the polyline with red color
                    hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                    hatch.paths.add_polyline_path(points, is_closed=True)
                else:
                    # Add boxes using the reverse logic for lists with more than two elements
                    for i in range(len(new_x_coordinates) - 1, 0, -2):
                        start_point = (new_x_coordinates[i], target_y)
                    
                        # Calculate the corners of the box
                        p1 = start_point
                        p2 = (p1[0] + width, p1[1])
                        p3 = (p2[0], p1[1] + height)
                        p4 = (p1[0], p3[1])

                        # Create a closed polyline for the box
                        points = [p1, p2, p3, p4, p1]
                        msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

                        # Add a hatch to fill the polyline with red color
                        hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                        hatch.paths.add_polyline_path(points, is_closed=True)
            else:
                pass

    # Return the modified DXF data
    return dxf_data

# %% [markdown]
# # Step 13 - Helper function

# %%
def filter_horizontal_lines(df, max_along_x, max_along_y, variance_threshold=0.95):
    '''
    Upstream Libraries:
    - numpy (for numerical operations)
    - pandas (for handling DataFrame)
    - ezdxf (for working with DXF files)
    - sklearn.decomposition.PCA (for principal component analysis)
    
    Upstream Functions needed:
    None
    
    Function Description:
    This function filters horizontal lines from the input DataFrame by applying PCA (Principal Component Analysis) and 
    ensuring the lines fall within a specified x and y coordinate range.

    Parameters:
    - df: Input DataFrame containing line data.
    - max_along_x: Maximum x-coordinate value (float).
    - max_along_y: Maximum y-coordinate value (float).
    - variance_threshold: Threshold (float) for PCA variance to consider a line horizontal. Default is 0.95.
    
    Returns:
    - A DataFrame containing filtered horizontal lines.
    '''
    # Function to determine if a line is horizontal using PCA
    def is_horizontal_pca(x_start, y_start, x_end, y_end):
        line_points = np.array([[x_start, y_start], [x_end, y_end]])
        pca = PCA(n_components=2)
        pca.fit(line_points)
        return abs(pca.components_[0][0]) >= variance_threshold

    # Drop rows with NaN values in critical columns
    df = df.dropna(subset=['X_start', 'Y_start', 'X_end', 'Y_end'])
    
    # Apply horizontal line check
    df['is_horizontal'] = df.apply(
        lambda row: is_horizontal_pca(row['X_start'], row['Y_start'], row['X_end'], row['Y_end']),
        axis=1
    )

    # Filter the DataFrame based on specified conditions
    df_filtered = df[
        (df['X_start'] >= 9) & (df['X_end'] <= (max_along_x - 9)) &
        (df['Y_start'] >= 9) & (df['Y_start'] <= (max_along_y - 9)) & 
        df['is_horizontal']
    ].drop(columns=['is_horizontal'])

    return df_filtered

def extract_matching_line_pairs(df):
    '''
    Upstream Libraries:
    - numpy (for numerical operations)
    - pandas (for handling DataFrame)
    
    Upstream Functions needed:
    None
    
    Function Description:
    This function extracts matching pairs of lines from the input DataFrame. It checks if two lines have similar starting 
    and ending points and ensures they are close to each other vertically.

    Parameters:
    - df: Input DataFrame containing filtered line data.
    
    Returns:
    - A DataFrame containing matching line pairs.
    '''
    matching_pairs = []
    
    for i, line1 in df.iterrows():
        for j, line2 in df.iterrows():
            if i >= j:
                continue
            if (
                (np.isclose(line1['X_start'], line2['X_start']) and np.isclose(line1['X_end'], line2['X_end'])) or
                (np.isclose(line1['X_start'], line2['X_end']) and np.isclose(line1['X_end'], line2['X_start']))
            ) and np.isclose(line1['Length'], line2['Length']) and (3.9 <= abs(line1['Y_start'] - line2['Y_start']) <= 5.1):
                matching_pairs.append({
                    'index_line1': i, 'line1': line1.to_dict(),
                    'index_line2': j, 'line2': line2.to_dict()
                })
    
    return pd.DataFrame(matching_pairs)

# %% [markdown]
# # Step 14 - Columns on walls connected with boundary 1 

# %%
def process_single_walls_left(df, max_along_x, max_along_y, dxf_data, width=12, height=9, variance_threshold=0.95, tolerance=1e-3):
    '''
    Function that processes the left walls in a DXF document based on a provided DataFrame of lines and other parameters.

    (Function docstring remains the same)
    '''

    # Step 1: Filter horizontal lines using the refactored helper function
    df_filtered = filter_horizontal_lines(df, max_along_x, max_along_y, variance_threshold)

    # Step 2: Extract matching line pairs using the refactored helper function
    df_pairs = extract_matching_line_pairs(df_filtered)

    # Step 2.1: Filter pairs by line length (only proceed if length is 60 or more)
    def filter_by_length(df_pairs, min_length=60):
        valid_pairs = []
        for _, pair in df_pairs.iterrows():
            # Calculate the length of both lines in the pair
            length1 = np.sqrt((pair['line1']['X_end'] - pair['line1']['X_start'])**2 + 
                              (pair['line1']['Y_end'] - pair['line1']['Y_start'])**2)
            length2 = np.sqrt((pair['line2']['X_end'] - pair['line2']['X_start'])**2 + 
                              (pair['line2']['Y_end'] - pair['line2']['Y_start'])**2)

            # Check if both lines meet the minimum length requirement
            if length1 >= min_length and length2 >= min_length:
                valid_pairs.append(pair)

        return pd.DataFrame(valid_pairs)

    df_pairs = filter_by_length(df_pairs)

    # Step 3: Extract X and Y values for creating boxes
    def extract_other_x_and_y_start(df_pairs, target_x=9, tolerance=1e-6):
        extracted_values = []
        for _, pair in df_pairs.iterrows():
            if np.isclose(pair['line1']['X_start'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line1'],
                    'X_other': pair['line1']['X_end'],
                    'Y_start': pair['line1']['Y_start']
                })
            elif np.isclose(pair['line1']['X_end'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line1'],
                    'X_other': pair['line1']['X_start'],
                    'Y_start': pair['line1']['Y_start']
                })
            if np.isclose(pair['line2']['X_start'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line2'],
                    'X_other': pair['line2']['X_end'],
                    'Y_start': pair['line2']['Y_start']
                })
            elif np.isclose(pair['line2']['X_end'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line2'],
                    'X_other': pair['line2']['X_start'],
                    'Y_start': pair['line2']['Y_start']
                })
        return pd.DataFrame(extracted_values)

    df_extracted = extract_other_x_and_y_start(df_pairs)

    # Step 4: Create boxes in the DXF data without overlapping
    def create_box_from_df(df_extracted, width, height, dxf_data):
        msp = dxf_data.modelspace()
        
        # Set to store existing box coordinates to prevent overlaps
        existing_boxes = set()

        # Detect nearby boxes within a 20-unit radius
        def is_near_existing_boxes(x, y, distance_threshold=96):
            for existing_x, existing_y in existing_boxes:
                if abs(existing_x - x) < distance_threshold and abs(existing_y - y) < distance_threshold:
                    return True
            return False

        for i in range(0, len(df_extracted), 2):
            if i + 1 < len(df_extracted):
                row1 = df_extracted.iloc[i]
                row2 = df_extracted.iloc[i + 1]
                x_value = row1['X_other']
                y_value = max(row1['Y_start'], row2['Y_start'])

                # Check if the proposed box location is clear of nearby boxes
                if is_near_existing_boxes(x_value, y_value):
                    continue  # Skip creating this box if it's too close to another

                # Check if a box already exists at these coordinates
                box_position = (x_value, y_value)
                if box_position in existing_boxes:
                    continue  # Skip creating this box if it overlaps

                # Add the new box coordinates to the set
                existing_boxes.add(box_position)

                # Define the box corners
                p1 = (x_value, y_value)
                p2 = (p1[0], p1[1] - height)
                p3 = (p1[0] - width, p1[1])
                p4 = (p1[0] - width, p1[1] - height)
                points = [p1, p3, p4, p2, p1]

                # Create the closed polyline for the box
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)

        # The DXF document is modified in memory; it does not need to be saved yet.
        return dxf_data

    # Call the function to add boxes to the DXF data
    dxf_data = create_box_from_df(df_extracted, width, height, dxf_data)

    # The modified dxf_data can now be used in further steps or saved when needed
    return dxf_data

# %% [markdown]
# #  Step 15 - Columns on walls connected to boundary 3

# %%
def process_single_walls_right(df, max_along_x, max_along_y, dxf_data, width=12, height=9, variance_threshold=0.95, tolerance=1e-3):
    '''
    (Function docstring remains the same)
    '''

    # Step 1: Filter horizontal lines using PCA
    df_filtered = filter_horizontal_lines(df, max_along_x, max_along_y, variance_threshold)

    # Step 2: Extract matching line pairs
    df_pairs = extract_matching_line_pairs(df_filtered)

    # Step 2.1: Filter pairs by line length (only proceed if length is 60 or more)
    def filter_by_length(df_pairs, min_length=60):
        valid_pairs = []
        for _, pair in df_pairs.iterrows():
            # Calculate the length of both lines in the pair
            length1 = np.sqrt((pair['line1']['X_end'] - pair['line1']['X_start'])**2 + 
                              (pair['line1']['Y_end'] - pair['line1']['Y_start'])**2)
            length2 = np.sqrt((pair['line2']['X_end'] - pair['line2']['X_start'])**2 + 
                              (pair['line2']['Y_end'] - pair['line2']['Y_start'])**2)

            # Check if both lines meet the minimum length requirement
            if length1 >= min_length and length2 >= min_length:
                valid_pairs.append(pair)

        return pd.DataFrame(valid_pairs)

    df_pairs = filter_by_length(df_pairs)

    # Step 3: Extract other X and Y coordinates for right-side walls
    def extract_other_x_and_y_start_right(df_pairs, max_along_x, tolerance=1e-6):
        target_x = max_along_x - 9  # Target X is now max_along_x - 9
        extracted_values = []
        
        for _, pair in df_pairs.iterrows():
            if np.isclose(pair['line1']['X_start'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line1'],
                    'X_other': pair['line1']['X_end'],
                    'Y_start': pair['line1']['Y_start']
                })
            elif np.isclose(pair['line1']['X_end'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line1'],
                    'X_other': pair['line1']['X_start'],
                    'Y_start': pair['line1']['Y_start']
                })

            if np.isclose(pair['line2']['X_start'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line2'],
                    'X_other': pair['line2']['X_end'],
                    'Y_start': pair['line2']['Y_start']
                })
            elif np.isclose(pair['line2']['X_end'], target_x, atol=tolerance):
                extracted_values.append({
                    'index': pair['index_line2'],
                    'X_other': pair['line2']['X_start'],
                    'Y_start': pair['line2']['Y_start']
                })
        
        if extracted_values:
            df_extracted_right = pd.DataFrame(extracted_values)
        else:
            df_extracted_right = pd.DataFrame(columns=['index', 'X_other', 'Y_start'])
        
        return df_extracted_right

    df_extracted_right = extract_other_x_and_y_start_right(df_pairs, max_along_x)

    # Step 4: Create boxes in the DXF data without overlapping
    def create_box_from_df_extracted_right(df_extracted_right, width, height, dxf_data):
        msp = dxf_data.modelspace()
        
        # Set to store existing box coordinates to prevent overlaps
        existing_boxes = set()

        # Detect nearby boxes within a 20-unit radius
        def is_near_existing_boxes(x, y, distance_threshold=96):
            for existing_x, existing_y in existing_boxes:
                if abs(existing_x - x) < distance_threshold and abs(existing_y - y) < distance_threshold:
                    return True
            return False

        for i in range(0, len(df_extracted_right), 2):
            if i + 1 < len(df_extracted_right):
                row1 = df_extracted_right.iloc[i]
                row2 = df_extracted_right.iloc[i + 1]

                x_value = row1['X_other']
                y_value = max(row1['Y_start'], row2['Y_start'])
                
                # Check if the proposed box location is clear of nearby boxes
                if is_near_existing_boxes(x_value, y_value):
                    continue  # Skip creating this box if it's too close to another

                # Check if a box already exists at these coordinates
                box_position = (x_value, y_value)
                if box_position in existing_boxes:
                    continue  # Skip creating this box if it overlaps

                # Add the new box coordinates to the set
                existing_boxes.add(box_position)

                # Define the box corners
                p1 = (x_value, y_value)
                p2 = (p1[0], p1[1] - height)
                p3 = (p1[0] + width, p2[1])
                p4 = (p3[0], p1[1])

                points = [p1, p2, p3, p4, p1]

                # Create the closed polyline for the box
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)

        # The DXF document is modified in memory; it does not need to be saved yet.

        return dxf_data

    # Call the function to add boxes to the DXF data
    dxf_data = create_box_from_df_extracted_right(df_extracted_right, width, height, dxf_data)

    # The modified dxf_data can now be used in further steps or saved when needed
    return dxf_data

# %% [markdown]
# # Step 16 - Columns on walls connected to upper boundary 

# %%
def filter_vertical_lines_by_pca(df, max_along_y, min_y=9, variance_threshold=0.95, tolerance=1e-6):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    required_columns = ['X_start', 'Y_start', 'X_end', 'Y_end']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=required_columns)
    
    def is_vertical_pca(x_start, y_start, x_end, y_end):
        line_points = np.array([[x_start, y_start], [x_end, y_end]])
        pca = PCA(n_components=2)
        pca.fit(line_points)
        return abs(pca.components_[0][1]) >= variance_threshold

    df['is_vertical'] = df.apply(lambda row: is_vertical_pca(row['X_start'], row['Y_start'], row['X_end'], row['Y_end']), axis=1)
    df_filtered = df[(df['Y_start'] >= min_y) & (df['Y_start'] <= (max_along_y - min_y)) & df['is_vertical']]
    return df_filtered.drop(columns=['is_vertical'])

def extract_specific_lines(df_filtered, max_along_x, max_along_y, min_x=9, tolerance=1e-6):
    target_y = max_along_y - 9
    df_extracted = df_filtered[
        ~((np.isclose(df_filtered['X_start'], min_x, atol=tolerance)) |
          (np.isclose(df_filtered['X_start'], max_along_x - min_x, atol=tolerance))) &
        (np.isclose(df_filtered['Y_start'], target_y, atol=tolerance) |
         np.isclose(df_filtered['Y_end'], target_y, atol=tolerance))
    ]
    return df_extracted

def calculate_line_length(row):
    return abs(row['Y_start'] - row['Y_end'])

def find_replacement_line(df_filtered, x_start, saved_y_value, tolerance=1e-6):
    matching_line = df_filtered[
        (np.isclose(df_filtered['X_start'], x_start, atol=tolerance)) &
        ((np.isclose(df_filtered['Y_start'], saved_y_value, atol=tolerance)) |
         (np.isclose(df_filtered['Y_end'], saved_y_value, atol=tolerance)))
    ]
    if not matching_line.empty:
        return matching_line.iloc[0]
    return None

def pair_lines_by_x_difference(df_extracted, df_filtered, min_x_diff=3.9, max_x_diff=5.1, max_along_y=9, tolerance=1e-6):
    matching_pairs = []
    y_coor_up = []  
    x_coor_up = []  
    target_y = max_along_y - 9

    for i, line1 in df_extracted.iterrows():
        for j, line2 in df_extracted.iterrows():
            if i >= j:
                continue
            
            x_diff = abs(line1['X_start'] - line2['X_start'])
            if min_x_diff <= x_diff <= max_x_diff:
                length1 = calculate_line_length(line1)
                length2 = calculate_line_length(line2)
                
                if not np.isclose(length1, length2, atol=tolerance):
                    if length1 < length2:
                        shorter_line = line1
                        longer_line = line2
                    else:
                        shorter_line = line2
                        longer_line = line1
                    
                    smaller_y_value = min(longer_line['Y_start'], longer_line['Y_end'])
                    matching_line = find_replacement_line(df_filtered, shorter_line['X_start'], saved_y_value=smaller_y_value)

                    if matching_line is not None:
                        replacement_length = calculate_line_length(matching_line)
                        total_shorter_length = calculate_line_length(shorter_line) + replacement_length

                        if np.isclose(total_shorter_length, calculate_line_length(longer_line), atol=tolerance):
                            y_match = (
                                np.isclose(matching_line['Y_start'], smaller_y_value, atol=tolerance) or
                                np.isclose(matching_line['Y_end'], smaller_y_value, atol=tolerance)
                            )
                            
                            if y_match:
                                y_coor_up.append(smaller_y_value)
                                x_coor_up.append(longer_line['X_start'])
                                x_coor_up.append(shorter_line['X_start'])

                matching_pairs.append({
                    'index_line1': i, 'line1': line1.to_dict(),
                    'index_line2': j, 'line2': line2.to_dict()
                })

    return pd.DataFrame(matching_pairs), y_coor_up, x_coor_up

def create_boxes_on_dxf(dxf_data, x_coor_up, y_coor_up, width=9, height=12):
    msp = dxf_data.modelspace()

    existing_boxes = set()

    for x_value, y_value in zip(x_coor_up, y_coor_up):
        box_position = (x_value, y_value)
        
        # Check for a 96-unit clearance around the box position
        if any(
            abs(x_value - ex[0]) < 96 and abs(y_value - ex[1]) < 96
            for ex in existing_boxes
        ):
            continue  # Skip creating this box if it’s too close to existing boxes

        existing_boxes.add(box_position)

        p1 = (x_value, y_value)
        p2 = (p1[0], p1[1] + height)
        p3 = (p1[0] + width, p2[1])
        p4 = (p3[0], p1[1])
        points = [p1, p2, p3, p4, p1]

        msp.add_lwpolyline(points, close=True)

        hatch = msp.add_hatch(color=1)
        hatch.paths.add_polyline_path(points, is_closed=True)

    # Return the modified dxf_data object
    return dxf_data

def single_wall_up(df, max_along_x, max_along_y, dxf_data, width=9, height=12):
    # Step 1: Filter vertical lines
    df_filtered_vertical = filter_vertical_lines_by_pca(df, max_along_y)

    # Step 2: Extract specific lines
    df_extracted_lines = extract_specific_lines(df_filtered_vertical, max_along_x, max_along_y)

    # Step 3: Pair lines based on x-difference
    df_paired_lines, y_coor_up, x_coor_up = pair_lines_by_x_difference(df_extracted_lines, df_filtered_vertical, max_along_y=max_along_y)

    # Step 4: Create boxes on the DXF data
    dxf_data = create_boxes_on_dxf(dxf_data, x_coor_up, y_coor_up, width, height)

    # Return the modified DXF data object
    return dxf_data

# %% [markdown]
# # Step 17 - Helper function

# %%
def filter_lines_by_pca(df, max_along_x, max_along_y, variance_threshold=0.95, tolerance=1):
    '''
    Upstream Libraries:
    - numpy (for numerical operations)
    - pandas (for handling DataFrame)
    - sklearn.decomposition.PCA (for principal component analysis)
    
    Function Description:
    This function filters both horizontal and vertical lines from the input DataFrame by applying PCA 
    and ensuring the lines fall within specified x and y coordinate ranges. Lines are categorized as 
    either 'Horizontal' or 'Vertical'.

    Parameters:
    - df: Input DataFrame containing line data.
    - max_along_x: Maximum x-coordinate value (float).
    - max_along_y: Maximum y-coordinate value (float).
    - variance_threshold: Threshold (float) for PCA component value to consider a line as horizontal or vertical. Default is 0.95.
    - tolerance: Tolerance value for filtering (default is 1).
    
    Returns:
    - A DataFrame containing filtered horizontal and vertical lines with an added 'Line_Type' column.
    '''
    
    # Drop rows with NaN values
    df = df.dropna(subset=['X_start', 'Y_start', 'X_end', 'Y_end'])

    # Function to determine line orientation based on PCA
    def get_line_orientation(x_start, y_start, x_end, y_end):
        line_points = np.array([[x_start, y_start], [x_end, y_end]])
        pca = PCA(n_components=2)
        pca.fit(line_points)

        # Check if line is horizontal or vertical based on PCA
        if abs(pca.components_[0][0]) >= variance_threshold:  # X-axis dominant
            return 'Horizontal'
        elif abs(pca.components_[0][1]) >= variance_threshold:  # Y-axis dominant
            return 'Vertical'
        return None  # Not strictly horizontal or vertical

    # Apply orientation check
    df['Line_Type'] = df.apply(
        lambda row: get_line_orientation(row['X_start'], row['Y_start'], row['X_end'], row['Y_end']),
        axis=1
    )

    # Filter DataFrame for lines within the specified ranges
    filtered_df = df[
        ((df['Line_Type'] == 'Horizontal') & 
         (df['X_start'] >= 0) & (df['X_end'] <= max_along_x) & 
         (df['Y_start'] >= (9 - tolerance)) & (df['Y_start'] <= (max_along_y - 9 + tolerance))) |
        ((df['Line_Type'] == 'Vertical') & 
         (df['X_start'] >= (9 - tolerance)) & (df['X_end'] <= (max_along_x - 9 + tolerance)) & 
         (df['Y_start'] >= 0) & (df['Y_end'] <= max_along_y))
    ]


    # Remove vertical lines where X_start or X_end is 9 or (max_along_x - 9) with tolerance
    vertical_filter = (
        (filtered_df['Line_Type'] == 'Vertical') &
        ((filtered_df['X_start'].between(9 - tolerance, 9 + tolerance)) |
         (filtered_df['X_end'].between(max_along_x - 9 - tolerance, max_along_x - 9 + tolerance)))
    )
    
    filtered_df = filtered_df[~vertical_filter]

    # Remove horizontal lines where Y_start or Y_end is 9 or (max_along_y - 9) with tolerance
    horizontal_filter = (
        (filtered_df['Line_Type'] == 'Horizontal') &
        ((filtered_df['Y_start'].between(9 - tolerance, 9 + tolerance)) |
         (filtered_df['Y_end'].between(max_along_y - 9 - tolerance, max_along_y - 9 + tolerance)))
    )
    
    filtered_df = filtered_df[~horizontal_filter]

    return filtered_df.dropna(subset=['Line_Type'])  # Return filtered DataFrame without NaNs in Line_Type

# %%
def extract_matching_horizontal_pairs(df, max_along_x): 
    '''
    Upstream Libraries:
    - numpy (for numerical operations)
    - pandas (for handling DataFrame)
    
    Upstream Functions needed:
    None
    
    Function Description:
    This function extracts matching pairs of horizontal lines from the input DataFrame. It filters horizontal lines, checks if
    either X_start or X_end is close to 9 or max_along_x - 9 with a tolerance of 0.5, ensures that the lines have similar 
    starting and ending points, that the vertical distance is either 5 or 9 with a tolerance of 1, and ensures that lines 
    with the layer name 'Entrance_Staircase' are not paired with each other.

    Parameters:
    - df: Input DataFrame containing filtered line data.
    - max_along_x: The maximum x-coordinate for filtering lines based on X_start or X_end.
    
    Returns:
    - A DataFrame containing matching horizontal line pairs.
    '''
    # Step 1: Filter out all horizontal lines
    horizontal_lines = df[np.isclose(df['Y_start'], df['Y_end'])]
    
    # Step 2: Extract lines where either X_start or X_end is close to 9 or max_along_x - 9 with a tolerance of 0.5
    filtered_lines = horizontal_lines[
        (np.isclose(horizontal_lines['X_start'], 9, atol=0.5) | 
         np.isclose(horizontal_lines['X_end'], 9, atol=0.5) |
         np.isclose(horizontal_lines['X_start'], max_along_x - 9, atol=0.5) | 
         np.isclose(horizontal_lines['X_end'], max_along_x - 9, atol=0.5))
    ]

    matching_pairs = []
    
    # Step 3: Loop through each line pair without re-checking previous pairs
    for i in range(len(filtered_lines) - 1):
        line1 = filtered_lines.iloc[i]
        
        for j in range(i + 1, len(filtered_lines)):
            line2 = filtered_lines.iloc[j]
            
            # Step 4: Check if lines have similar starting and ending points
            if (
                (np.isclose(line1['X_start'], line2['X_start']) and np.isclose(line1['X_end'], line2['X_end'])) or
                (np.isclose(line1['X_start'], line2['X_end']) and np.isclose(line1['X_end'], line2['X_start']))
            ):
                # Step 5: Check if the vertical distance is either 5 or 9 with a tolerance of 1
                vertical_distance = abs(line1['Y_start'] - line2['Y_start'])
                if 4 <= vertical_distance <= 6 or 8 <= vertical_distance <= 10:
                    # Step 6: Ensure 'Entrance_Staircase' lines are not paired with each other
                    if not (line1['Layer'] == 'Entrance_Staircase' and line2['Layer'] == 'Entrance_Staircase'):
                        # Append matching pairs
                        matching_pairs.append({
                            'index_line1': line1.name, 'line1': line1.to_dict(),
                            'index_line2': line2.name, 'line2': line2.to_dict()
                        })
    
    return pd.DataFrame(matching_pairs)

# %%
def extract_matching_vertical_pairs(df, max_along_y): 
    '''
    Upstream Libraries:
    - numpy (for numerical operations)
    - pandas (for handling DataFrame)
    
    Upstream Functions needed:
    None
    
    Function Description:
    This function extracts matching pairs of vertical lines from the input DataFrame. It filters vertical lines, checks if
    either Y_start or Y_end is close to 9 or max_along_y - 9 with a tolerance of 0.5, ensures that the lines have similar 
    starting and ending points, checks if the horizontal distance between them is either 5 or 9 with a tolerance of 1, 
    and ensures that lines with the layer name 'Entrance_Staircase' are not paired with each other.

    Parameters:
    - df: Input DataFrame containing filtered line data.
    - max_along_y: The maximum y-coordinate for filtering lines based on Y_start or Y_end.
    
    Returns:
    - A DataFrame containing matching vertical line pairs.
    '''
    # Step 1: Filter out all vertical lines
    vertical_lines = df[np.isclose(df['X_start'], df['X_end'])]
    
    # Step 2: Extract lines where either Y_start or Y_end is close to 9 or max_along_y - 9 with a tolerance of 0.5
    filtered_lines = vertical_lines[
        (np.isclose(vertical_lines['Y_start'], 9, atol=0.5) | 
         np.isclose(vertical_lines['Y_end'], 9, atol=0.5) |
         np.isclose(vertical_lines['Y_start'], max_along_y - 9, atol=0.5) | 
         np.isclose(vertical_lines['Y_end'], max_along_y - 9, atol=0.5))
    ]

    matching_pairs = []
    
    # Step 3: Loop through each line pair without re-checking previous pairs
    for i in range(len(filtered_lines) - 1):
        line1 = filtered_lines.iloc[i]
        
        for j in range(i + 1, len(filtered_lines)):
            line2 = filtered_lines.iloc[j]
            
            # Step 4: Check if lines have similar starting and ending points
            if (
                (np.isclose(line1['Y_start'], line2['Y_start']) and np.isclose(line1['Y_end'], line2['Y_end'])) or
                (np.isclose(line1['Y_start'], line2['Y_end']) and np.isclose(line1['Y_end'], line2['Y_start']))
            ):
                # Step 5: Check if horizontal distance is 5 or 9 with a tolerance of 1
                horizontal_distance = abs(line1['X_start'] - line2['X_start'])
                if 4 <= horizontal_distance <= 6 or 8 <= horizontal_distance <= 10:
                    # Step 6: Ensure 'Entrance_Staircase' lines are not paired with each other
                    if not (line1['Layer'] == 'Entrance_Staircase' and line2['Layer'] == 'Entrance_Staircase'):
                        # Append matching pairs
                        matching_pairs.append({
                            'index_line1': line1.name, 'line1': line1.to_dict(),
                            'index_line2': line2.name, 'line2': line2.to_dict()
                        })
    
    return pd.DataFrame(matching_pairs)

# %%
def filter_dataframe(filtered_lines, match_pair_vertical_df, match_pair_horizontal_df):
    """
    Filters out rows from `filtered_lines` based on indices found in `match_pair_vertical_df`
    and `match_pair_horizontal_df`.

    Parameters:
    - filtered_lines: The original DataFrame to filter.
    - match_pair_vertical_df: DataFrame with vertical matching pairs.
    - match_pair_horizontal_df: DataFrame with horizontal matching pairs.

    Returns:
    - filtered_df: A filtered DataFrame with rows removed where indices match those in `match_pair_vertical_df` and `match_pair_horizontal_df`.
    """
    # Check if the columns exist in match_pair_vertical_df
    if 'index_line1' in match_pair_vertical_df.columns and 'index_line2' in match_pair_vertical_df.columns:
        vertical_indices = match_pair_vertical_df[['index_line1', 'index_line2']].values.flatten()
    else:
        vertical_indices = []  # No vertical indices to filter if columns are missing

    # Check if the columns exist in match_pair_horizontal_df
    if 'index_line1' in match_pair_horizontal_df.columns and 'index_line2' in match_pair_horizontal_df.columns:
        horizontal_indices = match_pair_horizontal_df[['index_line1', 'index_line2']].values.flatten()
    else:
        horizontal_indices = []  # No horizontal indices to filter if columns are missing

    # Combine all indices and remove duplicates
    indices_to_remove = set(vertical_indices).union(horizontal_indices)

    # Filter out rows in filtered_lines based on indices
    filtered_df = filtered_lines.drop(indices_to_remove, errors='ignore')
    
    return filtered_df

# %%
def remove_specific_layer_rows(df):
    """
    Removes rows from the DataFrame based on specific layer names and returns the remaining rows.

    Parameters:
    - df : Input DataFrame containing line data, including a 'Layer' column.

    Returns:
    - remaining_df: A DataFrame containing rows that do not match the specified layer names.
    """
    
    # Define the list of layers to remove
    layers_to_remove = ["Entrance_Staircase", "Door3fPanel", "Door2.6fPanel", "Door4fPanel", "Staircase","Staircase_innerwall"]
    
    # Filter out rows with specified layers
    remaining_df = df[~df['Layer'].isin(layers_to_remove)]
    
    return remaining_df

# %%
def find_intersections(df, tolerance=1e-6):
    intersections = []
    
    # Separate horizontal and vertical lines based on the 'Type' column
    horizontal_lines = df[df['Line_Type'] == 'Horizontal']
    vertical_lines = df[df['Line_Type'] == 'Vertical']

    for _, h_row in horizontal_lines.iterrows():
        h_x_start, h_y_start, h_x_end, h_y_end = h_row['X_start'], h_row['Y_start'], h_row['X_end'], h_row['Y_end']
        
        for _, v_row in vertical_lines.iterrows():
            v_x_start, v_y_start, v_x_end, v_y_end = v_row['X_start'], v_row['Y_start'], v_row['X_end'], v_row['Y_end']
            
            # Check for intersection: horizontal line's Y in vertical line's range and vertical line's X in horizontal line's range
            if (min(h_x_start, h_x_end) - tolerance <= v_x_start <= max(h_x_start, h_x_end) + tolerance) and \
               (min(v_y_start, v_y_end) - tolerance <= h_y_start <= max(v_y_start, v_y_end) + tolerance):
                intersections.append((v_x_start, h_y_start))

    return intersections

# %%
def find_and_separate_points_in_range(intersection_df, range_threshold=9.5):
    """
    Identifies pairs of points within a specified range in both X and Y directions,
    and separates them based on equal X, equal Y, or unequal X and Y coordinates.
    
    Parameters:
    - intersection_df (pd.DataFrame): DataFrame containing 'X' and 'Y' columns with point coordinates.
    - range_threshold (float): The maximum allowed distance along both X and Y axes for points to be considered within range.
    
    Returns:
    - tuple: A tuple containing three DataFrames:
        - df_x_equal: Pairs with equal X values.
        - df_y_equal: Pairs with equal Y values.
        - df_other: Pairs where both X and Y values differ.
    """
    
    points_in_range = []
    
    # Iterate over each point to find pairs within the specified range
    for i in range(len(intersection_df) - 1):
        point = intersection_df.iloc[i]
        
        for j in range(i + 1, len(intersection_df)):
            other_point = intersection_df.iloc[j]
            
            # Calculate distances along X and Y
            x_dist = abs(point['X'] - other_point['X'])
            y_dist = abs(point['Y'] - other_point['Y'])
            
            # Check if distances are within the threshold
            if x_dist <= range_threshold and y_dist <= range_threshold:
                points_in_range.append((point['X'], point['Y'], other_point['X'], other_point['Y']))
    
    # Convert results into a DataFrame
    points_in_range_df = pd.DataFrame(points_in_range, columns=['X1', 'Y1', 'X2', 'Y2'])
    
    # Separate based on conditions
    df_x_equal = points_in_range_df[points_in_range_df['X1'] == points_in_range_df['X2']]
    df_y_equal = points_in_range_df[points_in_range_df['Y1'] == points_in_range_df['Y2']]
    df_other = points_in_range_df[(points_in_range_df['X1'] != points_in_range_df['X2']) & 
                                  (points_in_range_df['Y1'] != points_in_range_df['Y2'])]
    
    return df_x_equal, df_y_equal, df_other

# %% [markdown]
# # Step 18 - extract dataframes for intersecting points 

# %%
def semi_main_columns(df1, max_along_x, max_along_y):
    # Step 1: Filter lines based on PCA and coordinate ranges
    filtered_lines = filter_lines_by_pca(df1, max_along_x, max_along_y)
    
    # Step 2: Extract matching horizontal pairs
    match_pair_horizontal_df = extract_matching_horizontal_pairs(filtered_lines, max_along_x)
    
    # Step 3: Extract matching vertical pairs
    match_pair_vertical_df = extract_matching_vertical_pairs(filtered_lines, max_along_y)
    
    # Step 4: Filter DataFrame based on matching pairs
    filtered_df = filter_dataframe(filtered_lines, match_pair_vertical_df, match_pair_horizontal_df)
    
    # Step 5: Remove specific layer rows from the DataFrame
    new_filtered_df = remove_specific_layer_rows(filtered_df)
    
    # Step 6: Find intersections between horizontal and vertical lines
    intersection_points = find_intersections(new_filtered_df)
    intersection_df = pd.DataFrame(intersection_points, columns=['X', 'Y'])
    
    # Step 7: Find and separate points within a specified range
    df_x_equal, df_y_equal, df_other = find_and_separate_points_in_range(intersection_df, range_threshold=9.5)
    
    # Additional filtered subsets
    temp_h = new_filtered_df[new_filtered_df['Line_Type'] == 'Horizontal']
    temp_v = new_filtered_df[new_filtered_df['Line_Type'] == 'Vertical']
    
    # Returning all relevant DataFrames for further inspection or use
    return {
        "df_x_equal": df_x_equal,
        "df_y_equal": df_y_equal,
        "df_other": df_other,
        "temp_h": temp_h,
        "temp_v": temp_v
    }

# %% [markdown]
# # Step 19 - Columns on intersections-1 

# %%
def create_boxes_in_df_x_equal(df_x_equal, temp_v, dxf_data, width=9, height=12, tolerance_v=0.5, radius=12):
    """
    Creates boxes in the DXF data based on matching conditions in df_x_equal and temp_v, and checks for overlapping.
    """
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()

    # Track created box positions to prevent overlapping
    created_boxes = []

    for idx, row in df_x_equal.iterrows():
        X1, X2, Y1, Y2 = row['X1'], row['X2'], row['Y1'], row.get('Y2', None)
        match_found = False
        box_points = []

        # Step 1: Check X1 +5 and X1 +9 matches in `temp_v`
        for offset in [5, 9]:
            for _, temp_v_row in temp_v.iterrows():
                X_start_temp = temp_v_row['X_start']
                if abs((X1 + offset) - X_start_temp) <= tolerance_v:
                    match_found = True
                    Y = min(Y1, Y2)
                    X = X1
                    box_points = [
                        (X, Y),
                        (X + width, Y),
                        (X + width, Y + height),
                        (X, Y + height)
                    ]
                    break
            if match_found:
                break

        # Step 2: Check X1 -5 and X1 -9 matches in `temp_v` if Step 1 did not pass
        if not match_found:
            for offset in [-5, -9]:
                for _, temp_v_row in temp_v.iterrows():
                    X_start_temp = temp_v_row['X_start']
                    if abs((X1 + offset) - X_start_temp) <= tolerance_v:
                        match_found = True
                        Y = min(Y1, Y2)
                        X = X1
                        box_points = [
                            (X, Y),
                            (X - width, Y),
                            (X - width, Y + height),
                            (X, Y + height)
                        ]
                        break
                if match_found:
                    break

        # Only create the box if a match was found and no overlapping occurs
        if match_found and box_points:
            center_point = (box_points[0][0] + width / 2, box_points[0][1] + height / 2)
            
            # Check for nearby boxes within the radius
            too_close = False
            for existing_box in created_boxes:
                existing_center = (
                    (existing_box[0][0] + existing_box[2][0]) / 2,
                    (existing_box[0][1] + existing_box[2][1]) / 2
                )
                distance = math.sqrt((center_point[0] - existing_center[0])**2 + (center_point[1] - existing_center[1])**2)
                if distance < radius:
                    too_close = True
                    break

            if not too_close:
                # Add the box to the DXF data (in-memory manipulation)
                points = box_points + [box_points[0]]  # Closing the loop for polyline
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
                
                # Add box coordinates to created_boxes list
                created_boxes.append(box_points)

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

# %% [markdown]
# # Step 20 - Columns on intersections-2

# %%
def create_boxes_in_df_y_equal(df_y_equal, temp_h, dxf_data, tolerance_h=0.5, width=12, height=9, radius=12):
    """
    Creates boxes in the DXF data based on matching conditions in df_y_equal and temp_h, and prevents overlapping.
    """
    if not isinstance(dxf_data, ezdxf.document.Drawing):
        raise TypeError("dxf_data should be an instance of ezdxf.document.Drawing")
    
    msp = dxf_data.modelspace()

    # List to store coordinates of created boxes to prevent overlapping
    created_boxes = []

    # Iterate over each row in df_y_equal
    for idx, row in df_y_equal.iterrows():
        Y1, X1, X2 = row['Y1'], row['X1'], row['X2']
        match_found = False
        box_points = []

        # Check Y1 +5 and Y1 +9 for matches in `temp_h`
        for offset in [5, 9]:
            for _, temp_h_row in temp_h.iterrows():
                Y_start_temp = temp_h_row['Y_start']
                if abs((Y1 + offset) - Y_start_temp) <= tolerance_h:
                    match_found = True
                    X = min(X1, X2)
                    Y = Y1
                    box_points = [
                        (X, Y),
                        (X + width, Y),
                        (X + width, Y + height),
                        (X, Y + height)
                    ]
                    break
            if match_found:
                break

        # Check Y1 -5 and Y1 -9 for matches in `temp_h` if no match was found in the positive offset check
        if not match_found:
            for offset in [-5, -9]:
                for _, temp_h_row in temp_h.iterrows():
                    Y_start_temp = temp_h_row['Y_start']
                    if abs((Y1 + offset) - Y_start_temp) <= tolerance_h:
                        match_found = True
                        X = min(X1, X2)
                        Y = Y1
                        box_points = [
                            (X, Y),
                            (X + width, Y),
                            (X + width, Y - height),
                            (X, Y - height)
                        ]
                        break
                if match_found:
                    break

        # Only create the box if a match was found and no overlapping occurs
        if match_found and box_points:
            center_point = (box_points[0][0] + width / 2, box_points[0][1] + height / 2)
            
            # Check for overlap with existing boxes within a radius of 12 units
            too_close = False
            for existing_box in created_boxes:
                existing_center = (
                    (existing_box[0][0] + existing_box[2][0]) / 2,
                    (existing_box[0][1] + existing_box[2][1]) / 2
                )
                distance = math.sqrt((center_point[0] - existing_center[0])**2 + (center_point[1] - existing_center[1])**2)
                if distance < radius:
                    too_close = True
                    break

            if not too_close:
                # Add the box to the DXF data (in-memory manipulation)
                points = box_points + [box_points[0]]  # Closing the loop for polyline
                msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
                
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
                
                # Add box coordinates to created_boxes list
                created_boxes.append(box_points)

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

# %% [markdown]
# # Step 21- extracting group A and group B 

# %%
def group_by_x(df_other):
    # Create empty DataFrames for the groups
    df_other_groupA = pd.DataFrame(columns=df_other.columns)
    df_other_groupB = pd.DataFrame(columns=df_other.columns)

    # Iterate over each row in `df_other`
    for idx, row in df_other.iterrows():
        # Step 1 & 2: Determine the primary Y and corresponding primary X
        if row['Y1'] > row['Y2']:
            primary_x = row['X1']
            other_x = row['X2']
        else:
            primary_x = row['X2']
            other_x = row['X1']
        
        # Step 4 & 5: Compare primary_x with other_x and assign to appropriate group
        if primary_x > other_x:
            df_other_groupA = pd.concat([df_other_groupA, row.to_frame().T])
        else:
            df_other_groupB = pd.concat([df_other_groupB, row.to_frame().T])

    # Reset index for the grouped DataFrames if needed
    df_other_groupA.reset_index(drop=True, inplace=True)
    df_other_groupB.reset_index(drop=True, inplace=True)

    return df_other_groupA, df_other_groupB

# %% [markdown]
# # Step 22 - Columns on intersections-3

# %%
def create_boxes_in_df_other_groupA(df_other_groupA, width, height, dxf_data, radius=12):
    """
    Adds boxes to the DXF data based on coordinates in df_other_groupA, ensuring no overlap within a specified radius.

    Parameters:
    - df_other_groupA (DataFrame): DataFrame containing X1, Y1, X2, Y2 columns for box placement.
    - width (float): Width of each box.
    - height (float): Height of each box.
    - dxf_data (ezdxf.document): The DXF data object to modify.
    - radius (float): The minimum radius to check for overlap before adding a new box (default is 12).
    
    Returns:
    - The modified dxf_data object.
    """
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()

    # List to store the center coordinates of created boxes to prevent overlapping
    created_boxes = []

    # Iterate over each row in df_other_groupA
    for _, row in df_other_groupA.iterrows():
        X1, Y1, X2, Y2 = row['X1'], row['Y1'], row['X2'], row['Y2']

        # Calculate the start point and corners of the box
        X = min(X1, X2)
        Y = min(Y1, Y2)
        start_point = (X, Y)

        # Define the corners of the box
        p1 = start_point
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] + height)
        p4 = (p1[0], p3[1])

        # Calculate the center of the current box
        box_center = ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2)

        # Check for overlap with existing boxes within the specified radius
        overlapping = False
        for existing_center in created_boxes:
            distance = sqrt((box_center[0] - existing_center[0])**2 + (box_center[1] - existing_center[1])**2)
            if distance < radius:
                overlapping = True
                break

        # Only create the box if no overlap within the radius
        if not overlapping:
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

            # Add a hatch to fill the polyline with red color
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Add the box center to created_boxes list to track position
            created_boxes.append(box_center)

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

# %% [markdown]
# # Step 23 - Columns on intersections-4

# %%
def create_boxes_in_df_other_groupB(df_other_groupB, width, height, dxf_data, radius=12):
    """
    Adds boxes to the DXF data based on coordinates in df_other_groupB, ensuring no overlap within a specified radius.

    Parameters:
    - df_other_groupB (DataFrame): DataFrame containing X1, Y1, X2, Y2 columns for box placement.
    - width (float): Width of each box.
    - height (float): Height of each box.
    - dxf_data (ezdxf.document): The DXF data object to modify.
    - radius (float): The minimum radius to check for overlap before adding a new box (default is 12).
    
    Returns:
    - The modified dxf_data object.
    """
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()

    # List to store the center coordinates of created boxes to prevent overlapping
    created_boxes = []

    # Iterate over each row in df_other_groupB
    for _, row in df_other_groupB.iterrows():
        X1, Y1, X2, Y2 = row['X1'], row['Y1'], row['X2'], row['Y2']
        
        # Calculate the start point and corners of the box
        X = min(X1, X2)
        Y = max(Y1, Y2)
        start_point = (X, Y)

        # Define the corners of the box
        p1 = start_point
        p2 = (p1[0] + width, p1[1])
        p3 = (p2[0], p1[1] - height)
        p4 = (p1[0], p3[1])

        # Calculate the center of the new box
        box_center = ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2)

        # Check for overlap within a radius of 12 units
        overlapping = False
        for existing_box in created_boxes:
            existing_center = existing_box['center']
            distance = math.sqrt((box_center[0] - existing_center[0])**2 + (box_center[1] - existing_center[1])**2)
            if distance < radius:
                overlapping = True
                break

        # Only create the box if no overlap within the 12-unit radius
        if not overlapping:
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

            # Add a hatch to fill the polyline with red color
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Add the box coordinates and center to created_boxes list
            created_boxes.append({'points': [p1, p2, p3, p4], 'center': box_center})

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

# %% [markdown]
# # Step 24 - Detect and label boxes

# %%
def detect_and_label_boxes(dxf_data, label_position='right', offset=1, text_height=7, text_color=0, shift=0):
    '''
    Detects boxes (closed polylines) in a DXF file, labels them around the box (right, left, or top),
    and updates the DXF data in memory.
    '''
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()
    box_number = 1
    
    # Create 'column_text' layer if it doesn't exist
    if 'column_text' not in msp.doc.layers:
        msp.doc.layers.add('column_text')
    
    # Iterate over all entities in the modelspace
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
            points = entity.get_points('xy')  # Get points as (x, y) tuples
            
            if len(points) == 5 and points[0] == points[-1]:
                # Calculate the corners of the box
                min_x = min([p[0] for p in points])
                max_x = max([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_y = max([p[1] for p in points])
                
                # Determine where to place the label
                if label_position == 'right':
                    label_x = max_x + offset + shift
                    label_y = (min_y + max_y) / 2
                elif label_position == 'left':
                    label_x = min_x - offset - shift
                    label_y = (min_y + max_y) / 2
                elif label_position == 'top':
                    label_x = (min_x + max_x) / 2
                    label_y = max_y + offset
                else:
                    raise ValueError("label_position must be 'right', 'left', or 'top'")
                
                box_name = f"C{box_number}"
                box_number += 1
                
                # Create mtext directly with all attributes
                mtext = msp.add_mtext(box_name, dxfattribs={
                    'char_height': text_height,
                    'color': text_color,
                    'layer': 'column_text'
                })
                mtext.set_location((label_x, label_y))
    
    return dxf_data

# %% [markdown]
# # Step 25 - Detect and remove overlapping columns

# %%
def detect_and_remove_overlapping_columns(dxf_data, tolerance=5):
    """
    Detects overlapping columns (boxes) based on their center positions and removes the overlapping ones.
    
    Parameters:
    (1) dxf_data: The DXF data object to modify (ezdxf.document).
    (2) tolerance: The distance tolerance within which columns (boxes) are considered overlapping (float, default is 5).
    
    Returns:
    dxf_data: The modified DXF data object with overlapping columns removed.
    """
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()

    # Initializing lists to store box properties
    column_names = []
    x_centers = []
    y_centers = []
    lengths = []
    widths = []
    entities = []  # Store entity references for deletion

    # Collect all labels that start with "C" (e.g., "C1", "C2", etc.)
    for entity in msp.query('MTEXT TEXT'):
        if entity.dxftype() in ['MTEXT', 'TEXT']:
            label = entity.dxf.text.strip()
            if label.startswith('C'):
                column_names.append(label)
                entities.append(entity)  # Save entity reference
                
                # Assuming `entity.dxf.insert` represents the starting point
                start_x, start_y = entity.dxf.insert.x, entity.dxf.insert.y

                # Define corners for the box based on the start point
                p1 = (start_x, start_y)
                p2 = (p1[0] + 12, p1[1])   # Default box width of 12
                p3 = (p2[0], p1[1] - 9)    # Default box height of 9
                p4 = (p1[0], p3[1])

                # Calculate the center of the box
                x_center = (p1[0] + p3[0]) / 2
                y_center = (p1[1] + p3[1]) / 2
                x_centers.append(x_center)
                y_centers.append(y_center)
                lengths.append(abs(p2[0] - p1[0]))
                widths.append(abs(p1[1] - p3[1]))

    # Create a DataFrame with the box properties
    df = pd.DataFrame({
        'Column': column_names,
        'X Center': x_centers,
        'Y Center': y_centers,
        'Width': lengths,
        'Height': widths,
        'Entity': entities  # Store entity reference for deletion
    })

    # Detect and remove overlapping boxes
    to_remove = set()
    for i, (x1, y1, name1, entity1) in df[['X Center', 'Y Center', 'Column', 'Entity']].iterrows():
        for j, (x2, y2, name2, entity2) in df[['X Center', 'Y Center', 'Column', 'Entity']].iterrows():
            if i < j and j not in to_remove:  # Avoid duplicate pairs and self-comparison
                if abs(x1 - x2) <= tolerance and abs(y1 - y2) <= tolerance:
                    # Mark the second box (entity2) for removal
                    to_remove.add(j)
                    msp.delete_entity(entity2)

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

# %% [markdown]
# # Step 26 - Create column schedule dataframe

# %%
def create_column_schedule_dataframe(dxf_data, max_along_x, max_along_y):
    """
    Creates a DataFrame for 'Schedule of Column at Ground Floor' with column names,
    calculating actual length, width, and center coordinates from the drawing.
    
    Parameters:
        dxf_data: DXF data object containing the modelspace entities.
        max_along_x (float): Maximum x-coordinate value in the drawing.
        max_along_y (float): Maximum y-coordinate value in the drawing.
    
    Returns:
        DataFrame: A pandas DataFrame with the column schedule.
    """
    
    msp = dxf_data.modelspace()
    column_names = []
    x_centers = []
    y_centers = []
    lengths = []
    widths = []
    
    # Dictionary to store column labels and their coordinates
    column_positions = {}
    
    # First collect all column labels
    for entity in msp.query('MTEXT TEXT'):
        if entity.dxftype() in ['MTEXT', 'TEXT']:
            label = entity.dxf.text.strip()
            if label.startswith('C'):
                x_pos = entity.dxf.insert.x
                y_pos = entity.dxf.insert.y
                column_positions[label] = (x_pos, y_pos)
    
    # For each column label, find the nearest geometric elements
    for label, (label_x, label_y) in column_positions.items():
        # Find all nearby geometric entities
        nearby_coords = []
        search_radius = 15  # Increased search radius to find nearby entities
        
        for entity in msp.query('LINE LWPOLYLINE CIRCLE'):
            if entity.dxftype() == 'LINE':
                x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                nearby_coords.extend([(x1, y1), (x2, y2)])
            elif entity.dxftype() == 'LWPOLYLINE':
                points = entity.get_points()
                for point in points:
                    nearby_coords.append((point[0], point[1]))
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                nearby_coords.append((center.x, center.y))
                # Add points at cardinal directions to better capture the circle's extent
                nearby_coords.extend([
                    (center.x + radius, center.y),
                    (center.x - radius, center.y),
                    (center.x, center.y + radius),
                    (center.x, center.y - radius)
                ])
        
        if nearby_coords:
            # Calculate bounds from nearby coordinates
            x_coords = [x for x, y in nearby_coords]
            y_coords = [y for x, y in nearby_coords]
            
            # Filter coordinates to only include those within search radius of label
            filtered_x_coords = []
            filtered_y_coords = []
            for x, y in zip(x_coords, y_coords):
                distance = ((x - label_x) ** 2 + (y - label_y) ** 2) ** 0.5
                if distance <= search_radius:
                    filtered_x_coords.append(x)
                    filtered_y_coords.append(y)
            
            if filtered_x_coords and filtered_y_coords:
                min_x, max_x = min(filtered_x_coords), max(filtered_x_coords)
                min_y, max_y = min(filtered_y_coords), max(filtered_y_coords)
                
                width = abs(max_x - min_x)
                length = abs(max_y - min_y)
                x_center = (min_x + max_x) / 2
                y_center = (min_y + max_y) / 2
                
                # Only add if we found reasonable dimensions
                if width > 0 and length > 0:
                    column_names.append(label)
                    lengths.append(length)
                    widths.append(width)
                    x_centers.append(x_center)
                    y_centers.append(y_center)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'Column Nos': column_names,
        'Length': lengths,
        'Width': widths,
        'X Center': x_centers,
        'Y Center': y_centers
    })
    
    return df

# %% [markdown]
# # Step 27 - Add boxes on floors using dataframe 

# %%
def add_mtext(msp, text, position, height=2, color=0):
    '''
    Adds MText (multiline text) to the DXF file at the specified position with adjustable height and color.
    Parameters:
    (1) msp: The modelspace of the DXF document (ezdxf.modelspace).
    (2) text: The text to add (string).
    (3) position: The position to place the text (tuple of (x, y)).
    (4) height: The height of the text (float, default is 2, increased from 1).
    (5) color: The color of the text (DXF color index, default is 0 - black).
    Returns:
    None: This function does not return any values.
    '''
    mtext = msp.add_mtext(text, dxfattribs={'char_height': height, 'color': color, 'layer': 'column_text'})
    mtext.set_location(position)

def add_boxes_from_dataframe_with_hatch(dxf_data, column_info_df, 
                                      label_position='right', 
                                      offset=2, 
                                      shift=0,
                                      text_height=7,  # Increased from 5 to 7 for larger column labels
                                      text_color=0):  # Changed text color to black
    """
    Add boxes with hatching and labels to the DXF drawing based on dataframe information.
    
    Parameters:
    dxf_data: ezdxf drawing object
    column_info_df: DataFrame with columns ['X Center', 'Y Center', 'Width', 'Length', 'Label']
    label_position: string, one of 'right', 'left', or 'top'
    offset: float, distance between box and label
    shift: float, additional shift for label position
    text_height: float, height of the label text (default increased to 7)
    text_color: int, DXF color index for the label (default changed to 0 - black)
    """
    msp = dxf_data.modelspace()
    
    for index, row in column_info_df.iterrows():
        x_center = row['X Center']
        y_center = row['Y Center']
        length = row['Width']
        width = row['Length']
        label = row['Column Nos']  # Get the label from the dataframe
        
        # Calculate box corners
        half_length = length / 2
        half_width = width / 2
        min_x = x_center - half_length
        max_x = x_center + half_length
        min_y = y_center - half_width
        max_y = y_center + half_width
        
        points = [
            (min_x, min_y),  # bottom_left
            (max_x, min_y),  # bottom_right
            (max_x, max_y),  # top_right
            (min_x, max_y)   # top_left
        ]
        
        # Add polyline and hatch
        msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})
        hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Color 1 is red in DXF color index
        hatch.paths.add_polyline_path(points, is_closed=True)
        
        # Determine label position
        if label_position == 'right':
            label_x = max_x + offset + shift
            label_y = (min_y + max_y) / 2
        elif label_position == 'left':
            label_x = min_x - offset - shift
            label_y = (min_y + max_y) / 2
        elif label_position == 'top':
            label_x = (min_x + max_x) / 2
            label_y = max_y + offset
        else:
            raise ValueError("label_position must be 'right', 'left', or 'top'")
        
        # Add label with the updated text settings
        add_mtext(msp, label, (label_x, label_y), text_height, text_color)
    
    return dxf_data

# %% [markdown]
# # Step 28 - Change line colour to gray

# %%
def change_line_color_to_light_gray(dxf_data):
    """
    Changes the color of all lines in a DXF document (in memory) to extreme light gray.

    Parameters:
        dxf_data (ezdxf.document): The loaded DXF document in memory.

    Returns:
        dxf_data (ezdxf.document): The modified DXF document with updated line colors.
    """
    # Set the light gray color using RGB values 
    R, G, B = 128, 128, 128
    light_gray_color = (R << 16) | (G << 8) | B  # Convert RGB to a single integer

    # Modify the color of all lines in the modelspace
    msp = dxf_data.modelspace()

    for entity in msp:
        if entity.dxftype() == 'LINE':
            # Set the color of the line to light gray (RGB value)
            entity.dxf.color = 256  # 256 corresponds to "BYLAYER" which will take the layer's color
            entity.dxf.true_color = light_gray_color  # Set the true color to light gray

    # Return the modified dxf_data object
    return dxf_data

# %% [markdown]
# # Step 29 - extract C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col

# %%
def extract_columns(dxf_data):
    """
    Extract columns of vertically and horizontally aligned boxes based on a DXF document (in memory).
    Outputs four DataFrames: C1_ver_col, C2_hor_col, C3_ver_col, and C4_hor_col.

    Parameters:
        dxf_data (ezdxf.document): The loaded DXF document in memory.

    Returns:
        tuple: A tuple containing four DataFrames (C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col).
    """
    msp = dxf_data.modelspace()
    
    # Extract boxes and centers
    labels = {}
    centers = {}

    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(point[0], point[1]) for point in entity.get_points()]
            top_left = min(points, key=lambda p: (p[0], p[1]))
            bottom_right = max(points, key=lambda p: (p[0], p[1]))

            # Assign labels like C1, C2, etc.
            label = f"C{len(labels) + 1}"
            labels[(top_left, bottom_right)] = label

            # Calculate center
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            centers[label] = (center_x, center_y)

    # Helper to create vertically aligned DataFrame
    def get_vertical_col(base_label, reverse_order=False):
        if base_label not in centers:
            return None
        tolerance = 5
        base_center = centers[base_label]
        aligned = [(label, center[1]) for label, center in centers.items()
                   if label != base_label and abs(center[0] - base_center[0]) <= tolerance]
        aligned_sorted = sorted(aligned, key=lambda x: x[1], reverse=reverse_order)
        labels_in_order = [base_label] + [label for label, _ in aligned_sorted]
        return pd.DataFrame(labels_in_order, columns=['connected columns'])

    # Helper to create horizontally aligned DataFrame
    def get_horizontal_col(base_label, sort_by_distance=False):
        if base_label not in centers:
            return None
        tolerance = 5
        base_center = centers[base_label]
        aligned = [(label, center[0] if not sort_by_distance else abs(center[0] - base_center[0]))
                   for label, center in centers.items()
                   if label != base_label and abs(center[1] - base_center[1]) <= tolerance]
        aligned_sorted = sorted(aligned, key=lambda x: x[1])
        labels_in_order = [base_label] + [label for label, _ in aligned_sorted]
        return pd.DataFrame(labels_in_order, columns=['connected columns'])

    # Generate DataFrames
    C1_ver_col = get_vertical_col('C1')
    C2_hor_col = get_horizontal_col('C2')
    C3_ver_col = get_vertical_col('C3', reverse_order=True)
    C4_hor_col = get_horizontal_col('C4', sort_by_distance=True)

    return C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col

# %% [markdown]
# # Step 30 - Beam on boundary 1

# %%
def calculate_edges(entity):
    """
    Calculates the left and right edges, and the height of a box (polyline entity).
    """
    points = entity.get_points('xy') if entity.dxftype() == 'LWPOLYLINE' else [vertex.dxf.location for vertex in entity.vertices]
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    height = max_y - min_y
    left_edge = min([p for p in points if p[0] == min_x], key=lambda p: p[1])
    right_edge = min([p for p in points if p[0] == max_x], key=lambda p: p[1])

    return left_edge, right_edge, height

def connect_edges_vertically_boundary_1(dxf_data, aligned_boxes_df, beam_count=0):
    msp = dxf_data.modelspace()
    box_edges = {}
    beam_info_rows = []
    
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam', dxfattribs={'color': 3})
    
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            top_edge, bottom_edge, width = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'top': top_edge, 'bottom': bottom_edge, 'width': width}
    
    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, None, dxf_data
    
    label_counter = beam_count + 1
    label_count = 0
    
    for i in range(len(aligned_boxes_df) - 1):
        label_lower = aligned_boxes_df.iloc[i, 0]
        label_upper = aligned_boxes_df.iloc[i + 1, 0]
        upper_bottom_edge = box_edges[label_upper]['bottom']
        lower_top_edge = box_edges[label_lower]['top']
        
        # Line 1 coordinates
        line_1_x_start = 0
        line_1_x_end = 0
        line_1_y_start = upper_bottom_edge[1]
        line_1_y_end = lower_top_edge[1]
        
        # Line 2 coordinates
        line_2_x_start = 9
        line_2_x_end = 9
        line_2_y_start = upper_bottom_edge[1]
        line_2_y_end = lower_top_edge[1]
        
        # Draw the lines in 'beam' layer
        msp.add_line(
            (line_1_x_start, line_1_y_start), 
            (line_1_x_end, line_1_y_end), 
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        msp.add_line(
            (line_2_x_start, line_2_y_start), 
            (line_2_x_end, line_2_y_end), 
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        # Add text label in 'beam' layer with height 7 and black color
        midpoint_y = (upper_bottom_edge[1] + lower_top_edge[1]) / 2
        label_text = f"B{label_counter}"
        text_entity = msp.add_text(
            label_text, 
            dxfattribs={
                'color': 0,  # Changed to black color
                'height': 7,
                'layer': 'beam'
            }
        )
        text_entity.dxf.insert = (-1, midpoint_y)
        text_entity.dxf.rotation = 90
        
        length = abs(upper_bottom_edge[1] - lower_top_edge[1])
        
        beam_info_rows.append({
            'beam names': label_text, 
            'length': length,
            'line_1_x_start': line_1_x_start,
            'line_1_x_end': line_1_x_end,
            'line_1_y_start': line_1_y_start,
            'line_1_y_end': line_1_y_end,
            'line_2_x_start': line_2_x_start,
            'line_2_x_end': line_2_x_end,
            'line_2_y_start': line_2_y_start,
            'line_2_y_end': line_2_y_end
        })
        
        label_counter += 1
        label_count += 1
    
    beam_info_df = pd.DataFrame(beam_info_rows)
    return label_counter, beam_info_df, dxf_data

# %% [markdown]
# # Step 31 - Beam on boundary 2

# %%
def connect_edges_horizontally_boundary_2(dxf_data, max_along_y, aligned_boxes_df, beam_count, beam_info_df):
    """
    Connects horizontally aligned boxes by drawing green lines between their edge points and labeling pairs.
    Parameters:
        dxf_data (ezdxf.document): The loaded DXF document in memory.
        aligned_boxes_df (pd.DataFrame): DataFrame containing labels of horizontally aligned boxes.
        max_along_y (float): The maximum Y-coordinate where the beams should be placed.
        beam_count (int): Starting count for the beam labels.
        beam_info_df (pd.DataFrame): DataFrame to store beam information, with columns 'beam names' and 'length'.
    Returns:
        tuple: Updated beam count after the labels are added, and updated beam_info_df DataFrame.
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
    
    msp = dxf_data.modelspace()
    box_edges = {}
    
    # Extract edges for each box
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            left_edge, right_edge, height = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'left': left_edge, 'right': right_edge, 'height': height}
            
    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, beam_info_df
        
    label_counter = beam_count + 1
    label_count = 0
    
    for i in range(len(aligned_boxes_df) - 1):
        label_left = aligned_boxes_df.iloc[i, 0]
        label_right = aligned_boxes_df.iloc[i + 1, 0]
        right_edge_left_box = box_edges[label_left]['right']
        left_edge_right_box = box_edges[label_right]['left']
        beam_length = abs(left_edge_right_box[0] - right_edge_left_box[0])
        
        # Store coordinates for both lines
        line_1_x_start = right_edge_left_box[0]
        line_1_x_end = left_edge_right_box[0]
        line_1_y_start = max_along_y
        line_1_y_end = max_along_y
        
        line_2_x_start = right_edge_left_box[0]
        line_2_x_end = left_edge_right_box[0]
        line_2_y_start = max_along_y - 9
        line_2_y_end = max_along_y - 9
        
        # Draw the connecting lines with adjusted y-coordinates in 'beam' layer
        line1 = msp.add_line(
            (line_1_x_start, line_1_y_start),
            (line_1_x_end, line_1_y_end),
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        line2 = msp.add_line(
            (line_2_x_start, line_2_y_start),
            (line_2_x_end, line_2_y_end),
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        midpoint_x = (right_edge_left_box[0] + left_edge_right_box[0]) / 2
        midpoint_y = max_along_y - 9
        label_text = f"B{label_counter}"
        # Add text with increased height, black color, and in 'beam' layer
        text_entity = msp.add_text(
            label_text, 
            dxfattribs={
                'color': 0,  # Black color
                'height': 7,  # Increased height
                'layer': 'beam'  # Added to beam layer
            }
        )
        text_entity.dxf.insert = (midpoint_x, midpoint_y)
        
        # Create new row with all coordinates
        new_row = pd.DataFrame([{
            'beam names': label_text, 
            'length': beam_length,
            'line_1_x_start': line_1_x_start,
            'line_1_x_end': line_1_x_end,
            'line_1_y_start': line_1_y_start,
            'line_1_y_end': line_1_y_end,
            'line_2_x_start': line_2_x_start,
            'line_2_x_end': line_2_x_end,
            'line_2_y_start': line_2_y_start,
            'line_2_y_end': line_2_y_end
        }])
        
        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
        label_counter += 1
        label_count += 1
        
    return label_counter, beam_info_df, dxf_data

# %% [markdown]
# # Step 32 - Beam on boundary 3

# %%
def connect_edges_vertically_boundary_3(dxf_data, aligned_boxes_df, beam_count, beam_info_df):
    """
    Connect vertically aligned boxes by drawing green lines between their edge points, labeling pairs, 
    and updating the beam_info_df with beam label and length.
    Parameters:
        dxf_data (ezdxf.document): The loaded DXF document in memory.
        aligned_boxes_df (pd.DataFrame): DataFrame containing labels of vertically aligned boxes.
        beam_count (int): Starting count for the beam labels.
        beam_info_df (pd.DataFrame): DataFrame to store information about each created beam.
    
    Returns:
        Tuple[int, pd.DataFrame, ezdxf.document]: 
            - Updated beam count,
            - The modified beam_info_df,
            - The modified DXF document in memory.
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()
    box_edges = {}
    
    # Extract edges for each box
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            left_edge, right_edge, height = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'left': left_edge, 'right': right_edge, 'height': height}
            
    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, beam_info_df, dxf_data
        
    label_counter = beam_count
    label_count = 0
    
    for i in range(len(aligned_boxes_df) - 1):
        label_left = aligned_boxes_df.iloc[i, 0]
        label_right = aligned_boxes_df.iloc[i + 1, 0]
        right_edge_left_box = box_edges[label_left]['right']
        left_edge_right_box = box_edges[label_right]['left']
        height_left_box = box_edges[label_left]['height']
        height_right_box = box_edges[label_right]['height']
        box_width = abs(right_edge_left_box[0] - left_edge_right_box[0])
        
        # Store coordinates for both lines
        line_1_x_start = right_edge_left_box[0]
        line_1_x_end = right_edge_left_box[0]
        line_1_y_start = right_edge_left_box[1]
        line_1_y_end = left_edge_right_box[1]
        
        line_2_x_start = left_edge_right_box[0]
        line_2_x_end = left_edge_right_box[0]
        line_2_y_start = right_edge_left_box[1]
        line_2_y_end = left_edge_right_box[1]
        
        # Draw the green vertical lines in 'beam' layer
        line1 = msp.add_line(
            (line_1_x_start, line_1_y_start),
            (line_1_x_end, line_1_y_end),
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        line2 = msp.add_line(
            (line_2_x_start, line_2_y_start),
            (line_2_x_end, line_2_y_end),
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        midpoint_y = (right_edge_left_box[1] + left_edge_right_box[1]) / 2
        label_text = f"B{label_counter}"
        # Add text with increased height, black color, and in 'beam' layer
        text_entity = msp.add_text(
            label_text, 
            dxfattribs={
                'color': 0,  # Black color
                'height': 7,  # Increased height
                'layer': 'beam'  # Added to beam layer
            }
        )
        text_entity.dxf.insert = (line_1_x_start + 0.1, midpoint_y)
        
        beam_length = abs(right_edge_left_box[1] - left_edge_right_box[1])
        
        # Create new row with all coordinates
        new_row = pd.DataFrame([{
            'beam names': label_text,
            'length': beam_length,
            'line_1_x_start': line_1_x_start,
            'line_1_x_end': line_1_x_end,
            'line_1_y_start': line_1_y_start,
            'line_1_y_end': line_1_y_end,
            'line_2_x_start': line_2_x_start,
            'line_2_x_end': line_2_x_end,
            'line_2_y_start': line_2_y_start,
            'line_2_y_end': line_2_y_end
        }])
        
        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
        label_counter += 1
        label_count += 1
        
    return label_counter, beam_info_df, dxf_data

# %% [markdown]
# # Step 33 - Beams on horizontally aligned boundary 1

# %%
def find_parallel_lines(msp, x1, x2, y_center, tolerance=0.5):
    """
    Find parallel horizontal lines between two x-coordinates around a y-center.
    Returns True if valid wall spacing is found (4-6 or 8-9 units).
    """
    # Store y-coordinates of horizontal lines
    horizontal_lines = []
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            
            # Check if line is horizontal
            if abs(start.y - end.y) < tolerance:
                # Check if line is between x1 and x2
                line_x_min = min(start.x, end.x)
                line_x_max = max(start.x, end.x)
                
                # Check if line overlaps with our region of interest
                if (line_x_min <= x2 and line_x_max >= x1):
                    # Check if line is near our y_center
                    if abs(start.y - y_center) < 10:  # Search within 10 units
                        horizontal_lines.append(start.y)
    
    # Sort and find pairs of lines
    horizontal_lines.sort()
    for i in range(len(horizontal_lines) - 1):
        spacing = abs(horizontal_lines[i] - horizontal_lines[i + 1])
        # Check for valid wall spacings (4-6 or 8-9 units)
        if (4 <= spacing <= 6) or (8 <= spacing <= 9):
            return True
            
    return False

def check_horizontal_alignment_boundary_1(dxf_data, max_along_x, C1_ver_col, beam_count, tolerance=15, beam_info_df=None):
    """[previous docstring remains the same]"""
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()

    # Extract boxes and calculate their center points [unchanged]
    centers = {}
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(point[0], point[1]) for point in entity.get_points()]
            top_left = min(points, key=lambda p: (p[0], p[1]))
            bottom_right = max(points, key=lambda p: (p[0], p[1]))
            label = f"C{len(centers) + 1}"
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            centers[label] = (center_x, center_y)

    alignment_data = []

    # Original alignment detection logic [unchanged]
    for i in range(1, len(C1_ver_col) - 1):
        current_label = C1_ver_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue

        current_center = centers[current_label]
        aligned_boxes = []

        for label, center in centers.items():
            if label == current_label:
                continue

            if abs(center[1] - current_center[1]) <= tolerance and current_center[0] < center[0] <= max_along_x:
                aligned_boxes.append((label, center[0]))

        aligned_labels = [label for label, _ in sorted(aligned_boxes, key=lambda x: x[1])]
        alignment_data.append({
            'Boundary column': current_label,
            'Horizontal aligned column': aligned_labels
        })

    Boundary_1_connection = pd.DataFrame(alignment_data)

    # Modified beam drawing section to capture coordinates
    for index, row in Boundary_1_connection.iterrows():
        boundary_box = row['Boundary column']
        if row['Horizontal aligned column']:
            x_start = centers[boundary_box][0]
            y_center = centers[boundary_box][1]

            for target_label in row['Horizontal aligned column']:
                target_x = centers[target_label][0]
                
                # Check for wall between columns
                if find_parallel_lines(msp, x_start, target_x, y_center):
                    # Store coordinates for both lines
                    line_1_x_start = x_start
                    line_1_x_end = target_x
                    line_1_y_start = y_center
                    line_1_y_end = y_center

                    line_2_x_start = x_start
                    line_2_x_end = target_x
                    line_2_y_start = y_center + 5
                    line_2_y_end = y_center + 5

                    # Draw beam lines using stored coordinates in 'beam' layer
                    msp.add_line(
                        (line_1_x_start, line_1_y_start),
                        (line_1_x_end, line_1_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )
                    msp.add_line(
                        (line_2_x_start, line_2_y_start),
                        (line_2_x_end, line_2_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )

                    beam_label = f"B{beam_count}"
                    label_x = (x_start + target_x) / 2
                    label_y = y_center + 7

                    # Add text with increased height, black color, and in 'beam' layer
                    msp.add_text(
                        beam_label, 
                        dxfattribs={
                            'height': 7,    # Increased height
                            'color': 0,     # Black color
                            'layer': 'beam' # Added to beam layer
                        }
                    ).set_dxf_attrib("insert", (label_x, label_y))

                    # Create new row with beam information including coordinates
                    new_row = pd.DataFrame([{
                        'beam names': beam_label,
                        'length': abs(target_x - x_start),
                        'line_1_x_start': line_1_x_start,
                        'line_1_x_end': line_1_x_end,
                        'line_1_y_start': line_1_y_start,
                        'line_1_y_end': line_1_y_end,
                        'line_2_x_start': line_2_x_start,
                        'line_2_x_end': line_2_x_end,
                        'line_2_y_start': line_2_y_start,
                        'line_2_y_end': line_2_y_end
                    }])

                    beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                    beam_count += 1

                x_start = target_x + 3
                if x_start >= max_along_x:
                    break

    return beam_count, beam_info_df, dxf_data

# %% [markdown]
# # Step 34 -Beams on vertically aligned boundary 2

# %%
def find_vertical_wall(msp, x_center, y1, y2, tolerance=0.5):
    """
    Find parallel vertical lines between two y-coordinates around an x-center.
    Returns True if valid wall spacing is found (4-6 or 8-9 units).
    """
    vertical_lines = []
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            
            # Check if line is vertical
            if abs(start.x - end.x) < tolerance:
                # Check if line is near our x_center
                if abs(start.x - x_center) < 10:  # Search within 10 units
                    # Check if line spans between our y-coordinates
                    line_y_min = min(start.y, end.y)
                    line_y_max = max(start.y, end.y)
                    
                    # Check for overlap with our region of interest
                    if (line_y_min <= y2 and line_y_max >= y1):
                        vertical_lines.append(start.x)
    
    # Sort and find pairs of lines
    vertical_lines.sort()
    for i in range(len(vertical_lines) - 1):
        spacing = abs(vertical_lines[i] - vertical_lines[i + 1])
        # Check for valid wall spacings (4-6 or 8-9 units)
        if (4 <= spacing <= 6) or (8 <= spacing <= 9):
            return True
            
    return False

def check_vertical_alignment_boundary_2(dxf_data, C2_hor_col, max_along_y, beam_count, tolerance=15, beam_info_df=None):
    """[previous docstring remains the same]"""
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()

    # Extract centers from DXF data [unchanged]
    centers = {}
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(point[0], point[1]) for point in entity.get_points()]
            top_left = min(points, key=lambda p: (p[0], p[1]))
            bottom_right = max(points, key=lambda p: (p[0], p[1]))
            label = f"C{len(centers) + 1}"
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            centers[label] = (center_x, center_y)
    
    # Iterate over C2_hor_col
    for i in range(1, len(C2_hor_col) - 1):
        current_label = C2_hor_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue

        current_center = centers[current_label]
        aligned_boxes = []

        for label, center in centers.items():
            if label != current_label and abs(center[0] - current_center[0]) <= tolerance:
                distance = sqrt((center[0] - current_center[0])**2 + 
                              (center[1] - current_center[1])**2)
                aligned_boxes.append((label, center[1], distance))

        if aligned_boxes:
            aligned_boxes_sorted = sorted(aligned_boxes, key=lambda x: x[2])
            target_label, target_y, _ = aligned_boxes_sorted[0]
            
            if current_label in centers:
                midpoint_x = centers[current_label][0]
                min_y = min(current_center[1], target_y)
                max_y = max(current_center[1], target_y)

                if find_vertical_wall(msp, midpoint_x, min_y, max_y):
                    # Store coordinates for both lines
                    line_1_x_start = midpoint_x
                    line_1_x_end = midpoint_x
                    line_1_y_start = min_y
                    line_1_y_end = max_y

                    line_2_x_start = midpoint_x + 5
                    line_2_x_end = midpoint_x + 5
                    line_2_y_start = min_y
                    line_2_y_end = max_y

                    # Draw the lines using stored coordinates in 'beam' layer
                    msp.add_line(
                        (line_1_x_start, line_1_y_start),
                        (line_1_x_end, line_1_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )
                    msp.add_line(
                        (line_2_x_start, line_2_y_start),
                        (line_2_x_end, line_2_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )
                    
                    beam_label = f"B{beam_count}"
                    label_offset_x = 15
                    midpoint_y = (min_y + max_y) / 2
                    
                    # Add text with increased height, black color, and in 'beam' layer
                    text_entity = msp.add_text(
                        beam_label, 
                        dxfattribs={
                            'color': 0,     # Black color
                            'height': 7,    # Increased height
                            'layer': 'beam' # Added to beam layer
                        }
                    )
                    text_entity.dxf.insert = (midpoint_x - label_offset_x, midpoint_y)
                    text_entity.dxf.rotation = 90
                    
                    beam_length = max_y - min_y

                    # Create new row with beam information including coordinates
                    new_row = pd.DataFrame([{
                        'beam names': beam_label,
                        'length': beam_length,
                        'line_1_x_start': line_1_x_start,
                        'line_1_x_end': line_1_x_end,
                        'line_1_y_start': line_1_y_start,
                        'line_1_y_end': line_1_y_end,
                        'line_2_x_start': line_2_x_start,
                        'line_2_x_end': line_2_x_end,
                        'line_2_y_start': line_2_y_start,
                        'line_2_y_end': line_2_y_end
                    }])

                    beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                    beam_count += 1
    
    return beam_count, beam_info_df, dxf_data

# %% [markdown]
# # Steps 35 - Beams on horizontally aligned boundary 3

# %%
def check_horizontal_alignment_boundary_3(dxf_data, max_along_x, C3_ver_col, beam_count, beam_info_df, tolerance=10):
    """[previous docstring remains the same]"""
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()

    # [Previous code unchanged until line drawing section]
    centers = {}
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(point[0], point[1]) for point in entity.get_points()]
            top_left = min(points, key=lambda p: (p[0], p[1]))
            bottom_right = max(points, key=lambda p: (p[0], p[1]))
            label = f"C{len(centers) + 1}"
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            centers[label] = (center_x, center_y)

    alignment_data = []

    # [Previous alignment detection logic unchanged]
    for i in range(1, len(C3_ver_col) - 1):
        current_label = C3_ver_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue

        current_center = centers[current_label]
        aligned_boxes = []

        for label, center in centers.items():
            if label == current_label:
                continue

            if abs(center[1] - current_center[1]) <= tolerance and 0 < center[0] <= max_along_x - 24:
                aligned_boxes.append((label, center[0]))

        aligned_labels = [label for label, _ in sorted(aligned_boxes, key=lambda x: x[1], reverse=True)]

        alignment_data.append({
            'Boundary column': current_label,
            'Horizontal aligned column': aligned_labels
        })

    Boundary_3_connection = pd.DataFrame(alignment_data)

    # Modified line and text drawing section
    for index, row in Boundary_3_connection.iterrows():
        boundary_box = row['Boundary column']
        aligned_boxes = row['Horizontal aligned column']

        if aligned_boxes:
            for next_box in aligned_boxes:
                if boundary_box in centers and next_box in centers:
                    x_min = min(centers[boundary_box][0], centers[next_box][0])
                    x_max = max(centers[boundary_box][0], centers[next_box][0])
                    y_center = centers[boundary_box][1]

                    green_line_exists = False
                    for entity in msp.query('LINE[color==3]'):
                        existing_x_start = min(entity.dxf.start.x, entity.dxf.end.x)
                        existing_x_end = max(entity.dxf.start.x, entity.dxf.end.x)
                        existing_y = entity.dxf.start.y

                        if (existing_x_start <= x_min and existing_x_end >= x_max) and \
                           (y_center - tolerance <= existing_y <= y_center + tolerance):
                            green_line_exists = True
                            break

                    if not green_line_exists:
                        # Store coordinates for both lines
                        line_1_x_start = x_min
                        line_1_x_end = x_max
                        line_1_y_start = y_center
                        line_1_y_end = y_center

                        line_2_x_start = x_min
                        line_2_x_end = x_max
                        line_2_y_start = y_center - 5
                        line_2_y_end = y_center - 5

                        # Draw the lines in 'beam' layer
                        msp.add_line(
                            (line_1_x_start, line_1_y_start),
                            (line_1_x_end, line_1_y_end),
                            dxfattribs={'color': 3, 'layer': 'beam'}
                        )
                        msp.add_line(
                            (line_2_x_start, line_2_y_start),
                            (line_2_x_end, line_2_y_end),
                            dxfattribs={'color': 3, 'layer': 'beam'}
                        )

                        beam_label = f"B{beam_count}"
                        label_x = (x_min + x_max) / 2
                        label_y = y_center + 7

                        # Add text with increased height, black color, and in 'beam' layer
                        text_entity = msp.add_text(
                            beam_label, 
                            dxfattribs={
                                'color': 0,     # Black color
                                'height': 7,    # Increased height
                                'layer': 'beam' # Added to beam layer
                            }
                        )
                        text_entity.dxf.insert = (label_x, label_y)
                        text_entity.dxf.rotation = 90

                        length = x_max - x_min
                        
                        # [Rest of code for creating new row unchanged]
                        new_row = pd.DataFrame([{
                            'beam names': beam_label,
                            'length': length,
                            'line_1_x_start': line_1_x_start,
                            'line_1_x_end': line_1_x_end,
                            'line_1_y_start': line_1_y_start,
                            'line_1_y_end': line_1_y_end,
                            'line_2_x_start': line_2_x_start,
                            'line_2_x_end': line_2_x_end,
                            'line_2_y_start': line_2_y_start,
                            'line_2_y_end': line_2_y_end
                        }])

                        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                        beam_count += 1

                    boundary_box = next_box

            # Handle last aligned box to x=0
            x_min = 0
            x_max = centers[boundary_box][0]
            y_center = centers[boundary_box][1]

            green_line_exists = False
            for entity in msp.query('LINE[color==3]'):
                existing_x_start = min(entity.dxf.start.x, entity.dxf.end.x)
                existing_x_end = max(entity.dxf.start.x, entity.dxf.end.x)
                existing_y = entity.dxf.start.y

                if (existing_x_start <= x_min and existing_x_end >= x_max) and \
                   (y_center - 15 <= existing_y <= y_center + 15):
                    green_line_exists = True
                    break

            if not green_line_exists:
                # Store coordinates for final lines
                line_1_x_start = x_min
                line_1_x_end = x_max
                line_1_y_start = y_center
                line_1_y_end = y_center

                line_2_x_start = x_min
                line_2_x_end = x_max
                line_2_y_start = y_center - 5
                line_2_y_end = y_center - 5

                # Draw final lines in 'beam' layer
                msp.add_line(
                    (line_1_x_start, line_1_y_start),
                    (line_1_x_end, line_1_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )
                msp.add_line(
                    (line_2_x_start, line_2_y_start),
                    (line_2_x_end, line_2_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )

    return beam_count, beam_info_df, dxf_data

# %% [markdown]
# # Steps 36 - Beams on vertically aligned boundary 4

# %%
def check_vertical_alignment_boundary_4(dxf_data, C4_hor_col, max_along_y, beam_count, beam_info_df, tolerance=15):
    """[previous docstring remains the same]"""
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()
    
    # [Previous code unchanged until line drawing section]
    centers = {}
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            points = [(point[0], point[1]) for point in entity.get_points()]
            top_left = min(points, key=lambda p: (p[0], p[1]))
            bottom_right = max(points, key=lambda p: (p[0], p[1]))
            label = f"C{len(centers) + 1}"
            center_x = (top_left[0] + bottom_right[0]) / 2
            center_y = (top_left[1] + bottom_right[1]) / 2
            centers[label] = (center_x, center_y)
    
    # [Previous alignment detection logic unchanged]
    for i in range(1, len(C4_hor_col) - 1):
        current_label = C4_hor_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue

        current_center = centers[current_label]
        aligned_boxes = []

        for label, center in centers.items():
            if label != current_label and abs(center[0] - current_center[0]) <= tolerance:
                distance = sqrt((center[0] - current_center[0])**2 + (center[1] - current_center[1])**2)
                aligned_boxes.append((label, center[1], distance))

        if aligned_boxes:
            aligned_boxes_sorted = sorted(aligned_boxes, key=lambda x: x[2])
            target_label, target_y, _ = aligned_boxes_sorted[0]
            
            if current_label in centers:
                midpoint_x = centers[current_label][0]
                first_line_x = midpoint_x

                min_y = min(current_center[1], target_y)
                max_y = max(current_center[1], target_y)

                # [Previous line checking logic unchanged]
                parallel_lines = []
                for entity in msp:
                    if entity.dxftype() == 'LINE':
                        if abs(entity.dxf.start.x - entity.dxf.end.x) <= tolerance:
                            if abs(entity.dxf.start.x - first_line_x) <= tolerance:
                                parallel_lines.append(entity)

                if len(parallel_lines) == 1:
                    second_line_x = first_line_x + 5
                    parallel_line_found = False
                    for entity in msp:
                        if entity.dxftype() == 'LINE':
                            if abs(entity.dxf.start.x - second_line_x) <= tolerance and abs(entity.dxf.end.x - second_line_x) <= tolerance:
                                if (abs(entity.dxf.start.y - 0) <= 1 or abs(entity.dxf.start.y - 9) <= 1 or
                                    abs(entity.dxf.end.y - 0) <= 1 or abs(entity.dxf.end.y - 9) <= 1):
                                    parallel_line_found = True
                                    break
                    
                    if parallel_line_found:
                        # Store coordinates for both lines
                        line_1_x_start = first_line_x
                        line_1_x_end = first_line_x
                        line_1_y_start = min_y
                        line_1_y_end = max_y

                        line_2_x_start = first_line_x + 5
                        line_2_x_end = first_line_x + 5
                        line_2_y_start = min_y
                        line_2_y_end = max_y

                        # Draw the lines in 'beam' layer
                        msp.add_line(
                            (line_1_x_start, line_1_y_start),
                            (line_1_x_end, line_1_y_end),
                            dxfattribs={'color': 3, 'layer': 'beam'}
                        )
                        msp.add_line(
                            (line_2_x_start, line_2_y_start),
                            (line_2_x_end, line_2_y_end),
                            dxfattribs={'color': 3, 'layer': 'beam'}
                        )
                        
                        beam_label = f"B{beam_count}"
                        label_offset_x = 15
                        midpoint_y = (min_y + max_y) / 2
                        
                        # Add text with increased height, black color, and in 'beam' layer
                        text_entity = msp.add_text(
                            beam_label, 
                            dxfattribs={
                                'color': 0,     # Black color
                                'height': 7,    # Increased height
                                'layer': 'beam' # Added to beam layer
                            }
                        )
                        text_entity.dxf.insert = (midpoint_y - label_offset_x, midpoint_y)
                        text_entity.dxf.rotation = 90
                        
                        beam_length = max_y - min_y

                        # [Rest of code for creating new row unchanged]
                        new_row = pd.DataFrame([{
                            'beam_names': beam_label,
                            'length': beam_length,
                            'line_1_x_start': line_1_x_start,
                            'line_1_x_end': line_1_x_end,
                            'line_1_y_start': line_1_y_start,
                            'line_1_y_end': line_1_y_end,
                            'line_2_x_start': line_2_x_start,
                            'line_2_x_end': line_2_x_end,
                            'line_2_y_start': line_2_y_start,
                            'line_2_y_end': line_2_y_end
                        }])
                        
                        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                        beam_count += 1
    
    return beam_count, beam_info_df, dxf_data

# %% [markdown]
# # Steps 37 - Final code for alignment data 

# %%
def read_and_filter_columns(dxf_data, C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col):
    """
    Reads all closed polylines (columns) from the DXF file, creates a complete list of columns, and removes the ones
    present in the provided DataFrames.

    Parameters:
        dxf_data: The input DXF data.
        C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col: DataFrames containing columns to exclude.

    Returns:
        A list of filtered columns.
    """
    # Access the modelspace of the provided DXF data
    msp = dxf_data.modelspace()

    # List to store all column names from the DXF file
    all_columns = []
    column_positions = {}  # To store positions of columns

    # Iterate over all entities in the modelspace
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
            # Extract the polyline points to calculate the positions
            points = entity.get_points('xy')  # Get points as (x, y) tuples

            # Ensure it's a rectangular box (typically 5 points: 4 corners + closing point)
            if len(points) == 5 and points[0] == points[-1]:
                # Calculate the corners of the box
                min_x = min([p[0] for p in points])
                max_x = max([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_y = max([p[1] for p in points])

                # Create a unique label for the box (column)
                column_label = f"C{len(all_columns) + 1}"  # Example: C1, C2, C3, ...

                # Add the column label and its position to the list and dictionary
                all_columns.append(column_label)
                column_positions[column_label] = {
                    "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y,
                    "center_x": (min_x + max_x) / 2, "center_y": (min_y + max_y) / 2
                }

    # Combine all columns from the provided DataFrames into a single list
    excluded_columns = (
        C1_ver_col['connected columns'].tolist() +
        C2_hor_col['connected columns'].tolist() +
        C3_ver_col['connected columns'].tolist() +
        C4_hor_col['connected columns'].tolist()
    )

    # Filter out the excluded columns from the entire list of columns
    filtered_columns = [col for col in all_columns if col not in excluded_columns]

    return filtered_columns, column_positions

def find_aligned_columns(filtered_columns, column_positions, tolerance=7):
    """
    Find vertical and horizontal aligned columns for each column in the filtered list, with a given tolerance.
    For left-aligned and right-aligned columns, sort them based on their distance from the base column.
    Parameters:
        filtered_columns: List of column labels to check.
        column_positions: Dictionary with column labels as keys and their position data as values.
        tolerance: Allowed deviation for alignment in coordinates.
    Returns:
        A dictionary with each column and its aligned columns in the left, right, up, and down directions.
        Left-aligned and right-aligned columns are sorted by distance from base column.
    """
    alignment_data = {}
    for column in filtered_columns:
        current = column_positions[column]
        aligned = {"left": [], "right": [], "up": [], "down": []}
        left_columns_with_distances = []  # To store tuples of (column, distance)
        right_columns_with_distances = []  # To store tuples of (column, distance)
        up_columns_with_distances = []
        down_columns_with_distances = []
        
        for other_column in filtered_columns:
            if column == other_column:
                continue
            other = column_positions[other_column]
            # Check horizontal alignment (left and right)
            if abs(current["center_y"] - other["center_y"]) <= tolerance:
                if other["center_x"] < current["center_x"]:
                    # Store the column and its distance for later sorting
                    distance = current["center_x"] - other["center_x"]
                    left_columns_with_distances.append((other_column, distance))
                elif other["center_x"] > current["center_x"]:
                    # Store the column and its distance for right alignment sorting
                    distance = other["center_x"] - current["center_x"]
                    right_columns_with_distances.append((other_column, distance))
            # Check vertical alignment (up and down)
            if abs(current["center_x"] - other["center_x"]) <= tolerance:
                if other["center_y"] > current["center_y"]:
                    distance = other["center_y"] - current["center_y"]
                    up_columns_with_distances.append((other_column, distance))
                elif other["center_y"] < current["center_y"]:
                    distance = current["center_y"] - other["center_y"]
                    down_columns_with_distances.append((other_column, distance))
                    
        # Sort left columns based on distance (ascending) and extract just the column labels
        if left_columns_with_distances:
            left_columns_with_distances.sort(key=lambda x: x[1])
            aligned["left"] = [col for col, _ in left_columns_with_distances]
            
        # Sort right columns based on distance (ascending) and extract just the column labels
        if right_columns_with_distances:
            right_columns_with_distances.sort(key=lambda x: x[1])
            aligned["right"] = [col for col, _ in right_columns_with_distances]

        if up_columns_with_distances:
            up_columns_with_distances.sort(key=lambda x:x[1])
            aligned["up"] = [col for col, _ in up_columns_with_distances]

        if down_columns_with_distances:
            down_columns_with_distances.sort(key=lambda x:x[1])
            aligned["down"] = [col for col, _ in down_columns_with_distances]
            
        alignment_data[column] = aligned
    return alignment_data

# %% [markdown]
# # Step 38 - Individual dataframes 

# %%
def process_alignment_data_by_direction(alignment_data, column_info_df, direction):
    """
    Generic function to process alignment data for any direction (left, right, up, down).
    
    Parameters:
        alignment_data (dict): Dictionary containing alignment information for columns.
        column_info_df (pd.DataFrame): DataFrame with column details.
        direction (str): Direction to process ('left', 'right', 'up', 'down')
    
    Returns:
        pd.DataFrame: DataFrame containing pairs of columns and their x, y centers.
    """
    result = []
    for base_col, directions in alignment_data.items():
        aligned_cols = directions.get(direction, [])
        if not aligned_cols:  # Skip if direction is empty
            continue
            
        # Get base column's x and y centers
        base_row = column_info_df[column_info_df['Column Nos'] == base_col]
        if base_row.empty:
            continue
        base_x, base_y = base_row.iloc[0]['X Center'], base_row.iloc[0]['Y Center']
        
        # Iterate through aligned columns
        for i, aligned_col in enumerate(aligned_cols):
            aligned_row = column_info_df[column_info_df['Column Nos'] == aligned_col]
            if aligned_row.empty:
                continue
            aligned_x, aligned_y = aligned_row.iloc[0]['X Center'], aligned_row.iloc[0]['Y Center']
            
            if i == 0:
                # Base column and first aligned column
                result.append((base_col, base_x, base_y, aligned_col, aligned_x, aligned_y))
            else:
                # Consecutive aligned columns
                prev_col = aligned_cols[i - 1]
                prev_row = column_info_df[column_info_df['Column Nos'] == prev_col]
                if prev_row.empty:
                    continue
                prev_x, prev_y = prev_row.iloc[0]['X Center'], prev_row.iloc[0]['Y Center']
                result.append((prev_col, prev_x, prev_y, aligned_col, aligned_x, aligned_y))
    
    # Create a DataFrame from the results
    columns = ['Column 1', 'Column 1 X', 'Column 1 Y', 'Column 2', 'Column 2 X', 'Column 2 Y']
    result_df = pd.DataFrame(result, columns=columns)
    
    # Remove duplicate rows
    result_df = result_df.drop_duplicates()
    return result_df

def get_all_alignments(alignment_data, column_info_df):
    """
    Process alignment data for all four directions and return separate DataFrames.
    
    Parameters:
        alignment_data (dict): Dictionary containing alignment information for columns.
        column_info_df (pd.DataFrame): DataFrame with column details.
    
    Returns:
        tuple: (left_df, right_df, up_df, down_df)
    """
    # Process each direction
    left_df = process_alignment_data_by_direction(alignment_data, column_info_df, 'left')
    right_df = process_alignment_data_by_direction(alignment_data, column_info_df, 'right')
    up_df = process_alignment_data_by_direction(alignment_data, column_info_df, 'up')
    down_df = process_alignment_data_by_direction(alignment_data, column_info_df, 'down')
    
    return left_df, right_df, up_df, down_df

# %% [markdown]
# # Step 39 - Left aligned columns 

# %%
def beam_and_wall_detection_left(dxf_data, left_df, beam_count, beam_info_df):
    """
    Check for green and gray horizontal lines between pairs of columns.
    When exactly 2 parallel gray lines with spacing 4-9 units are found,
    draw green horizontal lines at those positions.
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
    
    # Create a copy of the input DataFrame
    df = left_df.copy()
    
    # [Previous column creation code unchanged]
    df['green_horizontal_line'] = False
    df['first_gray_horizontal_line'] = False
    df['gray_line_count'] = 0 
    df['has_parallel_gray_lines'] = False
    
    # Define the light gray color using RGB values
    R, G, B = 128, 128, 128
    light_gray_color = (R << 16) | (G << 8) | B
    
    msp = dxf_data.modelspace()
    
    # [Previous iteration and calculation logic unchanged]
    for idx, row in df.iterrows():
        Y1 = min(row['Column 1 Y'], row['Column 2 Y']) - 7
        Y2 = max(row['Column 1 Y'], row['Column 2 Y']) + 7
        X1 = min(row['Column 1 X'], row['Column 2 X']) + 5
        X2 = max(row['Column 1 X'], row['Column 2 X']) - 5
        
        # [Previous green line check logic unchanged]
        found_green_line = False
        for entity in msp:
            if entity.dxftype() == 'LINE' and entity.dxf.color == 3:
                start_point = entity.dxf.start
                end_point = entity.dxf.end
                
                if abs(start_point[1] - end_point[1]) <= 0.1:
                    line_y = start_point[1]
                    if Y1 <= line_y <= Y2:
                        line_x1 = min(start_point[0], end_point[0])
                        line_x2 = max(start_point[0], end_point[0])
                        
                        if line_x1 <= X1 + 2 and line_x2 >= X2 - 2:
                            found_green_line = True
                            break
        
        df.at[idx, 'green_horizontal_line'] = found_green_line
        
        # If no green line found, check for gray lines
        if not found_green_line:
            # [Previous gray line detection logic unchanged]
            gray_lines_y = []
            
            for entity in msp:
                if entity.dxftype() == 'LINE' and entity.dxf.true_color == light_gray_color:
                    start_point = entity.dxf.start
                    end_point = entity.dxf.end
                    
                    if abs(start_point[1] - end_point[1]) <= 0.1:
                        line_y = start_point[1]
                        if Y1 <= line_y <= Y2:
                            line_x1 = min(start_point[0], end_point[0])
                            line_x2 = max(start_point[0], end_point[0])
                            
                            if line_x1 <= X1 + 10 and line_x2 >= X2 - 10:
                                gray_lines_y.append(line_y)
            
            # [Previous parallel line check logic unchanged]
            gray_lines_y.sort()
            valid_parallel_pair = None
            
            if len(gray_lines_y) >= 2:
                for i in range(len(gray_lines_y) - 1):
                    for j in range(i + 1, len(gray_lines_y)):
                        spacing = abs(gray_lines_y[j] - gray_lines_y[i])
                        if 4 <= spacing <= 9:
                            if len(gray_lines_y) == 2:
                                valid_parallel_pair = (gray_lines_y[i], gray_lines_y[j])
                                break
                    if valid_parallel_pair:
                        break
            
            if valid_parallel_pair:
                # Draw green lines in 'beam' layer
                line_1_x_start = X1
                line_1_x_end = X2
                line_1_y_start = valid_parallel_pair[0]
                line_1_y_end = valid_parallel_pair[0]

                line_2_x_start = X1
                line_2_x_end = X2
                line_2_y_start = valid_parallel_pair[1]
                line_2_y_end = valid_parallel_pair[1]

                msp.add_line(
                    (line_1_x_start, line_1_y_start),
                    (line_1_x_end, line_1_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )
                
                msp.add_line(
                    (line_2_x_start, line_2_y_start),
                    (line_2_x_end, line_2_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )

                # Add beam label with new specifications
                beam_label = f"B{beam_count}"
                label_x = (X1 + X2) / 2
                label_y = max(valid_parallel_pair) + 7

                msp.add_text(
                    beam_label, 
                    dxfattribs={
                        'color': 0,     # Black color
                        'height': 7,    # Increased height
                        'layer': 'beam' # Added to beam layer
                    }
                ).set_dxf_attrib("insert", (label_x, label_y))

                # [Previous DataFrame handling unchanged]
                new_row = pd.DataFrame([{
                    'beam names': beam_label,
                    'length': X2 - X1,
                    'line_1_x_start': line_1_x_start,
                    'line_1_x_end': line_1_x_end,
                    'line_1_y_start': line_1_y_start,
                    'line_1_y_end': line_1_y_end,
                    'line_2_x_start': line_2_x_start,
                    'line_2_x_end': line_2_x_end,
                    'line_2_y_start': line_2_y_start,
                    'line_2_y_end': line_2_y_end
                }])

                beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                beam_count += 1
                
                df.at[idx, 'has_parallel_gray_lines'] = True
            
            df.at[idx, 'gray_horizontal_line'] = len(gray_lines_y) > 0
            df.at[idx, 'gray_line_count'] = len(gray_lines_y)
    
    return df, beam_info_df, beam_count, dxf_data

# %% [markdown]
# # Step 40 - Right aligned columns 

# %%
def beam_and_wall_detection_right(dxf_data, right_df, beam_count, beam_info_df):
    """
    Check for green and gray horizontal lines between pairs of columns.
    When exactly 2 parallel gray lines with spacing 4-9 units are found,
    draw green horizontal lines at those positions.
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
    
    # Create a copy of the input DataFrame
    df = right_df.copy()
    
    # [Previous DataFrame initialization code remains unchanged]
    df['green_horizontal_line'] = False
    df['first_gray_horizontal_line'] = False
    df['gray_line_count'] = 0
    df['has_parallel_gray_lines'] = False
    
    R, G, B = 128, 128, 128
    light_gray_color = (R << 16) | (G << 8) | B
    
    msp = dxf_data.modelspace()
    
    # [Previous iteration and line detection logic remains unchanged]
    for idx, row in df.iterrows():
        Y1 = min(row['Column 1 Y'], row['Column 2 Y']) - 7
        Y2 = max(row['Column 1 Y'], row['Column 2 Y']) + 7
        X1 = min(row['Column 1 X'], row['Column 2 X']) + 5
        X2 = max(row['Column 1 X'], row['Column 2 X']) - 5
        
        # [Previous green line checking logic remains unchanged]
        found_green_line = False
        for entity in msp:
            if entity.dxftype() == 'LINE' and entity.dxf.color == 3:
                start_point = entity.dxf.start
                end_point = entity.dxf.end
                
                if abs(start_point[1] - end_point[1]) <= 0.1:
                    line_y = start_point[1]
                    if Y1 <= line_y <= Y2:
                        line_x1 = min(start_point[0], end_point[0])
                        line_x2 = max(start_point[0], end_point[0])
                        
                        if line_x1 <= X1 + 2 and line_x2 >= X2 - 2:
                            found_green_line = True
                            break
        
        df.at[idx, 'green_horizontal_line'] = found_green_line
        
        if not found_green_line:
            # [Previous gray line detection logic remains unchanged]
            gray_lines_y = []
            
            for entity in msp:
                if entity.dxftype() == 'LINE' and entity.dxf.true_color == light_gray_color:
                    start_point = entity.dxf.start
                    end_point = entity.dxf.end
                    
                    if abs(start_point[1] - end_point[1]) <= 0.1:
                        line_y = start_point[1]
                        if Y1 <= line_y <= Y2:
                            line_x1 = min(start_point[0], end_point[0])
                            line_x2 = max(start_point[0], end_point[0])
                            
                            if line_x1 <= X1 + 10 and line_x2 >= X2 - 10:
                                gray_lines_y.append(line_y)
            
            # [Previous parallel line check logic remains unchanged]
            gray_lines_y.sort()
            valid_parallel_pair = None
            
            if len(gray_lines_y) >= 2:
                for i in range(len(gray_lines_y) - 1):
                    for j in range(i + 1, len(gray_lines_y)):
                        spacing = abs(gray_lines_y[j] - gray_lines_y[i])
                        if 4 <= spacing <= 9:
                            if len(gray_lines_y) == 2:
                                valid_parallel_pair = (gray_lines_y[i], gray_lines_y[j])
                                break
                    if valid_parallel_pair:
                        break
            
            if valid_parallel_pair:
                # Store coordinates for both lines
                line_1_x_start = X1
                line_1_x_end = X2
                line_1_y_start = valid_parallel_pair[0]
                line_1_y_end = valid_parallel_pair[0]

                line_2_x_start = X1
                line_2_x_end = X2
                line_2_y_start = valid_parallel_pair[1]
                line_2_y_end = valid_parallel_pair[1]

                # Draw lines in 'beam' layer
                msp.add_line(
                    (line_1_x_start, line_1_y_start),
                    (line_1_x_end, line_1_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )
                
                msp.add_line(
                    (line_2_x_start, line_2_y_start),
                    (line_2_x_end, line_2_y_end),
                    dxfattribs={'color': 3, 'layer': 'beam'}
                )

                # Add beam label with new specifications
                beam_label = f"B{beam_count}"
                label_x = (X1 + X2) / 2
                label_y = max(valid_parallel_pair) + 7

                msp.add_text(
                    beam_label, 
                    dxfattribs={
                        'color': 0,     # Black color
                        'height': 7,    # Increased height
                        'layer': 'beam' # Added to beam layer
                    }
                ).set_dxf_attrib("insert", (label_x, label_y))

                # [Previous DataFrame handling remains unchanged]
                new_row = pd.DataFrame([{
                    'beam names': beam_label,
                    'length': X2 - X1,
                    'line_1_x_start': line_1_x_start,
                    'line_1_x_end': line_1_x_end,
                    'line_1_y_start': line_1_y_start,
                    'line_1_y_end': line_1_y_end,
                    'line_2_x_start': line_2_x_start,
                    'line_2_x_end': line_2_x_end,
                    'line_2_y_start': line_2_y_start,
                    'line_2_y_end': line_2_y_end
                }])

                beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                beam_count += 1
                
                df.at[idx, 'has_parallel_gray_lines'] = True
            
            df.at[idx, 'gray_horizontal_line'] = len(gray_lines_y) > 0
            df.at[idx, 'gray_line_count'] = len(gray_lines_y)
    
    return df, beam_info_df, beam_count, dxf_data

# %% [markdown]
# # Step 41 - Up aligned columns 

# %%
def beam_and_wall_detection_up(dxf_data, up_df, beam_count, beam_info_df):
    """
    Check for green and gray vertical lines between pairs of columns.
    Draws green vertical lines where exactly 2 parallel gray lines with spacing 3.5-9.5 are found.
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
    
    # [Previous initialization code remains unchanged]
    df = up_df.copy()
    df['green_vertical_line'] = False
    df['gray_vertical_line'] = False
    df['gray_line_count'] = 0
    df['gray_line_spacings'] = ''
    df['has_parallel_gray_lines'] = False
    
    R, G, B = 128, 128, 128
    light_gray_color = (R << 16) | (G << 8) | B
    
    msp = dxf_data.modelspace()
    
    # [Previous iteration and detection logic remains unchanged]
    for idx, row in df.iterrows():
        X1 = min(row['Column 1 X'], row['Column 2 X']) - 8
        X2 = max(row['Column 1 X'], row['Column 2 X']) + 8
        Y1 = min(row['Column 1 Y'], row['Column 2 Y']) - 7
        Y2 = max(row['Column 1 Y'], row['Column 2 Y']) + 7
        
        found_green_line = False
        
        # [Previous green line check logic remains unchanged]
        for entity in msp:
            if entity.dxftype() == 'LINE' and entity.dxf.color == 3:
                start_point = entity.dxf.start
                end_point = entity.dxf.end
                
                if abs(start_point[0] - end_point[0]) <= 0.1:
                    line_x = start_point[0]
                    
                    if X1 <= line_x <= X2:
                        line_y1 = min(start_point[1], end_point[1])
                        line_y2 = max(start_point[1], end_point[1])
                        
                        if line_y1 <= Y1 + 2 and line_y2 >= Y2 - 2:
                            found_green_line = True
                            break
        
        df.at[idx, 'green_vertical_line'] = found_green_line
        
        if not found_green_line:
            # [Previous gray line detection logic remains unchanged until line drawing section]
            valid_gray_lines = []
            min_y = min(row['Column 1 Y'], row['Column 2 Y'])
            max_y = max(row['Column 1 Y'], row['Column 2 Y'])
            
            # [Previous gray line collection logic remains unchanged]
            for entity in msp:
                if entity.dxftype() == 'LINE' and entity.dxf.true_color == light_gray_color:
                    start_point = entity.dxf.start
                    end_point = entity.dxf.end
                    
                    if abs(start_point[0] - end_point[0]) <= 0.1:
                        line_x = start_point[0]
                        
                        if X1 <= line_x <= X2:
                            line_y1 = min(start_point[1], end_point[1])
                            line_y2 = max(start_point[1], end_point[1])
                            
                            if (line_y1 <= min_y + 10 and line_y2 >= max_y - 10):
                                valid_gray_lines.append({
                                    'x': line_x,
                                    'y1': line_y1,
                                    'y2': line_y2
                                })
            
            valid_gray_lines.sort(key=lambda x: x['x'])
            
            spacings = []
            if len(valid_gray_lines) >= 2:
                for i in range(len(valid_gray_lines) - 1):
                    spacing = round(valid_gray_lines[i + 1]['x'] - valid_gray_lines[i]['x'], 2)
                    spacings.append(str(spacing))
            
            has_parallel_lines = False
            if len(valid_gray_lines) == 2:
                spacing = abs(valid_gray_lines[1]['x'] - valid_gray_lines[0]['x'])
                if 3.5 <= spacing <= 9.5:
                    has_parallel_lines = True

                    # Draw green vertical lines in 'beam' layer
                    line_1_x_start = valid_gray_lines[0]['x']
                    line_1_x_end = valid_gray_lines[0]['x']
                    line_1_y_start = min_y
                    line_1_y_end = max_y

                    line_2_x_start = valid_gray_lines[1]['x']
                    line_2_x_end = valid_gray_lines[1]['x']
                    line_2_y_start = min_y
                    line_2_y_end = max_y

                    msp.add_line(
                        (line_1_x_start, line_1_y_start),
                        (line_1_x_end, line_1_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )
                    msp.add_line(
                        (line_2_x_start, line_2_y_start),
                        (line_2_x_end, line_2_y_end),
                        dxfattribs={'color': 3, 'layer': 'beam'}
                    )

                    # Add beam label with new specifications
                    beam_label = f"B{beam_count}"
                    label_x = (line_1_x_start + line_2_x_start) / 2
                    label_y = (min_y + max_y) / 2

                    msp.add_text(
                        beam_label,
                        dxfattribs={
                            'color': 0,     # Black color
                            'height': 7,    # Increased height
                            'layer': 'beam' # Added to beam layer
                        }
                    ).set_dxf_attrib("insert", (label_x - 1, label_y))

                    # [Previous DataFrame handling remains unchanged]
                    new_row = pd.DataFrame([{
                        'beam names': beam_label,
                        'length': max_y - min_y,
                        'line_1_x_start': line_1_x_start,
                        'line_1_x_end': line_1_x_end,
                        'line_1_y_start': line_1_y_start,
                        'line_1_y_end': line_1_y_end,
                        'line_2_x_start': line_2_x_start,
                        'line_2_x_end': line_2_x_end,
                        'line_2_y_start': line_2_y_start,
                        'line_2_y_end': line_2_y_end
                    }])

                    beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                    beam_count += 1
            
            # [Previous DataFrame updates remain unchanged]
            df.at[idx, 'gray_vertical_line'] = len(valid_gray_lines) > 0
            df.at[idx, 'gray_line_count'] = len(valid_gray_lines)
            df.at[idx, 'gray_line_spacings'] = ', '.join(spacings) if spacings else 'N/A'
            df.at[idx, 'has_parallel_gray_lines'] = has_parallel_lines
    
    return df, beam_info_df, beam_count, dxf_data

# %% [markdown]
# # Step 42 - Beams on additional floors using beam dataframe 

# %%
def add_lines_and_labels_from_dataframe(dxf_data, beam_info_df, text_height=1, text_color=7, text_offset=5):
    """
    Add lines and labels to the DXF drawing based on beam_info_df.
    Parameters:
    dxf_data: ezdxf drawing object
    beam_info_df: DataFrame with columns ['beam names', 'length', 'line_1_x_start', 'line_1_x_end', 
                                          'line_1_y_start', 'line_1_y_end', 'line_2_x_start', 
                                          'line_2_x_end', 'line_2_y_start', 'line_2_y_end']
    text_height: float, height of the text
    text_color: int, DXF color index for the label
    text_offset: float, distance to offset the text for better visibility
    """
    # Create 'beam' layer if it doesn't exist
    if 'beam' not in dxf_data.layers:
        dxf_data.layers.new(name='beam')
        
    msp = dxf_data.modelspace()
    
    for _, row in beam_info_df.iterrows():
        # Extract line coordinates
        line_1_start = (row['line_1_x_start'], row['line_1_y_start'])
        line_1_end = (row['line_1_x_end'], row['line_1_y_end'])
        line_2_start = (row['line_2_x_start'], row['line_2_y_start'])
        line_2_end = (row['line_2_x_end'], row['line_2_y_end'])
        
        # Draw first line in 'beam' layer
        msp.add_line(
            start=line_1_start, 
            end=line_1_end, 
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        # Draw second line in 'beam' layer
        msp.add_line(
            start=line_2_start, 
            end=line_2_end, 
            dxfattribs={'color': 3, 'layer': 'beam'}
        )
        
        # Determine if the beam is vertical or horizontal
        if row['line_1_x_start'] == row['line_1_x_end']:  # Vertical beam
            text_x = row['line_1_x_start'] + text_offset  # Shift text to the right
            text_y = (row['line_1_y_start'] + row['line_1_y_end']) / 2  # Center vertically
        else:  # Horizontal beam
            text_x = (row['line_1_x_start'] + row['line_1_x_end']) / 2  # Center horizontally
            text_y = row['line_1_y_start'] + text_offset  # Shift text above
            
        text_position = (text_x, text_y)
        
        # Add beam label with new specifications using add_text
        text = msp.add_text(
            row['beam names'],
            dxfattribs={
                'height': 7,     # Increased height
                'color': 0,      # Black color
                'layer': 'beam'  # Added to beam layer
            }
        )
        text.dxf.insert = text_position
    
    return dxf_data

# %% [markdown]
# # Step 43 - Column schedule dxf

# %%
def dataframe_to_dxf_table(df, output_dxf_path, start_x=0, start_y=100, col_width=80, row_height=20):
    """
    Convert a DataFrame to a table in a new DXF file with title 'COLUMN LEGEND'.
    """
    
    # Create a new DXF file
    doc = ezdxf.new()
    
    # Set the document units to inches
    doc.header['$INSUNITS'] = 1  # 1 = Inches
    doc.header['$LUNITS'] = 2    # 2 = Decimal
    doc.header['$MEASUREMENT'] = 1  # 1 = English (inches)
    
    # Calculate total height components (in inches)
    title_spacing = 20  # Space between title and table
    title_height = 10   # Height of title text
    table_height = (len(df) + 1) * row_height  # Height of table (+1 for header row)
    total_height = title_spacing + title_height + table_height
    
    msp = doc.modelspace()

    # Create column_legend layer
    if 'column_legend' not in doc.layers:
        doc.layers.new(name='column_legend')
    
    # Get column headers
    headers = df.columns.tolist()
    
    # Calculate table width
    table_width = col_width * len(headers)
    
    # Add title 'COLUMN LEGEND'
    title_height = 10  # Increased from 5 to 10
    title_x = start_x + table_width/2  # Center of table
    title_y = start_y + 20  # Increased spacing above table
    
    title = msp.add_text(
        'COLUMN LEGEND',
        dxfattribs={
            'height': title_height,
            'color': 0,  # Black color
            'layer': 'column_legend'
        }
    )
    title.dxf.insert = (title_x, title_y)
    title.dxf.halign = 1  # Center horizontal alignment
    title.dxf.valign = 1  # Center vertical alignment
    
    # Draw horizontal lines for table
    for i in range(len(df) + 2):  # +2 for header and bottom line
        y = start_y - i * row_height
        msp.add_line(
            (start_x, y),
            (start_x + col_width * len(headers), y),
            dxfattribs={'color': 0, 'layer': 'column_legend'}
        )
    
    # Draw vertical lines for table
    for i in range(len(headers) + 1):  # +1 for rightmost line
        x = start_x + i * col_width
        msp.add_line(
            (x, start_y),
            (x, start_y - (len(df) + 1) * row_height),
            dxfattribs={'color': 0, 'layer': 'column_legend'}
        )
    
    # Add headers
    for col_idx, header in enumerate(headers):
        x = start_x + col_idx * col_width + col_width/2  # Center of column
        text = msp.add_text(
            str(header),
            dxfattribs={
                'height': 7,  # Increased from 3 to 7
                'color': 0,
                'layer': 'column_legend'
            }
        )
        text.dxf.insert = (x, start_y - row_height/2)
        text.dxf.halign = 1  # Center horizontal alignment
        text.dxf.valign = 1  # Center vertical alignment
    
    # Add data
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row):
            x = start_x + col_idx * col_width + col_width/2  # Center of column
            y = start_y - (row_idx + 2) * row_height + row_height/2  # Center of row
            
            # Format value to one decimal place if it's a number
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = str(value)
                
            text = msp.add_text(
                formatted_value,
                dxfattribs={
                    'height': 5,  # Increased from 2.5 to 5
                    'color': 0,
                    'layer': 'column_legend'
                }
            )
            text.dxf.insert = (x, y)
            text.dxf.halign = 1  # Center horizontal alignment
            text.dxf.valign = 1  # Center vertical alignment
    
    # Save the DXF file
    doc.saveas(output_dxf_path)

    return total_height

# %% [markdown]
# # Merge code

# %%
def merge_dxf_files(input_dxf_paths, output_path):
    """
    Merge multiple DXF files into a single DXF file with detailed logging.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load all input drawings
    drawings = []
    for file_path in input_dxf_paths:
        try:
            drawing = ezdxf.readfile(file_path)
            # Log entity count before merging
            msp = drawing.modelspace()
            entity_count = len(list(msp))
            drawings.append((file_path, drawing))
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            continue

    # Create new DXF for merged output
    merged_dxf = ezdxf.new('R2010')
    merged_msp = merged_dxf.modelspace()
    
    total_entities_copied = 0

    # Combine all drawings
    for file_path, drawing in drawings:
        msp = drawing.modelspace()
        entities_copied = 0
        
        for entity in msp:
            try:
                # Copy entity
                merged_msp.add_entity(entity.copy())
                entities_copied += 1
                
            except Exception as e:
                logging.error(f"Error copying entity from {file_path}: {e}")
                continue
        
        total_entities_copied += entities_copied

    # Verify final entity count
    final_count = len(list(merged_dxf.modelspace()))

    # Save merged DXF
    try:
        merged_dxf.saveas(output_path)
        return merged_dxf
    except Exception as e:
        logging.error(f"Error saving merged file: {e}")
        return None
    
    # Verify input files exist
    for file_path in input_dxf_paths:
        try:
            doc = ezdxf.readfile(file_path)
            msp = doc.modelspace()
            entity_count = len(list(msp))
        except Exception as e:
            logging.error(f"Cannot read {file_path}: {e}")


# %%
def pipeline_main_final(input_file, output_file):
    """
    Optimized main pipeline function for in-memory DXF manipulation.

    Parameters:
    - input_file: Path to the input DXF file.
    - output_filename: Path to save the final modified DXF file.
    """

    base_offset = 500  # Offset in X-direction for each floor and column
    
    # Step 1: Adjust coordinates 
    adjust_dxf_coordinates_to00(input_file, 'temp_1.dxf')

    set_transparency_for_all_entities('temp_1.dxf', 'temp_2.dxf', transparency_percent = 80)
    
    # Step 2: Convert dxf to dataframe
    df= Dxf_to_DF_1('temp_2.dxf')
    
    # Remove blocks from dataframe 
    df = df[df['Type'] != 'INSERT']

    # Step 3: Distribute floors
    df = floor_main(df)
    
    # Get the unique values in the 'floor' column
    unique_floors = df['floor'].unique()

    # Create a dictionary to store DataFrames dynamically
    floor_dataframes = {}

    # Iterate through unique floor values and filter the DataFrame
    for floor in unique_floors:
        floor_dataframes[f'df_{floor}'] = df[df['floor'] == floor]

        # Dynamically create a variable for each floor
        globals()[f'df_{floor}'] = floor_dataframes[f'df_{floor}']
 
    # Step 4: Convert the dataframe without blocks to DXF 
    dxf_data_0 = create_dxf_from_dataframe(df_0)

    # Step 5: Calculate max_along_x and max_along_y
    max_along_x, max_along_y = calculate_max_along_x_y(df_0)

    # For saving each floors DXF's.
    for floor in unique_floors:
        if floor >= 1:
            current_df = globals().get(f'df_{floor}', None)  # Dynamically access each DataFrame
            if current_df is not None and not current_df.empty:
                dxf_data = create_dxf_from_dataframe(current_df)

                # Get the offset for the current floor
                target_x = floor * base_offset  # Default offset 0 if not found

                # Shift the DXF to align its base offset.
                dxf_data = shift_dxf_to_coordinates(dxf_data, target_x=target_x, target_y=0)

                dxf_filename = f"floor_{floor}.dxf"
                dxf_data.saveas(dxf_filename)
            else:
                raise ValueError(f"DataFrame for floor {floor} is missing or empty.")

    # Check if max_along_x and max_along_y are valid
    if max_along_x is None or max_along_y is None:
        raise ValueError("max_along_x or max_along_y is None")

    # Step 6-28: Pipeline steps 
    dxf_data_0 = four_corners(dxf_data_0, max_along_x, max_along_y, width=9, height=12)
    if dxf_data_0 is None:
        raise ValueError("four_corners returned None")
    
    dxf_data_0 = Boundary_1(dxf_data_0, target_x=9, tolerance=1, width=9, height=12, max_along_y=max_along_y)
    if dxf_data_0 is None:
        raise ValueError("Boundary_1 returned None")

    dxf_data_0 = Boundary_2(dxf_data_0, width=12, height=9, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    if dxf_data_0 is None:
        raise ValueError("Boundary_2 returned None")
        
    dxf_data_0 = Boundary_3(dxf_data_0, width=9, height=12, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    if dxf_data_0 is None:
        raise ValueError("Boundary_3 returned None")
        
    dxf_data_0 = Boundary_4(dxf_data_0, width=12, height=9, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    if dxf_data_0 is None:
        raise ValueError("Boundary_4 returned None")
        
    dxf_data_0 = process_single_walls_left(df, max_along_x, max_along_y, dxf_data_0, width=12, height=9)
    if dxf_data_0 is None:
        raise ValueError("process_single_walls_left returned None")
        
    dxf_data_0 = process_single_walls_right(df, max_along_x, max_along_y, dxf_data_0, width=12, height=9)
    if dxf_data_0 is None:
        raise ValueError("process_single_walls_right returned None")
        
    dxf_data_0 = single_wall_up(df, max_along_x, max_along_y, dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("single_wall_up returned None")

    # Additional intermediate processing using DataFrames
    pipeline_results = semi_main_columns(df_0, max_along_x, max_along_y)
    df_x_equal = pipeline_results['df_x_equal']
    df_y_equal = pipeline_results['df_y_equal']
    df_other = pipeline_results['df_other']
    temp_h = pipeline_results['temp_h']
    temp_v = pipeline_results['temp_v']

    dxf_data_0 = create_boxes_in_df_x_equal(df_x_equal, temp_v, dxf_data_0, width=9, height=12, tolerance_v=0.5, radius=84)
    if dxf_data_0 is None:
        raise ValueError("create_boxes_in_df_x_equal returned None")
        
    dxf_data_0 = create_boxes_in_df_y_equal(df_y_equal, temp_h, dxf_data_0,tolerance_h=0.5, width=12, height=9, radius=84)
    if dxf_data_0 is None:
        raise ValueError("create_boxes_in_df_y_equal returned None")
    
    df_other_groupA, df_other_groupB = group_by_x(df_other)
    
    dxf_data_0 = create_boxes_in_df_other_groupA(df_other_groupA, width=12, height=9, dxf_data=dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("create_boxes_in_df_other_groupA returned None")
        
    dxf_data_0 = create_boxes_in_df_other_groupB(df_other_groupB, width=12, height=9, dxf_data=dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("create_boxes_in_df_other_groupB returned None")
        
    dxf_data_0 = detect_and_label_boxes(dxf_data_0, label_position='right', offset=1, text_height=7, text_color=0, shift=0.5)
    if dxf_data_0 is None:
        raise ValueError("detect_and_label_boxes returned None")
        
    dxf_data_0 = detect_and_remove_overlapping_columns(dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("detect_and_remove_overlapping_columns returned None")
    
    base_column_info_df = create_column_schedule_dataframe(dxf_data_0, max_along_x, max_along_y)

    # Iterate through each floor and process its DXF file
    for floor in unique_floors:
        if floor >= 1:
            # Dynamically generate the filename for the current floor
            dxf_filename = f"floor_{floor}.dxf"

            # Load the existing DXF file
            try:
                dxf_data = ezdxf.readfile(dxf_filename)

                # Get the offset for the current floor
                floor_key = f'df_{floor}'
                x_offset = floor * base_offset  # Default offset 0 if not found

                # Adjust the 'X Center' column
                column_info_df = base_column_info_df.copy()
                column_info_df['X Center'] += x_offset

                # Add boxes with hatches to the DXF file
                dxf_data = add_boxes_from_dataframe_with_hatch(dxf_data, column_info_df, label_position='right', offset=5, shift=2, text_height=7, text_color=0)

                # Save the updated DXF file
                dxf_data.saveas(dxf_filename)
            except IOError:
                logging.error(f"Failed to load {dxf_filename}. File does not exist.")

    dxf_data_0 = change_line_color_to_light_gray(dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("change_line_color_to_light_gray returned None")


    C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col = extract_columns(dxf_data_0)

    beam_count = 0
    beam_info_df = pd.DataFrame(columns=['beam names', 'length','line_1_x_start','line_1_x_end','line_1_y_start','line_1_y_end',
                                         'line_2_x_start','line_2_x_end','line_2_y_start','line_2_y_end'])

    beam_count, beam_info_df, dxf_data_0 = connect_edges_vertically_boundary_1(dxf_data_0, C1_ver_col, beam_count)
    if dxf_data_0 is None:
        raise ValueError("connect_edges_vertically_boundary_1 returned None")
    
    beam_count, beam_info_df, dxf_data_0 = connect_edges_horizontally_boundary_2(dxf_data_0, max_along_y, C2_hor_col, beam_count, beam_info_df)
    if dxf_data_0 is None:
        raise ValueError("connect_edges_horizontally_boundary_2 returned None")
        
    beam_count, beam_info_df, dxf_data_0 = connect_edges_vertically_boundary_3(dxf_data_0, C3_ver_col, beam_count, beam_info_df)
    if dxf_data_0 is None:
        raise ValueError("connect_edges_vertically_boundary_3 returned None")

    beam_count, beam_info_df, dxf_data_0 = check_horizontal_alignment_boundary_1(
        dxf_data=dxf_data_0,      
        max_along_x=max_along_x,              
        C1_ver_col=C1_ver_col,         
        beam_count=beam_count,       
        tolerance=15,                  
        beam_info_df=beam_info_df
    )
    if dxf_data_0 is None:
        raise ValueError("check_horizontal_alignment_boundary_1 returned None")

    beam_count, beam_info_df, dxf_data_0 = check_vertical_alignment_boundary_2(
        dxf_data=dxf_data_0,
        C2_hor_col=C2_hor_col,
        max_along_y=max_along_y, 
        beam_count=beam_count, 
        tolerance=15, 
        beam_info_df=beam_info_df
    )
    if dxf_data_0 is None:
        raise ValueError("check_vertical_alignment_boundary_2 returned None")

    beam_count, beam_info_df, dxf_data_0 = check_horizontal_alignment_boundary_3(
        dxf_data=dxf_data_0, 
        max_along_x=max_along_x, 
        C3_ver_col=C3_ver_col, 
        beam_count=beam_count, 
        beam_info_df=beam_info_df, 
        tolerance=15
    )
    if dxf_data_0 is None:
        raise ValueError("check_horizontal_alignment_boundary_3 returned None")

    beam_count, beam_info_df, dxf_data_0 = check_vertical_alignment_boundary_4(
        dxf_data=dxf_data_0, 
        C4_hor_col=C4_hor_col, 
        max_along_y=max_along_y, 
        beam_count=beam_count, 
        beam_info_df=beam_info_df, 
        tolerance=15
    )
    if dxf_data_0 is None:
        raise ValueError("check_vertical_alignment_boundary_4 returned None")

    filtered_columns, column_positions = read_and_filter_columns(dxf_data_0, C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col)
    alignment_data = find_aligned_columns(filtered_columns, column_positions, tolerance=7)
    left_df, right_df, up_df, down_df = get_all_alignments(alignment_data, base_column_info_df)
    left_df, beam_info_df, beam_count, dxf_data_0 = beam_and_wall_detection_left(dxf_data_0, left_df, beam_count=beam_count, beam_info_df=beam_info_df)
    right_df, beam_info_df, beam_count, dxf_data_0 = beam_and_wall_detection_right(dxf_data_0, right_df, beam_count=beam_count, beam_info_df=beam_info_df)
    up_df, beam_info_df, beam_count, dxf_data_0 = beam_and_wall_detection_up(dxf_data_0, up_df, beam_count=beam_count, beam_info_df=beam_info_df)


    # Iterate through each floor and process its DXF file
    for floor in unique_floors:
        if floor >= 1:
            # Dynamically generate the filename for the current floor
            dxf_filename = f"floor_{floor}.dxf"

            # Load the existing DXF file
            try:
                dxf_data = ezdxf.readfile(dxf_filename)

                # Get the offset for the current floor
                floor_key = f'df_{floor}'
                x_offset = floor * base_offset  # Default offset 0 if not found

                # Adjust the 'X Center' column
                beam_info_df_copy = beam_info_df.copy()
                beam_info_df_copy['line_1_x_start'] += x_offset
                beam_info_df_copy['line_1_x_end'] += x_offset
                beam_info_df_copy['line_2_x_start'] += x_offset
                beam_info_df_copy['line_2_x_end'] += x_offset

                # Add beams to the floor DXF file
                dxf_data = add_lines_and_labels_from_dataframe(dxf_data, beam_info_df_copy)

                # Save the updated DXF file
                dxf_data.saveas(dxf_filename)
            except IOError:
                logging.error(f"Failed to load {dxf_filename}. File does not exist.")
                
    # Concatenate all floors into the output file
    drawings = []
    
    # Save the in-memory DXF to a temporary file
    beam_dxf_filename = "beam_output.dxf"
    dxf_data_0.saveas(beam_dxf_filename)
    
    # Load the temporary DXF file
    drawing = ezdxf.readfile(beam_dxf_filename)
    drawings.append(drawing)
    
    dxf_files = []
    
    # Collect DXF files for each floor
    for floor in unique_floors:
        if floor >= 1:
            dxf_filename = f"floor_{floor}.dxf"
            dxf_files.append(dxf_filename)
    
    # Read and append floor-specific DXFs
    for file in dxf_files:
        try:
            drawing = ezdxf.readfile(file)
            drawings.append(drawing)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    
    # Create a new DXF document
    combined_beam_dxf = ezdxf.new()
    
    # Access the modelspace of the combined DXF
    combined_beam_msp = combined_beam_dxf.modelspace()
    
    # Combine entities
    for drawing in drawings:
        msp = drawing.modelspace()
        for entity in msp:
            try:
                combined_beam_msp.add_entity(entity.copy())  # Add a copy of the entity to the combined modelspace
            except Exception as e:
                logging.error(f"Error copying entity from {drawing}: {e}")  # Debugging

    # Save the final combined DXF
    combined_beam_dxf.saveas('combined_beam.dxf')
    print("combined_beam.dxf created.")

    # Extract only columns A, B, and C
    col_legend_df= base_column_info_df[["Column Nos", "Length", "Width"]]

    # Column legend dxf
    col_file_pos = dataframe_to_dxf_table(col_legend_df, "column_legend.dxf")
    drawings = []

    input_dxf_paths = ['combined_beam.dxf', "column_legend.dxf"]
    
    # Load all drawings
    for file_path in input_dxf_paths:
        try:
            drawing = ezdxf.readfile(file_path)
            drawings.append(drawing)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Create new document with specified version
    merged_dxf = ezdxf.new('R2010')
    merged_msp = merged_dxf.modelspace()

    offsets = [
        (0, 0, 0),      # First file at origin
        (0, -col_file_pos, 0),    # Second file offset by 1000 units in X direction
    ]
    # Combine entities with offsets
    for drawing, offset in zip(drawings, offsets):
        msp = drawing.modelspace()
        offset_x, offset_y, offset_z = offset
        
        for entity in msp:
            try:
                # Create a copy of the entity
                copied_entity = entity.copy()
                
                # Apply offset based on entity type
                if entity.dxftype() == 'LINE':
                    # Offset start and end points
                    copied_entity.dxf.start = (
                        entity.dxf.start[0] + offset_x,
                        entity.dxf.start[1] + offset_y,
                        entity.dxf.start[2] + offset_z if len(entity.dxf.start) > 2 else 0
                    )
                    copied_entity.dxf.end = (
                        entity.dxf.end[0] + offset_x,
                        entity.dxf.end[1] + offset_y,
                        entity.dxf.end[2] + offset_z if len(entity.dxf.end) > 2 else 0
                    )
                
                elif entity.dxftype() == 'TEXT':
                    # Offset text insertion point
                    copied_entity.dxf.insert = (
                        entity.dxf.insert[0] + offset_x,
                        entity.dxf.insert[1] + offset_y,
                        entity.dxf.insert[2] + offset_z if len(entity.dxf.insert) > 2 else 0
                    )
                
                # Add the offset entity to merged modelspace
                merged_msp.add_entity(copied_entity)
                
            except Exception as e:
                print(f"Error copying entity: {e}")
    
    # Save the final dxf
    try:
        merged_dxf.saveas(output_file)
        print(f"Successfully saved merged file to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    # Return both output_filename and column_info_df
    return base_column_info_df , beam_info_df

# pipeline_main_final('smh_file_Testing1_multifloor.dxf', 'first_merged.dxf')
