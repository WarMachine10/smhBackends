
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
import os


# Suppress SettingWithCopyWarning from pandas
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure logging to only show warnings and above
logging.getLogger('ezdxf').setLevel(logging.WARNING)
logging.getLogger('root').setLevel(logging.WARNING)

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

# %%
def calculate_length(start, end):
    """Calculates the 3D length between two points with error handling for missing coordinates."""
    try:
        return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2 + (end.z - start.z)**2)
    except AttributeError as e:
        logging.error(f"Error calculating length: {e}")
        # Return a default length of 0 if start or end points are malformed
        return 0.0

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
            logging.info(f"Processing entity on layer '{entity.dxf.layer}'")
        except AttributeError:
            entity_data['Layer'] = 'Unknown'
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

# %%
from typing import List, Tuple

def calculate_ranges(df: pd.DataFrame):
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

# %%
def create_dxf_from_dataframe(df):
    doc = ezdxf.new()
    layers = {}

    for index, row in df.iterrows():
        layer_name = str(row['Layer'])  

        if not layer_name or layer_name.lower() == 'nan': 
            continue

        if layer_name not in layers and layer_name != '0': 
            layers[layer_name] = doc.layers.new(name=layer_name)  

        msp = doc.modelspace()
        
        if row['Type'] == 'LINE':
            start = (row['X_start'], row['Y_start'])
            end = (row['X_end'], row['Y_end'])
            msp.add_line(start, end, dxfattribs={'layer': layer_name})

        elif row['Type'] == 'CIRCLE':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            msp.add_circle(center, radius, dxfattribs={'layer': layer_name})

        elif row['Type'] == 'ARC':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            start_angle = row['Start Angle']
            end_angle = row['End Angle']
            msp.add_arc(center, radius, start_angle, end_angle, dxfattribs={'layer': layer_name})

        elif row['Type'] == 'TEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row.get('Text Content', '')
            msp.add_text(text, dxfattribs={'insert': insert, 'layer': layer_name})

        elif row['Type'] == 'MTEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row.get('Text Content', '')
            msp.add_mtext(text, dxfattribs={'insert': insert, 'layer': layer_name})
        
        elif row['Type'] == 'SPLINE':
            fit_point_count = int(row['Fit Point Count']) if not pd.isna(row['Fit Point Count']) else 0
            fit_points = [(row[f'X{i}'], row[f'Y{i}'], row[f'Z{i}']) for i in range(fit_point_count)]
            msp.add_spline(fit_points, dxfattribs={'layer': layer_name})
        
        elif row['Type'] == 'ELLIPSE':
            center = (row['X_center'], row['Y_center'])
            major_axis = (row['X_major_axis'], row['Y_major_axis'])
            ratio = row['Ratio']
            msp.add_ellipse(center, major_axis, ratio, dxfattribs={'layer': layer_name})
            
        elif row['Type'] == 'POINT':
            location = (row['X_start'], row['Y_start'])
            msp.add_point(location, dxfattribs={'layer': layer_name})
                                                
    return doc

# %%
def calculate_max_along_x_y(df):
    # Filter the DataFrame for rows where 'Layer' is 'Boundary'
    boundary_df = df[df['Layer'] == 'Boundary']
    
    # Calculate max_along_x and max_along_y for the filtered DataFrame
    max_along_x = np.round(np.max(np.abs(boundary_df['X_end'] - boundary_df['X_start'])), 1)
    max_along_y = np.round(np.max(np.abs(boundary_df['Y_end'] - boundary_df['Y_start'])), 1)

    # Return the values for later use
    return max_along_x, max_along_y

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
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    return dxf_data  # Ensure dxf_data is returned after modification

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
            hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    # Return the modified dxf_data object
    return dxf_data

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
                hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})  # Red color
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

        msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1, 'layer': 'column'})

        hatch = msp.add_hatch(color=1, dxfattribs={'layer': 'column'})
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

# %%
def detect_and_label_boxes(dxf_data, label_position='right', offset=1, text_height=5, text_color=0, shift=0):
    '''
    Detects boxes (closed polylines) in a DXF file, labels them around the box (right, left, or top),
    and updates the DXF data in memory. Supports hypertuning for label position, text height, color, and shifts.

    Parameters:
    (1) dxf_data: The DXF data object to modify (ezdxf.document).
    (2) label_position: The position of the label relative to the box ('right', 'left', 'top').
    (3) offset: The distance from the box to the label (float, default is 1).
    (4) text_height: The height of the label text (float, default is 5).
    (5) text_color: The color of the label text (DXF color index, default is 0 - black).
    (6) shift: Shift the text position (float, default is 0, can adjust the left or right position).

    Returns:
    None: The function updates the DXF data in memory.
    '''
    # Access the modelspace of the provided dxf_data
    msp = dxf_data.modelspace()

    # Add Arial text style to the DXF file
    dxf_data.styles.new('ArialStyle', dxfattribs={'font': 'arial.ttf'})
    
    box_number = 1

    # Iterate over all entities in the modelspace
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE' and entity.closed:
            # This entity is a closed polyline (a box)

            # Extract the polyline points to calculate the positions
            points = entity.get_points('xy')  # Get points as (x, y) tuples

            # Ensure it's a rectangular box (typically 5 points: 4 corners + closing point)
            if len(points) == 5 and points[0] == points[-1]:
                # Calculate the corners of the box
                min_x = min([p[0] for p in points])
                max_x = max([p[0] for p in points])
                min_y = min([p[1] for p in points])
                max_y = max([p[1] for p in points])

                # Determine where to place the label
                if label_position == 'right':
                    label_x = max_x + offset + shift  # Right of the box + optional shift
                    label_y = (min_y + max_y) / 2  # Vertically centered
                elif label_position == 'left':
                    label_x = min_x - offset - shift  # Left of the box + optional shift
                    label_y = (min_y + max_y) / 2  # Vertically centered
                elif label_position == 'top':
                    label_x = (min_x + max_x) / 2  # Horizontally centered
                    label_y = max_y + offset  # Above the box
                else:
                    raise ValueError("label_position must be 'right', 'left', or 'top'")

                # Create a name for the box like 'C1', 'C2', etc.
                box_name = f"C{box_number}"
                box_number += 1

                # Add text at the calculated position
                add_mtext(msp, box_name, (label_x, label_y), text_height, text_color)

    # Return the modified dxf_data object (without saving to file yet)
    return dxf_data

def add_mtext(msp, text, position, height=5, color=0):
    '''
    Adds MText (multiline text) to the DXF file at the specified position with adjustable height and color.

    Parameters:
    (1) msp: The modelspace of the DXF document (ezdxf.modelspace).
    (2) text: The text to add (string).
    (3) position: The position to place the text (tuple of (x, y)).
    (4) height: The height of the text (float, default is 5).
    (5) color: The color of the text (DXF color index, default is 0 - black).

    Returns:
    None: This function does not return any values.
    '''
    
    mtext = msp.add_mtext(text, dxfattribs={"style": "ArialStyle", 'char_height': height, 'color': color, "layer": "column"})
    mtext.set_location(position)


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
        'Columns No': column_names,
        'Length': lengths,
        'Width': widths,
        'X Center': x_centers,
        'Y Center': y_centers
    })
    
    return df

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

# %%
def remove_non_label_text(input_dxf, output_dxf, labels_to_keep):
    """
    Remove all text and mtext entities except for specific labels like C1, C2, etc.

    Parameters:
        input_dxf (str): Path to the input DXF file.
        output_dxf (str): Path to save the updated DXF file.
        labels_to_keep (list): List of text labels to keep (e.g., ["C1", "C2"]).

    Returns:
        None
    """
    # Load the DXF file
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

    # Loop through all text entities (TEXT and MTEXT)
    for entity in msp.query('TEXT MTEXT'):
        # Check if the text matches any of the labels in the list
        if entity.dxftype() == 'TEXT':
            if entity.text not in labels_to_keep:
                msp.delete_entity(entity)  # Remove TEXT entity
        elif entity.dxftype() == 'MTEXT':
            if entity.text not in labels_to_keep:
                msp.delete_entity(entity)  # Remove MTEXT entity

    # Save the updated DXF file
    doc.saveas(output_dxf)
    
# Example usage
def process_dxf_with_labels(input_dxf, output_dxf, df):
    """
    Process the DXF file and remove text/mtext entities that are not in the 'Columns No' column of the dataframe.

    Parameters:
        input_dxf (str): Path to the input DXF file.
        output_dxf (str): Path to save the updated DXF file.
        df (pd.DataFrame): DataFrame containing the 'Columns No' column with labels to keep.

    Returns:
        None
    """
    # Extract labels from the 'Columns No' column
    labels_to_keep = df['Columns No'].tolist()

    # Call the function to remove non-label text from the DXF
    remove_non_label_text(input_dxf, output_dxf, labels_to_keep)

# %% [markdown]
# # Center line core code

# %%
def draw_x_start_point(input_dxf, max_along_y):
    """
    Modify the input DXF document by adding a start point marker.
    
    Args:   
        input_dxf (ezdxf.drawing.Drawing or str): Input DXF document or path to DXF file
        max_along_y (float): Y-coordinate for positioning the start point marker
    
    Returns:
        ezdxf.drawing.Drawing: Modified DXF document
    """
    try:
        # Step 1: Read the DXF file safely
        doc = ezdxf.readfile(input_dxf)
        
        # Step 2: Check for corruption
        auditor = doc.audit()
        if auditor.has_errors:
            print("Warning: DXF file has corruption issues. It may not be fully valid.")
    except ezdxf.DXFStructureError as e:
        print(f"Error: The DXF file is corrupt or unreadable. Details: {e}")
        return None  # Return None if file is corrupted
    
    msp = doc.modelspace()
    
    # Define coordinates for the dotted vertical line
    start_point = (0, max_along_y)
    end_point = (0, max_along_y + 70)  # plot's length + 50 inches downward dotted line
    
    # Draw a dotted vertical line
    msp.add_line(start=start_point, end=end_point, dxfattribs={"linetype": "DOT", 'color':1})
    
    # Define the buffer distance (10 inches) to offset the MText from dotted line
    buffer_distance = 10
    mtext_position = (end_point[0] - 10, end_point[1])
    
    # Define the MText content and parameters
    mtext_content = "START POINT = 0"
    mtext_rotation = 90  # Rotate text by 90 degrees
    mtext_height = 8  # Approximate text height
    
    # Add rotated MText at the position 0.5 inches away from the end of the dotted line
    mtext_entity = msp.add_mtext(
        mtext_content,
        dxfattribs={
            "style": "ArialStyle",
            "char_height": mtext_height,  # Set character height
            "layer": "center_line_mtext",
            "width": 100},
    )
    mtext_entity.set_location(mtext_position)  # Set MText position
    mtext_entity.dxf.rotation = mtext_rotation  # Set the rotation
    
    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")
    
    return doc

# %%
def inches_to_feet_and_inches(value_in_inches):
    """Convert inches to feet and inches as a string (e.g., 2'5")."""
    feet = int(value_in_inches // 12)
    inches = round(value_in_inches % 12)  # Changed to match the original function
    return f"{feet}'{inches}\""

def draw_center_lines_with_X_distance(doc, df, max_along_x, max_along_y):
    """
    Draw vertical dotted red lines from column centers in a DXF document,
    and place column names (from 'Columns No') along with the distance from X = 0.
    
    Args:
        input_dxf (ezdxf.drawing.Drawing or str): Input DXF document or path to DXF file
        df (pd.DataFrame): DataFrame containing column information
        max_along_x (float): Maximum X coordinate for end point
        max_along_y (float): Maximum Y coordinate for line placement
    
    Returns:
        ezdxf.drawing.Drawing: Modified DXF document
    """
        
    msp = doc.modelspace()
    
    # Extract unique X Center values
    df['X Center'] = df['X Center'].round(1)
    unique_x_centers = sorted(df['X Center'].unique())
    
    # Define Y-coordinate range for the vertical lines
    y_start = 0
    y_end = max_along_y + 50
    
    # Store the actual text positions (not the line positions)
    text_positions = []
    
    # Iterate through sorted unique X Center values
    for idx, x_center in enumerate(unique_x_centers):
        # Draw vertical dotted red line
        msp.add_line(
            start=(x_center, y_start),
            end=(x_center, y_end),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )
        
        # Handle text positioning
        text_x = x_center
        text_y = y_end + 30
        
        # Check for overlaps with previous text positions
        while any(math.sqrt((text_x - px) ** 2 + (text_y - py) ** 2) < 15 for px, py in text_positions):
            text_x += 10  # Shift text to the right to avoid overlap
        
        # Extend vertical line and bend northeast
        extended_y = y_end + 20  # Extend 20 inches vertically
        msp.add_line(
            start=(x_center, y_end),
            end=(x_center, extended_y),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )
        
        msp.add_line(
            start=(x_center, extended_y),
            end=(text_x, text_y),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )
        
        # Store the final text position
        text_positions.append((text_x, text_y))
        
        # Get column labels and create text
        column_labels = df[df['X Center'] == x_center]['Columns No'].tolist()
        distance = inches_to_feet_and_inches(x_center)
        column_labels_text = ', '.join(column_labels) + f" = {distance}"
        
        # Add the text to the drawing
        mtext_entity = msp.add_mtext(
            column_labels_text,
            dxfattribs={
                "style": "ArialStyle",
                "char_height": 5,
                "width": 200,
                "layer": "center_line_text",
            },
        )
        mtext_entity.set_location((text_x, text_y))
        mtext_entity.dxf.rotation = 90
    
    # Draw the initial vertical line (matching the original function's coordinates)
    start_point = (max_along_x, max_along_y)
    end_point = (max_along_x, max_along_y + 50)
    msp.add_line(
        start=start_point,
        end=end_point,
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Get the last text position's x-coordinate and add 20
    last_text_x = text_positions[-1][0] + 20 if text_positions else max_along_x + 20
    
    # Convert max_along_x from inches to feet and inches using the correct format
    result = inches_to_feet_and_inches(max_along_x)
    
    # Define the END POINT text position and parameters
    buffer_distance = 10
    mtext_y = y_end + 30  # Match the height of other text elements
    mtext_position = (last_text_x, mtext_y)
    
    # Draw the extended vertical line and northeast bend for END POINT
    extended_y = y_end + 20  # Extended y-coordinate for the bend
    
    # Draw the extended vertical portion
    msp.add_line(
        start=end_point,
        end=(max_along_x, extended_y),
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Draw the northeast bend to the text position
    msp.add_line(
        start=(max_along_x, extended_y),
        end=mtext_position,
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Add the END POINT text
    mtext_content = f"END POINT = {result}"
    mtext_entity = msp.add_mtext(
        mtext_content,
        dxfattribs={
            "style": "ArialStyle",
            "char_height": 8,
            "width": 200,
            "layer": "center_line_text",
        },
    )
    mtext_entity.set_location(mtext_position)
    mtext_entity.dxf.rotation = 90

    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")
    
    return doc


# %%
def draw_y_start_point(doc, max_along_x, max_along_y):
    # Load the input DXF file
    msp = doc.modelspace()

    # Define coordinates for the dotted horizontal line
    start_point = (max_along_x, max_along_y)
    end_point = (max_along_x + 80, max_along_y )  # plot's width + 50 inches towards right dotted line

    # Draw a dotted vertical line
    msp.add_line(start=start_point, end=end_point, dxfattribs={"linetype": "DOT", 'color' : 1})

    # Define the buffer distance (10 inches) to offset the MText from dotted line
    buffer_distance = 10
    mtext_position = (end_point[0] + buffer_distance, end_point[1] + 10 )

    # Define the MText content and parameters
    mtext_content = "START POINT = 0"
    mtext_height = 8  # Approximate text height 

    # Add rotated MText at the position 0.5 inches away from the end of the dotted line
    mtext_entity = msp.add_mtext(
        mtext_content,
        dxfattribs={
            "style": "ArialStyle",
            "char_height": mtext_height,  # Set character height
            "width": 100,  # Initial width; will be adjusted later
            "layer": "center_line_text",
        },
    )
    mtext_entity.set_location(mtext_position)  # Set MText position

    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")

    return doc

# %%
def inches_to_feet_and_inches(value_in_inches):
    """Convert inches to feet and inches as a string (e.g., 0'4.5")."""
    feet = int(value_in_inches // 12)
    inches = round(value_in_inches % 12, 1)  # Round to 1 decimal place
    if inches == 0:
        return f"{feet}'"
    return f"{feet}'{inches}\""

def draw_center_lines_with_Y_distance(doc, df, max_along_x, max_along_y):
    """
    Draw horizontal dotted red lines from column centers in a DXF file,
    and place column names (from 'Columns No') beside the corresponding dotted line.
    """
    msp = doc.modelspace()

    # Extract unique Y Center values
    df['Y Center'] = df['Y Center'].round(1)
    unique_y_centers = df['Y Center'].unique()

    # Sort Y center values in descending order
    unique_y_centers = sorted(unique_y_centers, reverse=True)

    # Define Y-coordinate range for the horizontal lines
    x_start = 0
    x_end = max_along_x + 50

    # Store the actual text positions (not the line positions)
    text_positions = []
    
    # Iterate through sorted unique Y Center values
    for idx, y_center in enumerate(unique_y_centers):
        # Draw a horizontal dotted red line
        msp.add_line(
            start=(x_start, y_center),
            end=(x_end, y_center),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )

        # Handle text positioning
        text_x = x_end + 30
        text_y = y_center

        # Check for overlaps with previous text positions
        while any(math.sqrt((text_x - px) ** 2 + (text_y - py) ** 2) < 15 for px, py in text_positions):
            text_y -= 10

        # Extend vertical line and bend northeast
        extended_x = x_end + 20
        msp.add_line(
            start=(x_end, y_center),
            end=(extended_x, y_center),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )

        msp.add_line(
            start=(extended_x, y_center),
            end=(text_x, text_y),
            dxfattribs={
                'color': 1,
                'linetype': 'DOT',
            }
        )

        # Store the final text position
        text_positions.append((text_x, text_y))

        # Get column labels and create text
        column_labels = df[df['Y Center'] == y_center]['Columns No'].tolist()
        distance_from_max_along_y = abs(max_along_y - y_center)
        distance = inches_to_feet_and_inches(distance_from_max_along_y)
        column_labels_text = ', '.join(column_labels) + f" = {distance}"

        # Add the text to the drawing
        mtext_entity = msp.add_mtext(
            column_labels_text,
            dxfattribs={"style": "ArialStyle",
                "layer": "center_line_text",
                "char_height": 5,
                "width": 200,
            },
        )
        mtext_entity.set_location((text_x, text_y))
        mtext_entity.dxf.rotation = 0

    # Define coordinates for the dotted horizontal line
    start_point = (max_along_x, 0)
    end_point = (max_along_x + 50, 0)
    msp.add_line(
        start=start_point,
        end=end_point,
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Get the last unique y-center value and subtract 20 for the final text position
    last_y_center = unique_y_centers[-1] - 20 if len(unique_y_centers) > 0 else 0
    
    # Use the same x position as other texts
    mtext_x = x_end + 30
    mtext_position = (mtext_x, last_y_center)
    
    # Convert max_along_y from inches to feet and inches
    result = inches_to_feet_and_inches(max_along_y)
    
    # Draw the extended horizontal portion
    extended_x = x_end + 20
    
    # Draw the extended horizontal line
    msp.add_line(
        start=end_point,
        end=(extended_x, 0),
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Draw the southeast bend to the text position
    msp.add_line(
        start=(extended_x, 0),
        end=mtext_position,
        dxfattribs={
            'color': 1,
            'linetype': 'DOT',
        }
    )
    
    # Add the END POINT text
    mtext_content = f"END POINT = {result}"
    mtext_entity = msp.add_mtext(
        mtext_content,
        dxfattribs={"style": "ArialStyle",
            "char_height": 8,
            "width": 200,
            "layer": "center_line_text",
        },
    )
    mtext_entity.set_location(mtext_position)
    mtext_entity.dxf.rotation = 0

    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")
    
    return doc

# %%
def draw_diagonal_1_with_distance(doc, max_along_x, max_along_y):
    """
    Draw a diagonal line from (0,0) to (max_along_x, max_along_y) in a DXF file,
    calculate its length in feet and inches, and write the distance diagonally above the line.

    Parameters:
        input_dxf (str): Path to the input DXF file.
        output_dxf (str): Path to save the output DXF file.
        max_along_x (float): Maximum X-coordinate of the diagonal line.
        max_along_y (float): Maximum Y-coordinate of the diagonal line.

    Returns:
        None
    """
    # Load the DXF file
    msp = doc.modelspace()

    # Start and end points of the diagonal
    start_point = (0, 0)
    end_point = (max_along_x, max_along_y)

    # Draw the diagonal line (black color)
    msp.add_line(
        start=start_point,
        end=end_point,
        dxfattribs={'color': 1}  # Black color
    )

    # Calculate the length of the diagonal
    diagonal_length = math.sqrt(max_along_x**2 + max_along_y**2)

    # Convert diagonal length to feet and inches
    def inches_to_feet_and_inches(value_in_inches):
        feet = int(value_in_inches // 12)
        inches = round(value_in_inches % 12, 1)
        if inches == 0:
            return f"{feet}'"
        return f"{feet}'{inches}\""

    distance_text = inches_to_feet_and_inches(diagonal_length)

    # Calculate the midpoint of the diagonal
    old_midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    midpoint= ( old_midpoint[0] + 5, old_midpoint [1])
    
    # Calculate the angle of the diagonal line in degrees
    angle = math.degrees(math.atan2(max_along_y, max_along_x))

    # Offset the text position slightly above the diagonal line
    offset_distance = 15  # Adjust this value for more or less offset
    offset_x = offset_distance * math.cos(math.radians(angle + 90))
    offset_y = offset_distance * math.sin(math.radians(angle + 90))
    text_position = (midpoint[0] + offset_x, midpoint[1] + offset_y)

    # Add the distance text as MTEXT, rotated to align with the diagonal
    mtext_entity = msp.add_mtext(
        text=distance_text,
        dxfattribs={"style": "ArialStyle",
            "char_height": 8,  # Set character height
            "layer": "center_line_text",
        }
    )
    mtext_entity.set_location(text_position, rotation=angle)

    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")

    return doc 


# %%
def draw_diagonal_2_with_distance(doc, max_along_x, max_along_y):
    """
    Draw a diagonal line from (0,max_along_y) to (max_along_x, 0) in a DXF file,
    calculate its length in feet and inches, and write the distance diagonally above the line.

    Parameters:
        input_dxf (str): Path to the input DXF file.
        output_dxf (str): Path to save the output DXF file.
        max_along_x (float): Maximum X-coordinate of the diagonal line.
        max_along_y (float): Maximum Y-coordinate of the diagonal line.

    Returns:
        None
    """
    msp = doc.modelspace()

    # Start and end points of the diagonal
    start_point = (0, max_along_y)
    end_point = (max_along_x, 0)

    # Draw the diagonal line (black color)
    msp.add_line(
        start=start_point,
        end=end_point,
        dxfattribs={'color': 1}  # Black color
    )

    # Calculate the length of the diagonal
    diagonal_length = math.sqrt(max_along_x**2 + max_along_y**2)

    # Convert diagonal length to feet and inches
    def inches_to_feet_and_inches(value_in_inches):
        feet = int(value_in_inches // 12)
        inches = round(value_in_inches % 12, 1)
        if inches == 0:
            return f"{feet}'"
        return f"{feet}'{inches}\""

    distance_text = inches_to_feet_and_inches(diagonal_length)

    # Calculate the midpoint of the diagonal
    old_midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    midpoint=(old_midpoint[0] + 12, old_midpoint[1]-18)
    
    # Calculate the angle of the diagonal line in degrees
    angle = math.degrees(math.atan2(max_along_y, max_along_x))

    # Offset the text position slightly above the diagonal line in the southeast direction
    text_position = (midpoint[0] , midpoint[1] )

    # Add the distance text as MTEXT, rotated to align with the diagonal
    mtext_entity = msp.add_mtext(
        text=distance_text,
        dxfattribs={"style": "ArialStyle",
            "char_height": 8,  # Set character height
            "layer": "center_line_text",
        }
    )
    mtext_entity.set_location(text_position, rotation=angle-90)

    # Final validation before returning
    auditor = doc.audit()
    if auditor.has_errors:
        print("Warning: Modified DXF object has issues. Check before further processing.")

    return doc

def validate_dxf_entities(doc):
    for entity in doc.modelspace():
        if hasattr(entity, "is_valid"):
            if not entity.is_valid:
                print(f"Warning: Invalid entity found: {entity.dxftype()}")

def validate_coordinates(doc):
    for entity in doc.modelspace():
        if hasattr(entity, "dxf"):
            for key, value in entity.dxf.all_existing_dxf_attribs().items():
                if isinstance(value, (float, int)):
                    if not math.isfinite(value):
                        raise ValueError(f"Invalid coordinate found in {entity.dxftype()}: {key}={value}")

def save_dxf_safely(doc, output_dxf):
    try:
        # Set the DXF version to AutoCAD 2010
        doc.header['$ACADVER'] = 'AC1024'
        
        # Save the file
        doc.saveas(output_dxf)
        
        # Verify the saved file
        test_doc = ezdxf.readfile(output_dxf)
                
    except ezdxf.DXFStructureError as e:
        raise ValueError(f"Error: {output_dxf} is corrupt or unreadable. Details: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error while saving {output_dxf}: {str(e)}")

def center_line_main_new(input_dxf, output_dxf):
    # Step 1: Adjust coordinates 
    adjust_dxf_coordinates_to00(input_dxf, 'temp_1.dxf')

    # Step 2: Set transparency
    set_transparency_for_all_entities('temp_1.dxf', 'temp_2.dxf', transparency_percent = 50)

    # Step 3: Dxf to dataframe 
    df = Dxf_to_DF_1('temp_2.dxf')

    # Step 4: Remove blocks
    df = df[df['Type'] != 'INSERT']

    # Step 5: Distribute floors
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

    dxf_data_0 = detect_and_label_boxes(dxf_data_0, label_position='right', offset=1, text_height=5, text_color=0, shift=0.5)
    if dxf_data_0 is None:
        raise ValueError("detect_and_label_boxes returned None")
        
    dxf_data_0 = detect_and_remove_overlapping_columns(dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("detect_and_remove_overlapping_columns returned None")

    df = create_column_schedule_dataframe(dxf_data_0, max_along_x, max_along_y)

    dxf_data_0 = change_line_color_to_light_gray(dxf_data_0)
    if dxf_data_0 is None:
        raise ValueError("change_line_color_to_light_gray returned None")

    # Save the in-memory DXF to a temporary file
    col_dxf_filename = "col_output.dxf"
    dxf_data_0.saveas(col_dxf_filename)

    process_dxf_with_labels("col_output.dxf", 'Clean_col_output.dxf', df)
                                      
    doc = draw_x_start_point('Clean_col_output.dxf', max_along_y = max_along_y)

    doc = draw_center_lines_with_X_distance(doc, df, max_along_x=max_along_x, max_along_y=max_along_y)

    doc = draw_y_start_point(doc, max_along_x = max_along_x, max_along_y=max_along_y)

    doc = draw_center_lines_with_Y_distance(doc, df, max_along_x= max_along_x, max_along_y = max_along_y)

    doc = draw_diagonal_1_with_distance(doc, max_along_x = max_along_x, max_along_y = max_along_y)

    doc = draw_diagonal_2_with_distance(doc, max_along_x = max_along_x, max_along_y = max_along_y)

    # Set units and measurement system
    doc.header['$MEASUREMENT'] = 1  # Sets units to metric
    doc.header['$INSUNITS'] = 4    # Sets units to millimeters

    # Validate entities and coordinates
    validate_dxf_entities(doc)
    validate_coordinates(doc)

    # Save using the safe saving function
    save_dxf_safely(doc, output_dxf)

    # Additional check for minimal required DXF content
    try:
        with open(output_dxf, 'r') as f:
            content = f.read()
            if "ENTITIES" not in content or "HEADER" not in content:
                raise ValueError("DXF file missing critical sections")
    except Exception as e:
        raise ValueError(f"Error verifying DXF content: {str(e)}")

    # Validation: Ensure the saved file is not corrupt
    if not os.path.exists(output_dxf):
        raise FileNotFoundError(f"Error: {output_dxf} was not created.")

    try:
        test_doc = ezdxf.readfile(output_dxf)
        print(f"Success: {output_dxf} was saved and is valid.")
    except ezdxf.DXFStructureError as e:
        raise ValueError(f"Error: {output_dxf} is corrupt or unreadable. Details: {e}")


# center_line_main_new('smh_file_Testing1_multifloor.dxf', 'final_center_line_plan.dxf')
