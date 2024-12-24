from django.conf import settings
import pandas as pd
import numpy as np
import json
import ezdxf
import geopandas as gpd
import math
import matplotlib.pyplot as plt 
import random
import seaborn as sns
import math
from loguru import logger
import matplotlib.pyplot as plt
import os
# import fitz
import subprocess
# import ezd
from PIL import Image
from io import BytesIO
#from openai import OpenAIa
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder  
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
from matplotlib.patches import Arc, Circle
pd.set_option('display.float_format', '{:.5f}'.format)





def adjust_dxf_coordinates_to00(filename):
    # Read the DXF file
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()

    # Find the minimum X and Y coordinates
    min_x_entities = [entity.dxf.center.x if entity.dxftype() == 'CIRCLE' else
                      entity.dxf.center.x - entity.dxf.radius if entity.dxftype() == 'ARC' else
                      entity.dxf.insert.x if entity.dxftype() in ['TEXT', 'MTEXT'] else
                      min(entity.dxf.start.x, entity.dxf.end.x) if entity.dxftype() == 'LINE' else
                      None
                      for entity in msp]
    min_y_entities = [entity.dxf.center.y if entity.dxftype() == 'CIRCLE' else
                      entity.dxf.center.y - entity.dxf.radius if entity.dxftype() == 'ARC' else
                      entity.dxf.insert.y if entity.dxftype() in ['TEXT', 'MTEXT'] else
                      min(entity.dxf.start.y, entity.dxf.end.y) if entity.dxftype() == 'LINE' else
                      None
                      for entity in msp]
    min_x = min(x for x in min_x_entities if x is not None)
    min_y = min(y for y in min_y_entities if y is not None)

    # Calculate the offset
    offset_x = -min_x
    offset_y = -min_y

    # Update all coordinates in the DXF file
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            start = (start.x + offset_x, start.y + offset_y, start.z)
            end = (end.x + offset_x, end.y + offset_y, end.z)
            entity.dxf.start = start
            entity.dxf.end = end
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            center = (center.x + offset_x, center.y + offset_y, center.z)
            entity.dxf.center = center
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            center = (center.x + offset_x, center.y + offset_y, center.z)
            entity.dxf.center = center
        elif entity.dxftype() in ['TEXT', 'MTEXT']:
            insert = entity.dxf.insert
            insert = (insert.x + offset_x, insert.y + offset_y, insert.z)
            entity.dxf.insert = insert

    # Save the modified DXF file
    output_filename = 'new'+filename
    doc.saveas(output_filename)

# Example usage
def calculate_length(start, end):
    return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)

#DXF to PANDAS DATAFRAME
def Dxf_to_DF(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    
    entities_data = []
    for entity in msp:
        entity_data = {'Type': entity.dxftype(), 'Layer': entity.dxf.layer}
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            length = calculate_length(start, end)
            entity_data.update({
                'X_start': start.x, 'Y_start': start.y, 'Z_start': start.z,
                'X_end': end.x, 'Y_end': end.y, 'Z_end': end.z,
                'Length': length})
            horizontal = abs(end.x - start.x) > abs(end.y - start.y)
            vertical = not horizontal
            entity_data.update({'Horizontal': horizontal, 'Vertical': vertical})
            
            
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Radius': radius})
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Radius': radius,
                'Start Angle': start_angle,
                'End Angle': end_angle})
            
        elif entity.dxftype() == 'TEXT':
            insert = entity.dxf.insert
            text = entity.dxf.text
            entity_data.update({
                'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                'Text': text})
        elif entity.dxftype() == 'MTEXT':
            text = entity.plain_text()
            insertion_point = entity.dxf.insert
            entity_data.update({
                'Text': text,
                'X_insert': insertion_point.x,
                'Y_insert': insertion_point.y,
                'Z_insert': insertion_point.z
            })
            
            
        entities_data.append(entity_data)
    
    return pd.DataFrame(entities_data)
def adjust_Xstart_ystart(df):
    df_copy = df.copy()  # Create a copy of the original DataFrame
    # Swap X_start and X_end if X_start is greater than X_end
    df_copy.loc[df_copy['X_start'] > df_copy['X_end'], ['X_start', 'X_end']] = df_copy.loc[df_copy['X_start'] > df_copy['X_end'], ['X_end', 'X_start']].values
    # Swap Y_start and Y_end if Y_start is greater than Y_end
    df_copy.loc[df_copy['Y_start'] > df_copy['Y_end'], ['Y_start', 'Y_end']] = df_copy.loc[df_copy['Y_start'] > df_copy['Y_end'], ['Y_end', 'Y_start']].values
    return df_copy

def plot_dataframe(df):
    """
    Plots entities from a DataFrame using matplotlib.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the entities to be plotted.
    Returns:
        None
    """
    fig, ax = plt.subplots()
    for _, entity in df.iterrows():
        try:
            if entity['Type'] == 'LINE':
                start = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Convert inches to feet
                end = (entity['X_end'] / 12, entity['Y_end'] / 12)  # Convert inches to feet
                ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)
            elif entity['Type'] == 'CIRCLE':
                center = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Assuming center coordinates are in X_start, Y_start
                radius = entity['Length'] / 12  # Assuming Length column represents radius for circles
                circle = Circle(center, radius, color='black', fill=False, linewidth=1)
                ax.add_artist(circle)
            elif entity['Type'] == 'ARC':
                center = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Assuming center coordinates are in X_start, Y_start
                radius = entity['Length'] / 12  # Assuming Length column represents radius for arcs
                start_angle = entity['X_end']  # Assuming start angle is in X_end
                end_angle = entity['Y_end']  # Assuming end angle is in Y_end
                
                # Check for NaN values
                if np.isnan(start_angle) or np.isnan(end_angle):
                    print(f"Skipping ARC due to NaN angle: {entity}")
                    continue
                
                arc = Arc(xy=center, 
                          width=2*radius, 
                          height=2*radius, 
                          angle=0,
                          theta1=start_angle, 
                          theta2=end_angle, 
                          color='black', 
                          linewidth=1)
                ax.add_artist(arc)
            elif entity['Type'] == 'TEXT':
                insert = (entity['X_insert'] / 12, entity['Y_insert'] / 12)  # Convert inches to feet
                text = entity['Text']
                ax.text(insert[0], insert[1], text, fontsize=4, color='black')
            elif entity['Type'] == 'MTEXT':
                insert = (entity['X_insert'] / 12, entity['Y_insert'] / 12)  # Convert inches to feet
                text = entity['Text']
                ax.text(insert[0], insert[1], text, fontsize=2, color='black')
        except Exception as e:
            print(f"Error processing entity: {entity}")
            print(f"Error message: {str(e)}")
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (feet)')
    ax.set_ylabel('Y (feet)')
    ax.set_title('Architectural Plan')
    ax.tick_params(axis='both', direction='inout', which='both')
    
    # Set x and y axis ticks in feet
    x_ticks = [i * 10 for i in range(int(ax.get_xlim()[0] / 10), int(ax.get_xlim()[1] / 10) + 1)]
    y_ticks = [i * 10 for i in range(int(ax.get_ylim()[0] / 10), int(ax.get_ylim()[1] / 10) + 1)]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    plt.savefig(image_name, dpi=300)
    plt.show()
    
def remove_polylines(dxf_path, output_path):
    """
    Remove all LWPOLYLINE and POLYLINE entities from a DXF file and save to a new file.
    
    Parameters:
    dxf_path (str): Path to the original DXF file.
    output_path (str): Path to save the modified DXF file without LWPOLYLINE and POLYLINE entities.
    """
    try:
        # Read the DXF file
        doc = ezdxf.readfile(dxf_path)
        modelspace = doc.modelspace()
        
        # Delete all LWPOLYLINE entities
        lwpolylines = list(modelspace.query('LWPOLYLINE'))
        print(f"Found {len(lwpolylines)} LWPOLYLINE entities. Deleting them...")
        for lwpolyline in lwpolylines:
            modelspace.delete_entity(lwpolyline)
        
        # Delete all POLYLINE entities
        polylines = list(modelspace.query('POLYLINE'))
        print(f"Found {len(polylines)} POLYLINE entities. Deleting them...")
        for polyline in polylines:
            modelspace.delete_entity(polyline)
        
        # Save the modified DXF file
        doc.saveas(output_path)
        print(f"Removed all LWPOLYLINE and POLYLINE entities and saved to {output_path}")
    
    except ezdxf.DXFStructureError:
        print("Invalid or corrupted DXF file")
    except IOError:
        print("File not found or not accessible")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Run the function with your DXF file path and desired output path
#lwpolyline_to_lines(dxf_path, 'output_file.dxf')

# Remove Overlapped Lines and Connecting points

def get_line_length(line):
    """Calculate actual length of line"""
    if line['Horizontal']:
        return abs(line['X_end'] - line['X_start'])
    else:
        return abs(line['Y_end'] - line['Y_start'])
def check_line_overlap(line1, line2):
    """
    Check if lines overlap and return overlap type and full extent of both lines
    Returns: (overlap_type, full_extent)
    overlap_type: None, 'partial', or 'full'
    full_extent: (min_coord, max_coord) covering both lines
    """
    if line1['Horizontal'] != line2['Horizontal']:
        return None, None
        
    start1, end1 = get_line_endpoints(line1)
    start2, end2 = get_line_endpoints(line2)
    
    if line1['Horizontal']:
        if abs(start1[1] - start2[1]) > 0.001:  # Different Y coordinates
            return None, None
            
        # Get X coordinates for overlap check
        x1_min, x1_max = start1[0], end1[0]
        x2_min, x2_max = start2[0], end2[0]
        
        # Find overlap region
        overlap_start = max(x1_min, x2_min)
        overlap_end = min(x1_max, x2_max)
        
        if overlap_start > overlap_end:
            return None, None
        
        # Determine overlap type and return full extent
        full_extent = (min(x1_min, x2_min), max(x1_max, x2_max))
        if (overlap_start == x1_min and overlap_end == x1_max):
            return 'full', full_extent
        elif (overlap_start == x2_min and overlap_end == x2_max):
            return 'full', full_extent
        else:
            return 'partial', full_extent
            
    else:  # Vertical lines
        if abs(start1[0] - start2[0]) > 0.001:  # Different X coordinates
            return None, None
            
        # Get Y coordinates for overlap check
        y1_min, y1_max = start1[1], end1[1]
        y2_min, y2_max = start2[1], end2[1]
        
        # Find overlap region
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        
        if overlap_start > overlap_end:
            return None, None
        
        # Determine overlap type and return full extent
        full_extent = (min(y1_min, y2_min), max(y1_max, y2_max))
        if (overlap_start == y1_min and overlap_end == y1_max):
            return 'full', full_extent
        elif (overlap_start == y2_min and overlap_end == y2_max):
            return 'full', full_extent
        else:
            return 'partial', full_extent
        
        
def get_line_endpoints(line):
    """Get the endpoints of a line in order (min to max)"""
    if line['Horizontal']:
        x_min = min(line['X_start'], line['X_end'])
        x_max = max(line['X_start'], line['X_end'])
        return (x_min, line['Y_start']), (x_max, line['Y_start'])
    else:
        y_min = min(line['Y_start'], line['Y_end'])
        y_max = max(line['Y_start'], line['Y_end'])
        return (line['X_start'], y_min), (line['X_start'], y_max)

def extend_line_to_cover(line, start_point, end_point):
    """
    Extend a line to cover the given start and end points
    Returns a new line dictionary with updated coordinates
    """
    new_line = line.copy()
    
    if line['Horizontal']:
        # Find the min and max X coordinates to cover both lines
        current_min_x = min(line['X_start'], line['X_end'])
        current_max_x = max(line['X_start'], line['X_end'])
        new_min_x = min(current_min_x, start_point[0], end_point[0])
        new_max_x = max(current_max_x, start_point[0], end_point[0])
        
        # Update line coordinates
        if line['X_start'] <= line['X_end']:
            new_line['X_start'] = new_min_x
            new_line['X_end'] = new_max_x
        else:
            new_line['X_start'] = new_max_x
            new_line['X_end'] = new_min_x
    else:
        # Find the min and max Y coordinates to cover both lines
        current_min_y = min(line['Y_start'], line['Y_end'])
        current_max_y = max(line['Y_start'], line['Y_end'])
        new_min_y = min(current_min_y, start_point[1], end_point[1])
        new_max_y = max(current_max_y, start_point[1], end_point[1])
        
        # Update line coordinates
        if line['Y_start'] <= line['Y_end']:
            new_line['Y_start'] = new_min_y
            new_line['Y_end'] = new_max_y
        else:
            new_line['Y_start'] = new_max_y
            new_line['Y_end'] = new_min_y
            
    return new_line
def detect_overlapping_lines(df):
    """
    Detect and handle overlapping lines by extending longer lines to cover shorter ones.
    Removes shorter lines and extends longer lines without creating new lines.
    """
    main_df = df.copy()
    overlapped_df = pd.DataFrame(columns=df.columns)
    
    # Filter out non-line entities
    lines_df = main_df[main_df['Type'] == 'LINE']
    
    processed_indices = set()
    overlapped_indices = set()
    modifications = {}  # Store required modifications to longer lines
    
    # Process each line
    for idx1, line1 in lines_df.iterrows():
        if idx1 in processed_indices:
            continue
        
        # Check against all other lines
        for idx2, line2 in lines_df.iterrows():
            if idx1 == idx2 or idx2 in processed_indices:
                continue
                
            overlap_type, full_extent = check_line_overlap(line1, line2)
            if overlap_type in ['full', 'partial']:
                length1 = get_line_length(line1)
                length2 = get_line_length(line2)
                
                # Identify longer and shorter lines
                if length1 < length2:
                    shorter_idx, shorter_line = idx1, line1
                    longer_idx, longer_line = idx2, line2
                else:
                    shorter_idx, shorter_line = idx2, line2
                    longer_idx, longer_line = idx1, line1
                
                # Mark shorter line for removal
                overlapped_indices.add(shorter_idx)
                processed_indices.add(shorter_idx)
                
                # Get endpoints of both lines
                shorter_start, shorter_end = get_line_endpoints(shorter_line)
                longer_start, longer_end = get_line_endpoints(longer_line)
                
                # Determine if we need to extend the longer line
                if longer_idx not in modifications:
                    modifications[longer_idx] = {
                        'start': longer_start,
                        'end': longer_end
                    }
                
                # Update the extent if needed
                if longer_line['Horizontal']:
                    # Update X coordinates while keeping Y constant
                    new_start_x = min(modifications[longer_idx]['start'][0], shorter_start[0])
                    new_end_x = max(modifications[longer_idx]['end'][0], shorter_end[0])
                    
                    # Maintain original direction of the line
                    if longer_line['X_start'] <= longer_line['X_end']:
                        modifications[longer_idx]['start'] = (new_start_x, longer_line['Y_start'])
                        modifications[longer_idx]['end'] = (new_end_x, longer_line['Y_start'])
                    else:
                        modifications[longer_idx]['start'] = (new_end_x, longer_line['Y_start'])
                        modifications[longer_idx]['end'] = (new_start_x, longer_line['Y_start'])
                else:
                    # Update Y coordinates while keeping X constant
                    new_start_y = min(modifications[longer_idx]['start'][1], shorter_start[1])
                    new_end_y = max(modifications[longer_idx]['end'][1], shorter_end[1])
                    
                    # Maintain original direction of the line
                    if longer_line['Y_start'] <= longer_line['Y_end']:
                        modifications[longer_idx]['start'] = (longer_line['X_start'], new_start_y)
                        modifications[longer_idx]['end'] = (longer_line['X_start'], new_end_y)
                    else:
                        modifications[longer_idx]['start'] = (longer_line['X_start'], new_end_y)
                        modifications[longer_idx]['end'] = (longer_line['X_start'], new_start_y)
                
                if idx1 == shorter_idx:
                    break
    
    # Create the overlapped lines DataFrame
    overlapped_df = main_df.loc[list(overlapped_indices)]
    
    # Remove overlapped lines from main DataFrame
    main_df = main_df.drop(index=list(overlapped_indices))
    
    # Apply modifications to extend lines
    for idx, mod_info in modifications.items():
        if idx in main_df.index:
            # Update the line coordinates directly
            main_df.at[idx, 'X_start'] = mod_info['start'][0]
            main_df.at[idx, 'Y_start'] = mod_info['start'][1]
            main_df.at[idx, 'X_end'] = mod_info['end'][0]
            main_df.at[idx, 'Y_end'] = mod_info['end'][1]
    
    return main_df, overlapped_df

def process_dxf_dataframe(df):
    """
    Process DXF DataFrame to remove overlapped lines and extend longer lines.
    """
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['Type', 'X_start', 'Y_start', 'Z_start', 
                                  'X_end', 'Y_end', 'Z_end', 'Horizontal'])
    
    # Handle overlapping lines
    main_df, overlapped_df = detect_overlapping_lines(df)
    
    return main_df, overlapped_df

def find_closest_points_between_lines(line1, line2):
    """
    Find the closest points between two non-parallel lines.
    Returns the closest points on each line and the minimum distance between them.
    """
    # Extract points and create direction vectors
    p1 = np.array([line1['X_start'], line1['Y_start']])
    p2 = np.array([line1['X_end'], line1['Y_end']])
    p3 = np.array([line2['X_start'], line2['Y_start']])
    p4 = np.array([line2['X_end'], line2['Y_end']])
    
    v1 = p2 - p1  # Direction vector of line1
    v2 = p4 - p3  # Direction vector of line2
    
    # Normalize direction vectors
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    
    # Find closest points
    p13 = p3 - p1
    
    # Calculate dot products
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, p13)
    e = np.dot(v2, p13)
    
    # Calculate parameters for closest points
    denom = a * c - b * b
    if abs(denom) < 1e-10:  # Lines are nearly parallel
        return None, None, float('inf')
        
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    # Check if closest points are within line segments
    t1 = max(0, min(1, t1))
    t2 = max(0, min(1, t2))
    
    # Calculate closest points
    closest_on_line1 = p1 + t1 * v1
    closest_on_line2 = p3 + t2 * v2
    
    # Calculate distance between closest points
    distance = np.linalg.norm(closest_on_line1 - closest_on_line2)
    
    return closest_on_line1, closest_on_line2, distance

def connect_nearby_lines(df, tolerance):
    """
    Find and connect non-parallel lines that are nearly intersecting.
    Adjusts endpoints of lines that are within the tolerance distance.
    """
    modified_df = df.copy()
    lines_df = modified_df[modified_df['Type'] == 'LINE']
    
    # Track which lines have been modified
    modified_indices = set()
    
    for idx1, line1 in lines_df.iterrows():
        if line1['Horizontal'] is None:  # Skip if not a valid line
            continue
            
        for idx2, line2 in lines_df.iterrows():
            if (idx1 >= idx2 or line2['Horizontal'] is None or 
                idx1 in modified_indices or idx2 in modified_indices):
                continue
                
            # Skip if lines are parallel (both horizontal or both vertical)
            if line1['Horizontal'] == line2['Horizontal']:
                continue
                
            # Find closest points between lines
            closest1, closest2, distance = find_closest_points_between_lines(line1, line2)
            
            if closest1 is not None and distance <= tolerance:
                # Calculate midpoint between closest points
                connection_point = (closest1 + closest2) / 2
                
                # Update endpoints of both lines to meet at the connection point
                # For line1
                dist1_start = np.linalg.norm(np.array([line1['X_start'], line1['Y_start']]) - connection_point)
                dist1_end = np.linalg.norm(np.array([line1['X_end'], line1['Y_end']]) - connection_point)
                
                if dist1_start < dist1_end:
                    modified_df.at[idx1, 'X_start'] = connection_point[0]
                    modified_df.at[idx1, 'Y_start'] = connection_point[1]
                else:
                    modified_df.at[idx1, 'X_end'] = connection_point[0]
                    modified_df.at[idx1, 'Y_end'] = connection_point[1]
                
                # For line2
                dist2_start = np.linalg.norm(np.array([line2['X_start'], line2['Y_start']]) - connection_point)
                dist2_end = np.linalg.norm(np.array([line2['X_end'], line2['Y_end']]) - connection_point)
                
                if dist2_start < dist2_end:
                    modified_df.at[idx2, 'X_start'] = connection_point[0]
                    modified_df.at[idx2, 'Y_start'] = connection_point[1]
                else:
                    modified_df.at[idx2, 'X_end'] = connection_point[0]
                    modified_df.at[idx2, 'Y_end'] = connection_point[1]
                
                modified_indices.add(idx1)
                modified_indices.add(idx2)
    
    return modified_df

def process_dxf_dataframe_with_connections(df):
    """
    Enhanced version of process_dxf_dataframe that also connects nearby non-parallel lines.
    """
    # First handle overlapping lines
    main_df, overlapped_df = process_dxf_dataframe(df)    
    No_of_Overlapped_lines = overlapped_df.shape[0]
    # Then connect nearby non-parallel lines
    connected_df = connect_nearby_lines(main_df,1)
    
    return connected_df, overlapped_df,No_of_Overlapped_lines

def check_intersection(line1, line2):
    """
    Check if two perpendicular lines intersect.
    Returns the intersection point if they intersect, None otherwise.
    """
    # For vertical line
    if line1['Vertical']:
        v_x = float(line1['X_start'])
        v_y_min = min(float(line1['Y_start']), float(line1['Y_end']))
        v_y_max = max(float(line1['Y_start']), float(line1['Y_end']))
        
        h_y = float(line2['Y_start'])
        h_x_min = min(float(line2['X_start']), float(line2['X_end']))
        h_x_max = max(float(line2['X_start']), float(line2['X_end']))
        
        if (h_x_min <= v_x <= h_x_max and 
            v_y_min <= h_y <= v_y_max):
            return (v_x, h_y)
    # For horizontal line
    else:
        h_y = float(line1['Y_start'])
        h_x_min = min(float(line1['X_start']), float(line1['X_end']))
        h_x_max = max(float(line1['X_start']), float(line1['X_end']))
        
        v_x = float(line2['X_start'])
        v_y_min = min(float(line2['Y_start']), float(line2['Y_end']))
        v_y_max = max(float(line2['Y_start']), float(line2['Y_end']))
        
        if (h_x_min <= v_x <= h_x_max and 
            v_y_min <= h_y <= v_y_max):
            return (v_x, h_y)
    return None

def split_lines_at_intersections(df):
    """
    Split both horizontal and vertical lines at their intersection points.
    Input: DataFrame with the specified DXF columns
    Returns: DataFrame with split lines
    """
    # Convert DataFrame to list of dictionaries for easier processing
    lines = df.to_dict('records')
    
    # Separate vertical and horizontal lines
    vertical_lines = [line for line in lines if line['Vertical']]
    horizontal_lines = [line for line in lines if line['Horizontal']]
    
    new_lines = []
    
    # Process horizontal lines
    for h_line in horizontal_lines:
        intersection_points = []
        
        # Find all intersection points with vertical lines
        for v_line in vertical_lines:
            intersection = check_intersection(h_line, v_line)
            if intersection:
                intersection_points.append(intersection[0])  # We need x-coordinate for horizontal lines
        
        if not intersection_points:
            new_lines.append(h_line)
            continue
            
        # Sort intersection points from left to right
        intersection_points.sort()
        
        # Create new line segments
        x_points = [float(h_line['X_start'])] + intersection_points + [float(h_line['X_end'])]
        x_points = sorted(set(x_points))  # Remove duplicates and sort
        
        for i in range(len(x_points) - 1):
            # Calculate length of new segment
            length = abs(x_points[i] - x_points[i + 1])
            
            # Create new line segment
            new_line = {
                'Type': h_line['Type'],
                'Layer': h_line['Layer'],
                'X_start': x_points[i],
                'Y_start': h_line['Y_start'],
                'Z_start': h_line['Z_start'],
                'X_end': x_points[i + 1],
                'Y_end': h_line['Y_end'],
                'Z_end': h_line['Z_end'],
                'Length': length,
                'Horizontal': True,
                'Vertical': False
            }
            new_lines.append(new_line)
    
    # Process vertical lines
    for v_line in vertical_lines:
        intersection_points = []
        
        # Find all intersection points with horizontal lines
        for h_line in horizontal_lines:
            intersection = check_intersection(v_line, h_line)
            if intersection:
                intersection_points.append(intersection[1])  # We need y-coordinate for vertical lines
        
        if not intersection_points:
            new_lines.append(v_line)
            continue
            
        # Sort intersection points from top to bottom
        intersection_points.sort(reverse=True)
        
        # Create new line segments
        y_points = [float(v_line['Y_start'])] + intersection_points + [float(v_line['Y_end'])]
        y_points = sorted(set(y_points), reverse=True)  # Remove duplicates and sort
        
        for i in range(len(y_points) - 1):
            # Calculate length of new segment
            length = abs(y_points[i] - y_points[i + 1])
            
            # Create new line segment
            new_line = {
                'Type': v_line['Type'],
                'Layer': v_line['Layer'],
                'X_start': v_line['X_start'],
                'Y_start': y_points[i],
                'Z_start': v_line['Z_start'],
                'X_end': v_line['X_end'],
                'Y_end': y_points[i + 1],
                'Z_end': v_line['Z_end'],
                'Length': length,
                'Horizontal': False,
                'Vertical': True
            }
            new_lines.append(new_line)
    
    # Convert back to DataFrame and sort by Type and Layer
    result_df = pd.DataFrame(new_lines)
    return result_df.sort_values(['Type', 'Layer']).reset_index(drop=True)

def check_intersection(line1, line2):
    """
    Check if two perpendicular lines intersect.
    Returns the intersection point if they intersect, None otherwise.
    """
    # For vertical line
    if line1['Vertical']:
        v_x = float(line1['X_start'])
        v_y_min = min(float(line1['Y_start']), float(line1['Y_end']))
        v_y_max = max(float(line1['Y_start']), float(line1['Y_end']))
        
        h_y = float(line2['Y_start'])
        h_x_min = min(float(line2['X_start']), float(line2['X_end']))
        h_x_max = max(float(line2['X_start']), float(line2['X_end']))
        
        if (h_x_min <= v_x <= h_x_max and 
            v_y_min <= h_y <= v_y_max):
            return (v_x, h_y)
    # For horizontal line
    else:
        h_y = float(line1['Y_start'])
        h_x_min = min(float(line1['X_start']), float(line1['X_end']))
        h_x_max = max(float(line1['X_start']), float(line1['X_end']))
        
        v_x = float(line2['X_start'])
        v_y_min = min(float(line2['Y_start']), float(line2['Y_end']))
        v_y_max = max(float(line2['Y_start']), float(line2['Y_end']))
        
        if (h_x_min <= v_x <= h_x_max and 
            v_y_min <= h_y <= v_y_max):
            return (v_x, h_y)
    return None

def split_lines_at_intersections(df):
    """
    Split both horizontal and vertical lines at their intersection points.
    Input: DataFrame with the specified DXF columns
    Returns: DataFrame with split lines
    """
    # Convert DataFrame to list of dictionaries for easier processing
    lines = df.to_dict('records')
    
    # Separate vertical and horizontal lines
    vertical_lines = [line for line in lines if line['Vertical']]
    horizontal_lines = [line for line in lines if line['Horizontal']]
    
    new_lines = []
    
    # Process horizontal lines
    for h_line in horizontal_lines:
        intersection_points = []
        
        # Find all intersection points with vertical lines
        for v_line in vertical_lines:
            intersection = check_intersection(h_line, v_line)
            if intersection:
                intersection_points.append(intersection[0])  # We need x-coordinate for horizontal lines
        
        if not intersection_points:
            new_lines.append(h_line)
            continue
            
        # Sort intersection points from left to right
        intersection_points.sort()
        
        # Create new line segments
        x_points = [float(h_line['X_start'])] + intersection_points + [float(h_line['X_end'])]
        x_points = sorted(set(x_points))  # Remove duplicates and sort
        
        for i in range(len(x_points) - 1):
            # Calculate length of new segment
            length = abs(x_points[i] - x_points[i + 1])
            
            # Create new line segment
            new_line = {
                'Type': h_line['Type'],
                'Layer': h_line['Layer'],
                'X_start': x_points[i],
                'Y_start': h_line['Y_start'],
                'Z_start': h_line['Z_start'],
                'X_end': x_points[i + 1],
                'Y_end': h_line['Y_end'],
                'Z_end': h_line['Z_end'],
                'Length': length,
                'Horizontal': True,
                'Vertical': False
            }
            new_lines.append(new_line)
    
    # Process vertical lines
    for v_line in vertical_lines:
        intersection_points = []
        
        # Find all intersection points with horizontal lines
        for h_line in horizontal_lines:
            intersection = check_intersection(v_line, h_line)
            if intersection:
                intersection_points.append(intersection[1])  # We need y-coordinate for vertical lines
        
        if not intersection_points:
            new_lines.append(v_line)
            continue
            
        # Sort intersection points from top to bottom
        intersection_points.sort(reverse=True)
        
        # Create new line segments
        y_points = [float(v_line['Y_start'])] + intersection_points + [float(v_line['Y_end'])]
        y_points = sorted(set(y_points), reverse=True)  # Remove duplicates and sort
        
        for i in range(len(y_points) - 1):
            # Calculate length of new segment
            length = abs(y_points[i] - y_points[i + 1])
            
            # Create new line segment
            new_line = {
                'Type': v_line['Type'],
                'Layer': v_line['Layer'],
                'X_start': v_line['X_start'],
                'Y_start': y_points[i],
                'Z_start': v_line['Z_start'],
                'X_end': v_line['X_end'],
                'Y_end': y_points[i + 1],
                'Z_end': v_line['Z_end'],
                'Length': length,
                'Horizontal': False,
                'Vertical': True
            }
            new_lines.append(new_line)
    
    # Convert back to DataFrame and sort by Type and Layer
    result_df = pd.DataFrame(new_lines)
    return result_df.sort_values(['Type', 'Layer']).reset_index(drop=True)

from ezdxf.math import Vec3

def create_dxf_from_combined_dataframe(df, output_filename):
    """
    Creates a DXF file from a combined DataFrame containing both regular entities and block entities.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing entity information
        output_filename (str): Output DXF filename
    Returns:
        str: Path to the created DXF file
    """
    doc = ezdxf.new()  # Use newer version for better block support
    msp = doc.modelspace()
    
    doc.header['$ACADVER'] = 'AC1032'  # AutoCAD 2018
    doc.header['$AUNITS'] = 0  # Angular units in degrees
    doc.header['$INSUNITS'] = 4
    # Create dictionary to store layers and blocks
    layers = {}
    blocks = {}
    
    # First pass: Create all necessary layers
    unique_layers = df['Layer'].unique()
    for layer_name in unique_layers:
        if pd.notna(layer_name) and layer_name != '0':
            try:
                layers[layer_name] = doc.layers.new(name=str(layer_name))
            except ezdxf.lldxf.const.DXFTableEntryError:
                # Layer already exists
                continue

    # Second pass: Create blocks
    block_names = df[df['Block_Name'].notna()]['Block_Name'].unique()
    for block_name in block_names:
        if block_name not in blocks:
            blocks[block_name] = doc.blocks.new(name=block_name)

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        layer_name = str(row['Layer']) if pd.notna(row['Layer']) else '0'
        
        try:
            # Check if this is a block entity
            is_block_entity = pd.notna(row.get('Block_Name'))
            
            if is_block_entity:
                # Handle block insertion
                insert_point = (
                    float(row['Insert_X']) if pd.notna(row.get('Insert_X')) else 0,
                    float(row['Insert_Y']) if pd.notna(row.get('Insert_Y')) else 0,
                    0
                )
                
                scale_x = float(row['Scale_X']) if pd.notna(row.get('Scale_X')) else 1
                scale_y = float(row['Scale_Y']) if pd.notna(row.get('Scale_Y')) else 1
                rotation = float(row['Rotation']) if pd.notna(row.get('Rotation')) else 0
                
                # Add entities to the block definition if it exists
                if row['Block_Name'] in blocks:
                    block = blocks[row['Block_Name']]
                    
                    if row['Type'] == 'LINE':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['X_end'], row['Y_end']])):
                            block.add_line(
                                (float(row['X_start']), float(row['Y_start'])),
                                (float(row['X_end']), float(row['Y_end'])),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] == 'CIRCLE':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['Length']])):
                            block.add_circle(
                                (float(row['X_start']), float(row['Y_start'])),
                                float(row['Length']),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] == 'ARC':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['Length'], row['X_end'], row['Y_end']])):
                            block.add_arc(
                                (float(row['X_start']), float(row['Y_start'])),
                                float(row['Length']),
                                float(row['X_end']),
                                float(row['Y_end']),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] in ['TEXT', 'MTEXT']:
                        if pd.notna(row.get('X_insert')) and pd.notna(row.get('Y_insert')):
                            text = str(row['Text']) if pd.notna(row.get('Text')) else ''
                            if row['Type'] == 'TEXT':
                                block.add_text(
                                    text,
                                    dxfattribs={
                                        'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                        'layer': layer_name,
                                        'rotation': float(row.get('Rotation', 0))
                                    }
                                )
                            else:  # MTEXT
                                block.add_mtext(
                                    text,
                                    dxfattribs={
                                        'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                        'layer': layer_name,
                                        'rotation': float(row.get('Rotation', 0))
                                    }
                                )
                
                # Add block reference to modelspace
                msp.add_blockref(
                    row['Block_Name'],
                    insert_point,
                    dxfattribs={
                        'xscale': scale_x,
                        'yscale': scale_y,
                        'rotation': rotation,
                        'layer': layer_name
                    }
                )
            
            else:
                # Handle regular entities (non-block)
                if row['Type'] == 'LINE':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['X_end'], row['Y_end']])):
                        msp.add_line(
                            (float(row['X_start']), float(row['Y_start'])),
                            (float(row['X_end']), float(row['Y_end'])),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] == 'CIRCLE':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['Length']])):
                        msp.add_circle(
                            (float(row['X_start']), float(row['Y_start'])),
                            float(row['Length']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] == 'ARC':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['Length'], row['X_end'], row['Y_end']])):
                        msp.add_arc(
                            (float(row['X_start']), float(row['Y_start'])),
                            float(row['Length']),
                            float(row['X_end']),
                            float(row['Y_end']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] in ['TEXT', 'MTEXT']:
                    if pd.notna(row.get('X_insert')) and pd.notna(row.get('Y_insert')):
                        text = str(row['Text']) if pd.notna(row.get('Text')) else ''
                        if row['Type'] == 'TEXT':
                            msp.add_text(
                                text,
                                dxfattribs={
                                    'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                    'layer': layer_name,
                                    'rotation': float(row.get('Rotation', 0))
                                }
                            )
                        else:  # MTEXT
                            msp.add_mtext(
                                text,
                                dxfattribs={
                                    'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                    'layer': layer_name,
                                    'rotation': float(row.get('Rotation', 0))
                                }
                            )
        
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            print(f"Row data: {row}")
            continue

    # Save the document
    try:
        doc.saveas(output_filename)
        print(f"DXF file successfully created: {output_filename}")
    except Exception as e:
        print(f"Error saving DXF file: {str(e)}")
        return None

    return output_filename

# Example usage:
def combine_and_create_dxf(regular_df, blocks_df, output_filename):
    """
    Combines regular entities DataFrame with blocks DataFrame and creates a DXF file.
    
    Parameters:
        regular_df (pd.DataFrame): DataFrame containing regular entities
        blocks_df (pd.DataFrame): DataFrame containing block entities
        output_filename (str): Output DXF filename
    Returns:
        str: Path to the created DXF file
    """
    # Ensure all required columns exist in both DataFrames
    required_columns = ['Type', 'Layer', 'X_start', 'Y_start', 'X_end', 'Y_end', 
                       'Length', 'Text', 'X_insert', 'Y_insert', 'Rotation']
    
    # Add missing columns to regular_df
    for col in required_columns:
        if col not in regular_df.columns:
            regular_df[col] = None
    
    # Add Block_Name and scaling columns to regular_df
    if 'Block_Name' not in regular_df.columns:
        regular_df['Block_Name'] = None
    if 'Scale_X' not in regular_df.columns:
        regular_df['Scale_X'] = 1.0
    if 'Scale_Y' not in regular_df.columns:
        regular_df['Scale_Y'] = 1.0
    
    # Combine DataFrames
    combined_df = pd.concat([regular_df, blocks_df], ignore_index=True)
    
    # Create DXF file
    return create_dxf_from_combined_dataframe(combined_df, output_filename)

from ezdxf.entities import Insert
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc
from matplotlib.lines import Line2D
import math

def transform_point(x, y, scale_x, scale_y, rotation, insert_point):
    """
    Transform a point based on scaling, rotation, and insertion point.
    
    Parameters:
        x, y: Original coordinates
        scale_x, scale_y: Scaling factors
        rotation: Rotation angle in degrees
        insert_point: Insertion point coordinates
    Returns:
        Transformed x, y coordinates
    """
    # Apply scaling
    x_scaled = x * scale_x
    y_scaled = y * scale_y
    
    # Convert rotation to radians
    rotation_rad = math.radians(rotation)
    
    # Apply rotation
    x_rot = x_scaled * math.cos(rotation_rad) - y_scaled * math.sin(rotation_rad)
    y_rot = x_scaled * math.sin(rotation_rad) + y_scaled * math.cos(rotation_rad)
    
    # Apply translation (insertion point)
    x_final = x_rot + insert_point[0]
    y_final = y_rot + insert_point[1]
    
    return x_final, y_final

def extract_blocks_to_df(dxf_path):
    """
    Extracts blocks from a DXF file and converts them to a DataFrame with correct coordinates.
    
    Parameters:
        dxf_path (str): Path to the DXF file
    Returns:
        pd.DataFrame: DataFrame containing block information
    """
    doc = ezdxf.readfile(dxf_path)
    modelspace = doc.modelspace()
    blocks = doc.blocks
    
    block_data = []
    
    # Process block insertions first
    for insert in modelspace.query('INSERT'):
        block_name = insert.dxf.name
        if block_name in blocks:
            block = blocks[block_name]
            
            # Get insertion parameters
            insert_point = insert.dxf.insert
            scale_x = insert.dxf.xscale
            scale_y = insert.dxf.yscale
            rotation = insert.dxf.rotation
            
            # Process each entity in the block
            for entity in block:
                base_data = {
                    'Block_Name': block_name,
                    'Type': entity.dxftype(),
                    'Layer': entity.dxf.layer,
                    'Insert_X': insert_point[0],
                    'Insert_Y': insert_point[1],
                    'Scale_X': scale_x,
                    'Scale_Y': scale_y,
                    'Rotation': rotation
                }
                
                if entity.dxftype() == 'LINE':
                    # Transform start point
                    x_start, y_start = transform_point(
                        entity.dxf.start[0], entity.dxf.start[1],
                        scale_x, scale_y, rotation, insert_point
                    )
                    
                    # Transform end point
                    x_end, y_end = transform_point(
                        entity.dxf.end[0], entity.dxf.end[1],
                        scale_x, scale_y, rotation, insert_point
                    )
                    
                    data = {
                        **base_data,
                        'X_start': x_start,
                        'Y_start': y_start,
                        'X_end': x_end,
                        'Y_end': y_end,
                        'Length': None
                    }
                    block_data.append(data)
                    
                elif entity.dxftype() == 'CIRCLE':
                    # Transform center point
                    x_center, y_center = transform_point(
                        entity.dxf.center[0], entity.dxf.center[1],
                        scale_x, scale_y, rotation, insert_point
                    )
                    
                    data = {
                        **base_data,
                        'X_start': x_center,
                        'Y_start': y_center,
                        'X_end': None,
                        'Y_end': None,
                        'Length': entity.dxf.radius * scale_x  # Scale radius
                    }
                    block_data.append(data)
                    
                elif entity.dxftype() == 'ARC':
                    # Transform center point
                    x_center, y_center = transform_point(
                        entity.dxf.center[0], entity.dxf.center[1],
                        scale_x, scale_y, rotation, insert_point
                    )
                    
                    # Adjust angles based on rotation
                    start_angle = (entity.dxf.start_angle + rotation) % 360
                    end_angle = (entity.dxf.end_angle + rotation) % 360
                    
                    data = {
                        **base_data,
                        'X_start': x_center,
                        'Y_start': y_center,
                        'X_end': start_angle,
                        'Y_end': end_angle,
                        'Length': entity.dxf.radius * scale_x  # Scale radius
                    }
                    block_data.append(data)
                    
                elif entity.dxftype() in ['TEXT', 'MTEXT']:
                    # Transform text insertion point
                    x_text, y_text = transform_point(
                        entity.dxf.insert[0], entity.dxf.insert[1],
                        scale_x, scale_y, rotation, insert_point
                    )
                    
                    text_rotation = entity.dxf.rotation + rotation if hasattr(entity.dxf, 'rotation') else rotation
                    
                    data = {
                        **base_data,
                        'X_start': None,
                        'Y_start': None,
                        'X_end': None,
                        'Y_end': None,
                        'Length': None,
                        'Text': entity.dxf.text if hasattr(entity.dxf, 'text') else '',
                        'X_insert': x_text,
                        'Y_insert': y_text,
                        'Rotation': text_rotation
                    }
                    block_data.append(data)
    
    return pd.DataFrame(block_data)


# In[78]:


import ezdxf
from ezdxf.math import Vec3
import pandas as pd

def create_dxf_from_combined_dataframe1(df, output_filename):
    """
    Creates an AutoCAD-compatible DXF file from a combined DataFrame containing both regular entities and block entities.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing entity information
        output_filename (str): Output DXF filename
    Returns:
        str: Path to the created DXF file
    """
    # Create a new DXF document with AutoCAD 2018 format
    doc = ezdxf.new(setup=True)  # Added setup=True for proper initialization
    msp = doc.modelspace()
    
    # Set up some basic AutoCAD requirements
    doc.header['$ACADVER'] = 'AC1032'  # AutoCAD 2018
    doc.header['$AUNITS'] = 0  # Angular units in degrees
    doc.header['$INSUNITS'] = 4  # Millimeters
    
    # Create dictionary to store layers and blocks
    layers = {}
    blocks = {}
    
    # First pass: Create all necessary layers
    unique_layers = df['Layer'].unique()
    for layer_name in unique_layers:
        if pd.notna(layer_name) and layer_name != '0':
            try:
                layers[layer_name] = doc.layers.new(name=str(layer_name))
            except ezdxf.lldxf.const.DXFTableEntryError:
                # Layer already exists
                continue

    # Second pass: Create blocks with proper attributes
    block_names = df[df['Block_Name'].notna()]['Block_Name'].unique()
    for block_name in block_names:
        if block_name not in blocks:
            blocks[block_name] = doc.blocks.new(name=block_name)

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        layer_name = str(row['Layer']) if pd.notna(row['Layer']) else '0'
        
        try:
            # Check if this is a block entity
            is_block_entity = pd.notna(row.get('Block_Name'))
            
            if is_block_entity:
                # Handle block insertion
                insert_point = (
                    float(row['Insert_X']) if pd.notna(row.get('Insert_X')) else 0,
                    float(row['Insert_Y']) if pd.notna(row.get('Insert_Y')) else 0,
                    0
                )
                
                scale_x = float(row['Scale_X']) if pd.notna(row.get('Scale_X')) else 1
                scale_y = float(row['Scale_Y']) if pd.notna(row.get('Scale_Y')) else 1
                rotation = float(row['Rotation']) if pd.notna(row.get('Rotation')) else 0
                
                # Add entities to the block definition if it exists
                if row['Block_Name'] in blocks:
                    block = blocks[row['Block_Name']]
                    
                    if row['Type'] == 'LINE':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['X_end'], row['Y_end']])):
                            block.add_line(
                                (float(row['X_start']), float(row['Y_start'])),
                                (float(row['X_end']), float(row['Y_end'])),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] == 'CIRCLE':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['Length']])):
                            block.add_circle(
                                (float(row['X_start']), float(row['Y_start'])),
                                float(row['Length']),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] == 'ARC':
                        if all(pd.notna([row['X_start'], row['Y_start'], row['Length'], row['X_end'], row['Y_end']])):
                            block.add_arc(
                                (float(row['X_start']), float(row['Y_start'])),
                                float(row['Length']),
                                float(row['X_end']),
                                float(row['Y_end']),
                                dxfattribs={'layer': layer_name}
                            )
                    
                    elif row['Type'] in ['TEXT', 'MTEXT']:
                        if pd.notna(row.get('X_insert')) and pd.notna(row.get('Y_insert')):
                            text = str(row['Text']) if pd.notna(row.get('Text')) else ''
                            if row['Type'] == 'TEXT':
                                block.add_text(
                                    text,
                                    dxfattribs={
                                        'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                        'layer': layer_name,
                                        'rotation': float(row.get('Rotation', 0))
                                    }
                                )
                            else:  # MTEXT
                                block.add_mtext(
                                    text,
                                    dxfattribs={
                                        'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                        'layer': layer_name,
                                        'rotation': float(row.get('Rotation', 0))
                                    }
                                )
                
                # Add block reference to modelspace
                msp.add_blockref(
                    row['Block_Name'],
                    insert_point,
                    dxfattribs={
                        'xscale': scale_x,
                        'yscale': scale_y,
                        'rotation': rotation,
                        'layer': layer_name
                    }
                )
            
            
            else:
                # Handle regular entities (non-block)
                if row['Type'] == 'LINE':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['X_end'], row['Y_end']])):
                        msp.add_line(
                            (float(row['X_start']), float(row['Y_start'])),
                            (float(row['X_end']), float(row['Y_end'])),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] == 'CIRCLE':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['Length']])):
                        msp.add_circle(
                            (float(row['X_start']), float(row['Y_start'])),
                            float(row['Length']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] == 'ARC':
                    if all(pd.notna([row['X_start'], row['Y_start'], row['Length'], row['X_end'], row['Y_end']])):
                        msp.add_arc(
                            (float(row['X_start']), float(row['Y_start'])),
                            float(row['Length']),
                            float(row['X_end']),
                            float(row['Y_end']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif row['Type'] in ['TEXT', 'MTEXT']:
                    if pd.notna(row.get('X_insert')) and pd.notna(row.get('Y_insert')):
                        text = str(row['Text']) if pd.notna(row.get('Text')) else ''
                        if row['Type'] == 'TEXT':
                            msp.add_text(
                                text,
                                dxfattribs={
                                    'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                    'layer': layer_name,
                                    'rotation': float(row.get('Rotation', 0))
                                }
                            )
                        else:  # MTEXT
                            msp.add_mtext(
                                text,
                                dxfattribs={
                                    'insert': (float(row['X_insert']), float(row['Y_insert'])),
                                    'layer': layer_name,
                                    'rotation': float(row.get('Rotation', 0))
                                }
                            )
        
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
            print(f"Row data: {row}")
            continue

    # Save the document with audit and cleanup
    try:
        doc.set_modelspace_vport(height=297, center=(148.5, 148.5))  # Set default viewport
        doc.audit()  # Audit the document for errors
        doc.saveas(output_filename)
        print(f"DXF file successfully created: {output_filename}")
    except Exception as e:
        print(f"Error saving DXF file: {str(e)}")
        return None

    return output_filename

def combine_and_create_dxf1(regular_df, blocks_df, output_filename):
    """
    Combines regular entities DataFrame with blocks DataFrame and creates a DXF file.
    
    Parameters:
        regular_df (pd.DataFrame): DataFrame containing regular entities
        blocks_df (pd.DataFrame): DataFrame containing block entities
        output_filename (str): Output DXF filename
    Returns:
        str: Path to the created DXF file
    """
    # Ensure all required columns exist in both DataFrames
    required_columns = [
        'Type', 'Layer', 'X_start', 'Y_start', 'X_end', 'Y_end',
        'Length', 'Text', 'X_insert', 'Y_insert', 'Rotation',
        'Height', 'Start_angle', 'End_angle'  # Added additional required columns
    ]
    
    # Add missing columns to regular_df
    for col in required_columns:
        if col not in regular_df.columns:
            regular_df[col] = None
    
    # Add Block_Name and scaling columns to regular_df
    if 'Block_Name' not in regular_df.columns:
        regular_df['Block_Name'] = None
    if 'Scale_X' not in regular_df.columns:
        regular_df['Scale_X'] = 1.0
    if 'Scale_Y' not in regular_df.columns:
        regular_df['Scale_Y'] = 1.0
    
    # Combine DataFrames
    combined_df = pd.concat([regular_df, blocks_df], ignore_index=True)
    
    # Create DXF file
    return create_dxf_from_combined_dataframe1(combined_df, output_filename)


# In[93]:


import ezdxf
from ezdxf.math import Vec3
import pandas as pd

def create_dxf_from_combined_dataframe2(df, output_filename):
    """
    Creates an AutoCAD-compatible DXF file from a combined DataFrame containing both regular entities and block entities.
    Fixed version that properly handles repeated block entities.
    """
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # Set up AutoCAD requirements
    doc.header['$ACADVER'] = 'AC1032'
    doc.header['$AUNITS'] = 0
    doc.header['$INSUNITS'] = 4
    
    layers = {}
    blocks = {}
    
    # Create layers
    unique_layers = df['Layer'].unique()
    for layer_name in unique_layers:
        if pd.notna(layer_name) and layer_name != '0':
            try:
                layers[layer_name] = doc.layers.new(name=str(layer_name))
            except ezdxf.lldxf.const.DXFTableEntryError:
                continue
    
    # Process blocks - Group by block name and insertion point
    block_groups = df[df['Block_Name'].notna()].groupby(['Block_Name', 'Insert_X', 'Insert_Y'])
    
    for (block_name, insert_x, insert_y), block_entities in block_groups:
        # Create block if it doesn't exist
        if block_name not in doc.blocks:
            block = doc.blocks.new(name=block_name)
            
            # Calculate offset for making coordinates relative to insertion point
            offset_x = float(insert_x)
            offset_y = float(insert_y)
            
            # Add all entities to block definition
            for _, entity in block_entities.iterrows():
                layer_name = str(entity['Layer']) if pd.notna(entity['Layer']) else '0'
                
                if entity['Type'] == 'LINE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['X_end'], entity['Y_end']])):
                        block.add_line(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            (float(entity['X_end']) - offset_x, float(entity['Y_end']) - offset_y),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] == 'CIRCLE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        block.add_circle(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            float(entity['Length']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] == 'ARC':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        start_angle = float(entity['Start_angle']) if pd.notna(entity.get('Start_angle')) else 0
                        end_angle = float(entity['End_angle']) if pd.notna(entity.get('End_angle')) else 360
                        
                        block.add_arc(
                            center=(float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            radius=float(entity['Length']),
                            start_angle=start_angle,
                            end_angle=end_angle,
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] in ['TEXT', 'MTEXT']:
                    if pd.notna(entity.get('X_insert')) and pd.notna(entity.get('Y_insert')):
                        text = str(entity['Text']) if pd.notna(entity.get('Text')) else ''
                        x_text = float(entity['X_insert']) - offset_x
                        y_text = float(entity['Y_insert']) - offset_y
                        
                        if entity['Type'] == 'TEXT':
                            block.add_text(
                                text,
                                dxfattribs={
                                    'insert': (x_text, y_text),
                                    'layer': layer_name,
                                    'height': float(entity.get('Height', 2.5)),
                                    'rotation': float(entity.get('Rotation', 0))
                                }
                            )
                        else:  # MTEXT
                            block.add_mtext(
                                text,
                                dxfattribs={
                                    'insert': (x_text, y_text),
                                    'layer': layer_name,
                                    'char_height': float(entity.get('Height', 2.5)),
                                    'rotation': float(entity.get('Rotation', 0))
                                }
                            )
        
        # Insert block reference
        scale_x = float(block_entities['Scale_X'].iloc[0]) if pd.notna(block_entities['Scale_X'].iloc[0]) else 1
        scale_y = float(block_entities['Scale_Y'].iloc[0]) if pd.notna(block_entities['Scale_Y'].iloc[0]) else 1
        rotation = float(block_entities['Rotation'].iloc[0]) if pd.notna(block_entities['Rotation'].iloc[0]) else 0
        layer_name = str(block_entities['Layer'].iloc[0]) if pd.notna(block_entities['Layer'].iloc[0]) else '0'
        
        msp.add_blockref(
            block_name,
            (float(insert_x), float(insert_y)),
            dxfattribs={
                'xscale': scale_x,
                'yscale': scale_y,
                'rotation': rotation,
                'layer': layer_name
            }
        )

    # Handle regular entities (non-block)
    regular_entities = df[df['Block_Name'].isna()]
    for _, entity in regular_entities.iterrows():
        layer_name = str(entity['Layer']) if pd.notna(entity['Layer']) else '0'
        
        if entity['Type'] == 'LINE':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['X_end'], entity['Y_end']])):
                msp.add_line(
                    (float(entity['X_start']), float(entity['Y_start'])),
                    (float(entity['X_end']), float(entity['Y_end'])),
                    dxfattribs={'layer': layer_name}
                )
        
        elif entity['Type'] == 'CIRCLE':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                msp.add_circle(
                    (float(entity['X_start']), float(entity['Y_start'])),
                    float(entity['Length']),
                    dxfattribs={'layer': layer_name}
                )
        
        elif entity['Type'] == 'ARC':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                start_angle = float(entity.get('Start_angle', 0))
                end_angle = float(entity.get('End_angle', 360))
                
                msp.add_arc(
                    center=(float(entity['X_start']), float(entity['Y_start'])),
                    radius=float(entity['Length']),
                    start_angle=start_angle,
                    end_angle=end_angle,
                    dxfattribs={'layer': layer_name}
                )

    # Save the document
    try:
        doc.set_modelspace_vport(height=297, center=(148.5, 148.5))
        doc.audit()
        doc.saveas(output_filename)
        print(f"DXF file successfully created: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error saving DXF file: {str(e)}")
        return None

def combine_and_create_dxf2(regular_df, blocks_df, output_filename):
    """
    Combines regular entities DataFrame with blocks DataFrame and creates a DXF file.
    """
    # Ensure all required columns exist in both DataFrames
    required_columns = [
        'Type', 'Layer', 'X_start', 'Y_start', 'X_end', 'Y_end',
        'Length', 'Start_angle', 'End_angle'
    ]
    
    # Add missing columns to regular_df
    for col in required_columns:
        if col not in regular_df.columns:
            regular_df[col] = None
    
    # Add block-related columns to regular_df
    if 'Block_Name' not in regular_df.columns:
        regular_df['Block_Name'] = None
    if 'Scale_X' not in regular_df.columns:
        regular_df['Scale_X'] = 1.0
    if 'Scale_Y' not in regular_df.columns:
        regular_df['Scale_Y'] = 1.0
    if 'Rotation' not in regular_df.columns:
        regular_df['Rotation'] = 0.0
    
    # Combine DataFrames
    combined_df = pd.concat([regular_df, blocks_df], ignore_index=True)
    
    return create_dxf_from_combined_dataframe(combined_df, output_filename)


# In[88]:


import ezdxf
import pandas as pd

def create_dxf_from_dataframe(df, output_filename):
    """
    Simplified approach focusing on correct block placement
    """
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # Basic setup
    doc.header['$ACADVER'] = 'AC1032'
    doc.header['$INSUNITS'] = 4
    
    # First, identify unique blocks by their insert points
    block_groups = df[df['Block_Name'].notna()].groupby(['Block_Name', 'Insert_X', 'Insert_Y'])
    
    # Process each unique block
    for (block_name, insert_x, insert_y), block_entities in block_groups:
        # Create block if it doesn't exist
        if block_name not in doc.blocks:
            block = doc.blocks.new(name=block_name)
            
            # Calculate offset for making coordinates relative to insertion point
            offset_x = float(insert_x)
            offset_y = float(insert_y)
            
            # Add all entities to block definition
            for _, entity in block_entities.iterrows():
                if entity['Type'] == 'LINE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['X_end'], entity['Y_end']])):
                        block.add_line(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            (float(entity['X_end']) - offset_x, float(entity['Y_end']) - offset_y)
                        )
                
                elif entity['Type'] == 'CIRCLE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        block.add_circle(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            float(entity['Length'])
                        )
                
                elif entity['Type'] == 'ARC':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        start_angle = float(entity['Start_angle']) if pd.notna(entity.get('Start_angle')) else 0
                        end_angle = float(entity['End_angle']) if pd.notna(entity.get('End_angle')) else 360
                        
                        block.add_arc(
                            center=(float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            radius=float(entity['Length']),
                            start_angle=start_angle,
                            end_angle=end_angle
                        )
        
        # Insert block reference
        scale_x = float(block_entities['Scale_X'].iloc[0]) if pd.notna(block_entities['Scale_X'].iloc[0]) else 1
        scale_y = float(block_entities['Scale_Y'].iloc[0]) if pd.notna(block_entities['Scale_Y'].iloc[0]) else 1
        rotation = float(block_entities['Rotation'].iloc[0]) if pd.notna(block_entities['Rotation'].iloc[0]) else 0
        
        msp.add_blockref(
            block_name,
            (float(insert_x), float(insert_y)),
            dxfattribs={
                'xscale': scale_x,
                'yscale': scale_y,
                'rotation': rotation
            }
        )
    
    # Save the document
    try:
        doc.saveas(output_filename)
        print(f"DXF file created: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error saving DXF file: {str(e)}")
        return None

def process_blocks_to_dxf(blocks_df, output_filename):
    """
    Wrapper function to process blocks DataFrame to DXF
    """
    # Ensure numeric columns are float
    numeric_columns = ['Insert_X', 'Insert_Y', 'X_start', 'Y_start', 
                      'X_end', 'Y_end', 'Length', 'Scale_X', 'Scale_Y', 
                      'Rotation', 'Start_angle', 'End_angle']
    
    for col in numeric_columns:
        if col in blocks_df.columns:
            blocks_df[col] = pd.to_numeric(blocks_df[col], errors='coerce')
    
    return create_dxf_from_dataframe(blocks_df, output_filename)




import ezdxf
from ezdxf.math import Vec3
import pandas as pd

def create_dxf_from_combined_dataframe(df, output_filename):
    """
    Creates an AutoCAD-compatible DXF file from a combined DataFrame containing both regular entities and block entities.
    Integrates the successful block handling approach.
    """
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()
    
    # Set up AutoCAD requirements
    doc.header['$ACADVER'] = 'AC1032'
    doc.header['$AUNITS'] = 0
    doc.header['$INSUNITS'] = 4
    
    # Create layers
    layers = {}
    unique_layers = df['Layer'].unique()
    for layer_name in unique_layers:
        if pd.notna(layer_name) and layer_name != '0':
            try:
                layers[layer_name] = doc.layers.new(name=str(layer_name))
            except ezdxf.lldxf.const.DXFTableEntryError:
                continue
    
    # Process blocks - Group by block name and insertion point
    block_groups = df[df['Block_Name'].notna()].groupby(['Block_Name', 'Insert_X', 'Insert_Y'])
    
    for (block_name, insert_x, insert_y), block_entities in block_groups:
        # Create block if it doesn't exist
        if block_name not in doc.blocks:
            block = doc.blocks.new(name=block_name)
            
            # Calculate offset for making coordinates relative to insertion point
            offset_x = float(insert_x)
            offset_y = float(insert_y)
            
            # Add all entities to block definition
            for _, entity in block_entities.iterrows():
                layer_name = str(entity['Layer']) if pd.notna(entity['Layer']) else '0'
                
                if entity['Type'] == 'LINE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['X_end'], entity['Y_end']])):
                        block.add_line(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            (float(entity['X_end']) - offset_x, float(entity['Y_end']) - offset_y),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] == 'CIRCLE':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        block.add_circle(
                            (float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            float(entity['Length']),
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] == 'ARC':
                    if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                        start_angle = float(entity['Start_angle']) if pd.notna(entity.get('Start_angle')) else 0
                        end_angle = float(entity['End_angle']) if pd.notna(entity.get('End_angle')) else 360
                        
                        block.add_arc(
                            center=(float(entity['X_start']) - offset_x, float(entity['Y_start']) - offset_y),
                            radius=float(entity['Length']),
                            start_angle=start_angle,
                            end_angle=end_angle,
                            dxfattribs={'layer': layer_name}
                        )
                
                elif entity['Type'] in ['TEXT', 'MTEXT']:
                    if pd.notna(entity.get('X_insert')) and pd.notna(entity.get('Y_insert')):
                        text = str(entity['Text']) if pd.notna(entity.get('Text')) else ''
                        x_text = float(entity['X_insert']) - offset_x
                        y_text = float(entity['Y_insert']) - offset_y
                        
                        if entity['Type'] == 'TEXT':
                            block.add_text(
                                text,
                                dxfattribs={
                                    'insert': (x_text, y_text),
                                    'layer': layer_name,
                                    'height': float(entity.get('Height', 2.5)),
                                    'rotation': float(entity.get('Rotation', 0))
                                }
                            )
                        else:  # MTEXT
                            block.add_mtext(
                                text,
                                dxfattribs={
                                    'insert': (x_text, y_text),
                                    'layer': layer_name,
                                    'char_height': float(entity.get('Height', 2.5)),
                                    'rotation': float(entity.get('Rotation', 0))
                                }
                            )
        
        # Insert block reference
        scale_x = float(block_entities['Scale_X'].iloc[0]) if pd.notna(block_entities['Scale_X'].iloc[0]) else 1
        scale_y = float(block_entities['Scale_Y'].iloc[0]) if pd.notna(block_entities['Scale_Y'].iloc[0]) else 1
        rotation = float(block_entities['Rotation'].iloc[0]) if pd.notna(block_entities['Rotation'].iloc[0]) else 0
        layer_name = str(block_entities['Layer'].iloc[0]) if pd.notna(block_entities['Layer'].iloc[0]) else '0'
        
        msp.add_blockref(
            block_name,
            (float(insert_x), float(insert_y)),
            dxfattribs={
                'xscale': scale_x,
                'yscale': scale_y,
                'rotation': rotation,
                'layer': layer_name
            }
        )

    # Handle regular entities (non-block)
    regular_entities = df[df['Block_Name'].isna()]
    for _, entity in regular_entities.iterrows():
        layer_name = str(entity['Layer']) if pd.notna(entity['Layer']) else '0'
        
        if entity['Type'] == 'LINE':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['X_end'], entity['Y_end']])):
                msp.add_line(
                    (float(entity['X_start']), float(entity['Y_start'])),
                    (float(entity['X_end']), float(entity['Y_end'])),
                    dxfattribs={'layer': layer_name}
                )
        
        elif entity['Type'] == 'CIRCLE':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                msp.add_circle(
                    (float(entity['X_start']), float(entity['Y_start'])),
                    float(entity['Length']),
                    dxfattribs={'layer': layer_name}
                )
        
        elif entity['Type'] == 'ARC':
            if all(pd.notna([entity['X_start'], entity['Y_start'], entity['Length']])):
                start_angle = float(entity['Start_angle']) if pd.notna(entity.get('Start_angle')) else 0
                end_angle = float(entity['End_angle']) if pd.notna(entity.get('End_angle')) else 360
                
                msp.add_arc(
                    center=(float(entity['X_start']), float(entity['Y_start'])),
                    radius=float(entity['Length']),
                    start_angle=start_angle,
                    end_angle=end_angle,
                    dxfattribs={'layer': layer_name}
                )
        
        elif entity['Type'] in ['TEXT', 'MTEXT']:
            if pd.notna(entity.get('X_insert')) and pd.notna(entity.get('Y_insert')):
                text = str(entity['Text']) if pd.notna(entity.get('Text')) else ''
                if entity['Type'] == 'TEXT':
                    msp.add_text(
                        text,
                        dxfattribs={
                            'insert': (float(entity['X_insert']), float(entity['Y_insert'])),
                            'layer': layer_name,
                            'height': float(entity.get('Height', 2.5)),
                            'rotation': float(entity.get('Rotation', 0))
                        }
                    )
                else:  # MTEXT
                    msp.add_mtext(
                        text,
                        dxfattribs={
                            'insert': (float(entity['X_insert']), float(entity['Y_insert'])),
                            'layer': layer_name,
                            'char_height': float(entity.get('Height', 2.5)),
                            'rotation': float(entity.get('Rotation', 0))
                        }
                    )

    # Save the document
    try:
        doc.set_modelspace_vport(height=297, center=(148.5, 148.5))
        doc.audit()
        doc.saveas(output_filename)
        print(f"DXF file successfully created: {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error saving DXF file: {str(e)}")
        return None

def combine_and_create_dxf(regular_df, blocks_df, output_filename):
    """
    Combines regular entities DataFrame with blocks DataFrame and creates a DXF file.
    """
    # Ensure all required columns exist
    required_columns = [
        'Type', 'Layer', 'X_start', 'Y_start', 'X_end', 'Y_end',
        'Length', 'Text', 'X_insert', 'Y_insert', 'Rotation',
        'Height', 'Start_angle', 'End_angle'
    ]
    
    # Add missing columns to regular_df
    for col in required_columns:
        if col not in regular_df.columns:
            regular_df[col] = None
    
    # Add block-related columns to regular_df
    if 'Block_Name' not in regular_df.columns:
        regular_df['Block_Name'] = None
    if 'Scale_X' not in regular_df.columns:
        regular_df['Scale_X'] = 1.0
    if 'Scale_Y' not in regular_df.columns:
        regular_df['Scale_Y'] = 1.0
    
    # Ensure numeric columns are float
    numeric_columns = ['Insert_X', 'Insert_Y', 'X_start', 'Y_start', 
                      'X_end', 'Y_end', 'Length', 'Scale_X', 'Scale_Y', 
                      'Rotation', 'Start_angle', 'End_angle']
    
    for df in [regular_df, blocks_df]:
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Combine DataFrames
    combined_df = pd.concat([regular_df, blocks_df], ignore_index=True)
    
    return create_dxf_from_combined_dataframe(combined_df, output_filename)


# In[117]:


import pandas as pd
import ezdxf
from ezdxf import units

def create_dxf_from_dataframes(blocks_df, lines_df, output_path='output.dxf'):
    """
    Convert dataframes containing block and line information into a DXF file.
    
    Parameters:
    blocks_df (pd.DataFrame): DataFrame containing block information
    lines_df (pd.DataFrame): DataFrame containing lines information
    output_path (str): Path where the DXF file will be saved
    """
    # Create a new DXF document - using R12 format for better compatibility
    doc = ezdxf.new('R12')
    
    # Get the modelspace
    msp = doc.modelspace()
    
    # Create layers first
    all_layers = pd.concat([blocks_df['Layer'], lines_df['Layer']]).unique()
    for layer_name in all_layers:
        if layer_name and str(layer_name).lower() != 'nan':
            doc.layers.new(name=str(layer_name))
    
    # Process blocks dataframe
    for _, row in blocks_df.iterrows():
        entity_type = str(row['Type']).upper()
        layer = str(row['Layer'])
        
        if pd.isna(layer) or layer.lower() == 'nan':
            layer = '0'
            
        try:
            if entity_type == 'LINE':
                msp.add_line(
                    start=(float(row['X_start']), float(row['Y_start'])),
                    end=(float(row['X_end']), float(row['Y_end'])),
                    dxfattribs={'layer': layer}
                )
                
            elif entity_type == 'CIRCLE':
                msp.add_circle(
                    center=(float(row['Insert_X']), float(row['Insert_Y'])),
                    radius=float(row['Length']),
                    dxfattribs={'layer': layer}
                )
                
            elif entity_type == 'ARC':
                msp.add_arc(
                    center=(float(row['Insert_X']), float(row['Insert_Y'])),
                    radius=float(row['Length']),
                    start_angle=float(row['X_start']),
                    end_angle=float(row['Y_start']),
                    dxfattribs={'layer': layer}
                )
        except Exception as e:
            print(f"Error processing block: {str(e)}")
            continue
    
    # Process lines dataframe
    for _, row in lines_df.iterrows():
        try:
            layer = str(row['Layer'])
            if pd.isna(layer) or layer.lower() == 'nan':
                layer = '0'
                
            # Create line with layer information
            msp.add_line(
                start=(float(row['X_start']), float(row['Y_start']), 0),  # Z=0 for 2D compatibility
                end=(float(row['X_end']), float(row['Y_end']), 0),
                dxfattribs={'layer': layer}
            )
        except Exception as e:
            print(f"Error processing line: {str(e)}")
            continue
    
    try:
        # Save the DXF file
        doc.saveas(output_path)
        return f"DXF file created successfully at {output_path}"
    except Exception as e:
        return f"Error saving DXF file: {str(e)}"

def convert_data_to_dxf(blocks_df, lines_df, output_path='output.dxf'):
    """
    Convert DataFrames to DXF file.
    
    Parameters:
    blocks_df (pd.DataFrame): DataFrame containing block information
    lines_df (pd.DataFrame): DataFrame containing lines information
    output_path (str): Path where the DXF file will be saved
    """
    try:
        # Clean and prepare numeric columns
        numeric_cols = ['X_start', 'Y_start', 'X_end', 'Y_end', 'Insert_X', 'Insert_Y', 'Length']
        
        # Convert numeric columns in both dataframes
        for df in [blocks_df, lines_df]:
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return create_dxf_from_dataframes(blocks_df, lines_df, output_path)
    except Exception as e:
        return f"Error during conversion: {str(e)}"


# In[121]:





# In[125]:


def remove_lines_keep_blocks(input_dxf, output_dxf):
    # Load the DXF file
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()  # Access the model space

    # List of entity types that are not blocks
    entities_to_remove = ['LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC', 'CIRCLE', 'ELLIPSE', 'SPLINE', 'POINT', 'HATCH', 'TEXT', 'MTEXT', 'DIMENSION']

    # Iterate over entities in reverse order to safely delete items
    for entity in reversed(msp):
        # If entity type is in the removal list, delete it
        if entity.dxftype() in entities_to_remove:
            msp.delete_entity(entity)

    # Save the modified DXF file
    doc.saveas(output_dxf)
    print(f"Saved modified DXF with only blocks to '{output_dxf}'")

  # Path to save the modified DXF


# In[133]:


def create_dxf_from_dataframe_special(df,filename ,output_filename):
    doc = ezdxf.readfile(filename)
    
    # Create dictionary to store layers
    layers = {}
    
    for index, row in df.iterrows():
        layer_name = str(row['Layer'])  # Convert to string
        if not layer_name or layer_name == 'nan':  # Check for empty or NaN values
            continue
        
        if layer_name not in layers:
            if layer_name == '0':
                continue
                #print(f"Creating layer '{layer_name}'")  # Debug print
                layers[layer_name] = doc.layers.new(name=layer_name)  # Create new layer if it doesn't exist
        
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
                text = row['Text']
                msp.add_text(text, dxfattribs={'insert': insert, 'layer': layer_name})
            elif row['Type'] == 'MTEXT':
                insert = (row['X_insert'], row['Y_insert'])
                text = row['Text']
                msp.add_mtext(text, dxfattribs={'insert': insert, 'layer': layer_name})

    doc.saveas(output_filename)
    return output_filename



def process_dxf_file(input_path, output_path, temp_dir=None):
    try:
        logger.info("Starting DXF file processing")
        # Step 1: Convert DXF to lines (process polyline to lines)
        conversion_result = remove_polylines(input_path, output_path)
        logger.info(f"Conversion result: {conversion_result}")
        remove_lines_keep_blocks(output_path, 'blocks_only.dxf')
        # Step 2: Load DXF data into a DataFrame
        df = Dxf_to_DF(output_path)
        blocks = extract_blocks_to_df(output_path)
        # Step 3: Adjust the X and Y coordinates for proper alignment
        step1 = adjust_Xstart_ystart(df)
        # Step 4: Filter the data for 'MTEXT' type
        step2 = step1[step1['Type'] == 'MTEXT']
        # Step 5: Process the DXF data for connections
        connected_df, overlapped_df, overlapped_lines = process_dxf_dataframe_with_connections(step1)
        # Step 6: Optionally, split lines at intersections (if needed)
        # split_lines = split_lines_at_intersections(connected_df)
        # Step 7: Remove duplicates and combine the processed data
        connected_df.drop_duplicates(inplace=True)
        # Combine processed dataframes (connected lines, blocks, and MTEXT)
        step3 = pd.concat((connected_df,step2))
        step3.drop_duplicates(inplace=True) 
        final_df = pd.concat([step3, blocks]).drop_duplicates()
        # Extract unique layer names for final output
        layer_list = final_df['Layer'].unique().tolist()
        # Step 8: Generate the final DXF file (save as a new file)
        output_filename = 'Final_output.dxf'
        final_output_path = os.path.join(settings.BASE_DIR,'Temp',output_filename) 
        # Generate final DXF with combined data
        create_dxf_from_dataframe_special(step3,'blocks_only.dxf' ,output_filename)
        # Prepare response data
        final_dict = {
            'Dxf_path': input_path,
            'Layer_names': layer_list,
            'Number_overlapped_lines': overlapped_lines
        }
        logger.info(f"DXF processing completed. Final file saved to {final_output_path}")
        return final_dict, output_filename
    except Exception as e:
        logger.error(f"Error processing DXF file: {str(e)}")
        return None
