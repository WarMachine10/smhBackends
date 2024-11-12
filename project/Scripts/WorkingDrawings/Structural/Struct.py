# %% [markdown]
# # Nested functions

# %%
import ezdxf
import math
import pandas as pd
import numpy as np
from typing import Tuple
import warnings
from collections import Counter
from math import sqrt

# Suppress SettingWithCopyWarning from pandas
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# %%
def adjust_dxf_coordinates_to00(filename, output_filename):
    # Read the DXF file
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
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

        except AttributeError:
            # Skip any entities that cannot be updated
            continue

    # Save the modified DXF file
    try:
        doc.saveas(output_filename)
        print(f"File saved as: {output_filename}")
    except IOError:
        print(f"Failed to save file: {output_filename}")

# %%
def calculate_length(start, end):
    """Calculates the 3D length between two points with error handling for missing coordinates."""
    try:
        return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2 + (end.z - start.z)**2)
    except AttributeError:
        # Return a default length of 0 if start or end points are malformed
        return 0.0

# DXF to PANDAS DATAFRAME
def Dxf_to_DF(filename):
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return pd.DataFrame()  # Return an empty DataFrame on failure
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return pd.DataFrame()

    msp = doc.modelspace()
    entities_data = []

    for entity in msp:
        entity_data = {'Type': entity.dxftype()}

        try:
            entity_data['Layer'] = entity.dxf.layer
        except AttributeError:
            entity_data['Layer'] = 'Unknown'

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

            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                entity_data.update({
                    'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                    'Radius': radius
                })

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

            elif entity.dxftype() == 'TEXT':
                insert = entity.dxf.insert
                text = entity.dxf.text
                entity_data.update({
                    'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                    'Text': text
                })

            elif entity.dxftype() == 'MTEXT':
                text = entity.plain_text()
                insertion_point = entity.dxf.insert
                entity_data.update({
                    'Text': text,
                    'X_insert': insertion_point.x,
                    'Y_insert': insertion_point.y,
                    'Z_insert': insertion_point.z
                })

        except AttributeError:
            # Skip the entity if key attributes are missing
            continue

        entities_data.append(entity_data)

    # Return a DataFrame of all extracted entities
    return pd.DataFrame(entities_data)


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
import ezdxf

def four_corners(input_file, output_file, max_along_x, max_along_y, width=9, height=12):
    '''
    Upstream Libraries:
    (1) ezdxf

    Upstream Functions needed:
    (1) create_box: Creates a box in the DXF modelspace.
    
    Function Description:
    This function adds four boxes to an existing DXF file. 
    The boxes are placed in the corners of the drawing area based on the calculated minimum and 
    maximum coordinates of the entities within the DXF file. 
    The first box is positioned at the bottom-left corner, the second at the top-left corner,
    the third at the top-right corner, and the fourth at the bottom-right corner. 
    The width and height of the boxes can be specified as parameters.

    Parameters:
    (1) input_file: The path to the input DXF file (str).
    (2) output_file: The path to save the modified DXF file (str).
    (3) width: The width of each box (float, default is 9 units).
    (4) height: The height of each box (float, default is 12 units).
    
    Returns:
    None: This function does not return any values but saves the modified DXF file to the
    specified output path.
    '''

    # Load the existing DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()

    # Get the bounds of the drawing to calculate max coordinates
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')

    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            min_x = min(min_x, start.x, end.x)
            min_y = min(min_y, start.y, end.y)
            max_x = max(max_x, start.x, end.x)
            max_y = max(max_y, start.y, end.y)

    # Add the four boxes using the create_box function
    create_box(msp, (0, 0), width, height)                # Bottom-left
    create_box(msp, (0, max_y), width, -height)          # Top-left
    create_box(msp, (max_x, max_y), -width, -height)     # Top-right
    create_box(msp, (max_x, 0), -width, height)          # Bottom-right

    # Save the modified DXF file
    doc.saveas(output_file)


def create_box(msp, start_point, width, height):
    '''
    Upstream Libraries:
    - ezdxf 
    
    Upstream Functions needed:
    None
    
    Function Description:
    This function creates a rectangular box in the DXF modelspace with a specified width and height, 
    starting from the given start_point. It adds a closed polyline and fills it with a red hatch.

    Parameters:
    (1) msp: The modelspace of the DXF document (ezdxf.modelspace).
    (2) start_point: The starting point for the box (tuple of (x, y)).
    (3) width: The width of the box (float).
    (4) height: The height of the box (float).
    
    Returns:
    None: This function does not return any values but modifies the DXF modelspace by adding the box.
    '''
    p1 = start_point
    p2 = (p1[0] + width, p1[1])
    p3 = (p2[0], p2[1] + height)
    p4 = (p1[0], p1[1] + height)
    
    # Create a closed polyline for the box and set its color to red (DXF color index 1)
    points = [p1, p2, p3, p4, p1]
    polyline = msp.add_lwpolyline(points, close=True, dxfattribs={'color': 1})  # Color 1 is red
    
    # Add a hatch to fill the polyline with red color
    hatch = msp.add_hatch(color=1)  # Color 1 is red in DXF color index
    hatch.paths.add_polyline_path(points, is_closed=True)


# %%
def Boundary_1(input_file, output_file, target_x=9, tolerance=1, width=9, height=12, max_along_y=None):
    """
    Function to add multiple boxes to a DXF file along vertical lines at a specified x-coordinate.
    Parameters:
        - input_file: Path to the input DXF file (str).
        - output_file: Path to save the modified DXF file (str).
        - target_x: Target x-coordinate for vertical lines (default: 9).
        - tolerance: Tolerance for matching vertical lines (default: 1).
        - width: Width of the boxes to be added (default: 9 units).
        - height: Height of the boxes to be added (default: 12 units).
        - max_along_y: Maximum y-coordinate value (float, must be provided).
    """
    
    if max_along_y is None:
        raise ValueError("max_along_y must be provided and cannot be None")
    
    # Load DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()

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
        num_boxes = max(0, math.ceil((max_along_y - 9) / 144) - 2)

        # Initial start point for the first box
        start_point = (0, 153)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1])
            p3 = (p2[0], p1[1] + height)
            p4 = (p1[0], p3[1])

            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True)

            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1)  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0], start_point[1] + 156)


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
                msp.add_lwpolyline(points, close=True)
                
                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1)  # DXF color index: 1 is red
                hatch.paths.add_polyline_path(points, is_closed=True)

    # Save the modified DXF file
    doc.saveas(output_file)
    
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
def Boundary_2(input_file, output_file, width=12, height=9, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Upstream Libraries:
    (1) ezdxf
    (2) collections.Counter

    Function Description:
    This function identifies horizontal lines near a specified y-coordinate within a tolerance 
    range and creates boxes at filtered x-coordinates along these lines. It adds these closed
    polylines (boxes) into the modelspace of a DXF file. The boxes are filled with red color 
    using hatches. The width and height of the boxes are customizable.

    Parameters:
    (1) input_file: The path to the input DXF file (str).
    (2) output_file: The path to save the modified DXF file (str).
    (3) width: The width of each box (float, default is 12 units).
    (4) height: The height of each box (float, default is 9 units).
    (5) tolerance: The tolerance for matching the y-coordinate of horizontal lines (float, default is 1 unit).
    (6) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (7) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    None: This function does not return any values directly. Instead, it saves the modified DXF file
    with the added boxes and hatches to the specified output path (`output_file`).
    '''
    
    # Load the DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()

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
        num_boxes = max(0, math.ceil((max_along_x - 9) / 144) - 2)


        # Initial start point for the first box
        start_point = (153, max_along_y)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] + width, p1[1])
            p3 = (p2[0], p1[1] - height)
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True)
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1)  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0] + 156, start_point[1])

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
            msp.add_lwpolyline(points, close=True)
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1)  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    # Save the modified DXF file
    doc.saveas(output_file)

# %%
def Boundary_3(input_file, output_file, width=9, height=12, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Upstream Libraries:
    (1) ezdxf
    (2) collections.Counter

    Function Description:
    This function identifies vertical lines near a specified x-coordinate within a tolerance range 
    and creates boxes at filtered y-coordinates along these lines. It adds closed polylines (boxes) 
    into the modelspace of a DXF file. The boxes are filled with red color using hatches. The width
    and height of the boxes are customizable.

    Parameters:
    (1) input_file: The path to the input DXF file (str).
    (2) output_file: The path to save the modified DXF file (str).
    (3) width: The width of each box (float, default is 9 units).
    (4) height: The height of each box (float, default is 12 units).
    (5) tolerance: The tolerance for matching the x-coordinate of vertical lines (float, default is 1 unit).
    (6) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (7) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    None: This function does not return any values directly. Instead, it saves the modified DXF 
    file with the added boxes and hatches to the specified output path (`output_file`).
    '''
    
    # Load the DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()

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
        num_boxes = math.ceil((max_along_y - 9) // 144)- 2

        # Initial start point for the first box
        start_point = (max_along_x, 153)

        # Draw boxes based on num_boxes
        for _ in range(int(num_boxes)):
            # Calculate the corners of the box
            p1 = start_point
            p2 = (p1[0] - width, p1[1]) 
            p3 = (p2[0], p1[1] + height) 
            p4 = (p1[0], p3[1])
            
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True)
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1)  # DXF color index: 1 is red
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Update start_point for the next box
            start_point = (start_point[0], start_point[1] + 144)

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
            msp.add_lwpolyline(points, close=True)
            
            # Add a red hatch to fill the box
            hatch = msp.add_hatch(color=1)  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

    # Save the modified DXF file
    doc.saveas(output_file)

# %%
def Boundary_4(input_file, output_file, width=12, height=9, tolerance=1, max_along_x=None, max_along_y=None):
    '''
    Upstream Libraries:
    (1) ezdxf
    (2) collections.Counter

    Upstream Functions needed:
    (1) remove_first_last_if_conditions_met: Filters x-coordinates based on specific conditions.

    Function Description:
    This function identifies horizontal lines near a specified y-coordinate within a tolerance range 
    and creates boxes at filtered x-coordinates along these lines. It adds closed polylines (boxes) 
    into the modelspace of a DXF file, filling them with red color using hatches. The width and height 
    of the boxes are customizable. The function also tracks if a trim condition is applied to the 
    coordinates, and handles special cases for box creation based on the provided `max_along_x`.

    Parameters:
    (1) input_file: The path to the input DXF file (str).
    (2) output_file: The path to save the modified DXF file (str).
    (3) width: The width of each box (float, default is 12 units).
    (4) height: The height of each box (float, default is 9 units).
    (5) tolerance: The tolerance for matching the y-coordinate of horizontal lines (float, default is 1 unit).
    (6) max_along_x: The maximum x-coordinate value in the drawing (float, must be provided).
    (7) max_along_y: The maximum y-coordinate value in the drawing (float, must be provided).

    Returns:
    A dictionary containing two keys:
    (1) 'trim_applied': Boolean indicating if the trim condition was applied to the x-coordinates.
    (2) 'new_x_coordinates_empty': Boolean indicating if no x-coordinates were left after trimming.

    Function Steps:
    1. Loads the DXF file and identifies horizontal lines near the target y-coordinate (default 9).
    2. Extracts x-coordinates from the lines and filters them based on occurrence.
    3. Applies trimming logic to remove first and last coordinates if conditions are met.
    4. Adds boxes at filtered x-coordinates, with special handling if trimming was applied or not.
    5. Saves the modified DXF file to the specified output path.
    '''
    
    # Load the DXF file
    doc = ezdxf.readfile(input_file)
    msp = doc.modelspace()

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
                msp.add_lwpolyline(points, close=True)
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1)  # Red color
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
                msp.add_lwpolyline(points, close=True)
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1)  # Red color
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
                msp.add_lwpolyline(points, close=True)
                
                # Add a hatch to fill the polyline with red color
                hatch = msp.add_hatch(color=1)  # Red color
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
                    msp.add_lwpolyline(points, close=True)
                    
                    # Add a hatch to fill the polyline with red color
                    hatch = msp.add_hatch(color=1)  # Red color
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
                    msp.add_lwpolyline(points, close=True)

                    # Add a hatch to fill the polyline with red color
                    hatch = msp.add_hatch(color=1)  # Red color
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
                        msp.add_lwpolyline(points, close=True)

                        # Add a hatch to fill the polyline with red color
                        hatch = msp.add_hatch(color=1)  # Red color
                        hatch.paths.add_polyline_path(points, is_closed=True)
            else:
                pass
            
    # Save the modified DXF file
    doc.saveas(output_file)


# %% [markdown]
# ## Helper function

# %%
import numpy as np
import pandas as pd
import ezdxf
from sklearn.decomposition import PCA

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
# # Nested functions

# %%
def process_single_walls_left(df, max_along_x, max_along_y, dxf_file, new_dxf_name, width=12, height=9, variance_threshold=0.95, tolerance=1e-3):
    '''
    Upstream Libraries:
    - numpy 
    - pandas 
    - ezdxf 

    Upstream Functions needed:
    - filter_horizontal_lines: Filters horizontal lines based on PCA.
    - extract_matching_line_pairs: Extracts pairs of matching lines.

    Function Description:
    This function processes single walls on the left side of a DXF file. It filters horizontal lines, extracts 
    matching pairs of lines, and creates boxes in the specified DXF file based on those lines. 
    The boxes are filled with red color.

    Parameters:
    - df: Input DataFrame containing line data.
    - max_along_x: Maximum x-coordinate value (float).
    - max_along_y: Maximum y-coordinate value (float).
    - dxf_file: Path to the original DXF file (string).
    - new_dxf_name: Name of the new DXF file to save (string).
    - width: Width of the box to create (float, default is 12).
    - height: Height of the box to create (float, default is 9).
    - variance_threshold: Threshold (float) for PCA variance to consider a line horizontal. Default is 0.95.
    - tolerance: Tolerance for comparing floating-point values (float, default is 1e-3).

    Returns:
    None: This function does not return any values but saves the modified DXF file to the specified output path.
    '''

    # Step 1: Filter horizontal lines using the refactored helper function
    df_filtered = filter_horizontal_lines(df, max_along_x, max_along_y, variance_threshold)

    # Step 2: Extract matching line pairs using the refactored helper function
    df_pairs = extract_matching_line_pairs(df_filtered)

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

    # Step 4: Create boxes in the DXF file without overlapping
    def create_box_from_df(df_extracted, width, height, dxf_file, new_dxf_name):
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()
        
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
                msp.add_lwpolyline(points, close=True)
                
                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1)  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)

        # Save the modified DXF file
        doc.saveas(new_dxf_name)

    create_box_from_df(df_extracted, width, height, dxf_file, new_dxf_name)


# %%
def process_single_walls_right(df, max_along_x, max_along_y, dxf_file, modified_dxf_file, width=12, height=9, variance_threshold=0.95, tolerance=1e-3):
    '''
    Upstream Libraries:
    - numpy 
    - pandas 
    - ezdxf 

    Upstream Functions needed:
    - filter_horizontal_lines: Filters horizontal lines based on PCA.
    - extract_matching_line_pairs: Extracts pairs of matching lines.

    Function Description:
    This function processes single walls on the right side of a DXF file by filtering horizontal lines, extracting
    matching pairs, and creating boxes based on those lines. The boxes are filled with red color, and the modified DXF file
    is saved with a new name.

    Parameters:
    - df: Input DataFrame containing line data.
    - max_along_x: Maximum x-coordinate value (float).
    - max_along_y: Maximum y-coordinate value (float).
    - dxf_file: Path to the original DXF file (string).
    - modified_dxf_file: Name of the new DXF file to save (string).
    - width: Width of the box to create (float, default is 12).
    - height: Height of the box to create (float, default is 9).
    - variance_threshold: Threshold (float) for PCA variance to consider a line horizontal. Default is 0.95.
    - tolerance: Tolerance for comparing floating-point values (float, default is 1e-3).

    Returns:
    None: The function saves the modified DXF file to the specified output path.
    '''

    # Step 1: Filter horizontal lines using PCA
    df_filtered = filter_horizontal_lines(df, max_along_x, max_along_y, variance_threshold)

    # Step 2: Extract matching line pairs
    df_pairs = extract_matching_line_pairs(df_filtered)

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

    # Step 4: Create boxes in the DXF file without overlapping
    def create_box_from_df_extracted_right(df_extracted_right, width, height, dxf_file, modified_dxf_file):
        doc = ezdxf.readfile(dxf_file)
        msp = doc.modelspace()
        
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
                msp.add_lwpolyline(points, close=True)
                
                # Add a red hatch to fill the box
                hatch = msp.add_hatch(color=1)  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)

        # Save the modified DXF file
        doc.saveas(modified_dxf_file)

    create_box_from_df_extracted_right(df_extracted_right, width, height, dxf_file, modified_dxf_file)


# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import ezdxf

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

def create_boxes_on_dxf(input_dxf_file, output_dxf_file, x_coor_up, y_coor_up, width=9, height=12):
    doc = ezdxf.readfile(input_dxf_file)
    msp = doc.modelspace()

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

    doc.saveas(output_dxf_file)

def single_wall_up(df, max_along_x, max_along_y, input_dxf_file, output_dxf_file):
    df_filtered_vertical = filter_vertical_lines_by_pca(df, max_along_y)
    df_extracted_lines = extract_specific_lines(df_filtered_vertical, max_along_x, max_along_y)
    df_paired_lines, y_coor_up, x_coor_up = pair_lines_by_x_difference(df_extracted_lines, df_filtered_vertical, max_along_y=max_along_y)
    create_boxes_on_dxf(input_dxf_file, output_dxf_file, x_coor_up, y_coor_up)

# %% [markdown]
# # For all inner columns

# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
import ezdxf
import math

def create_boxes_in_df_x_equal(df_x_equal, temp_v, temp_h, input_dxf_file, output_dxf_file, width=9, height=12, tolerance_v=0.5, tolerance_h=0.5, tolerance_y=0.1, buffer=24, radius=12):
    """
    Matches rows in `df_x_equal` against `temp_v` and `temp_h` with specified tolerance thresholds,
    and draws boxes based on matched conditions. Modifies an input DXF file and saves an updated version.
    
    Parameters:
    - df_x_equal (DataFrame): DataFrame containing X1, X2, Y1, Y2 columns for matching conditions.
    - temp_v (DataFrame): DataFrame containing X_start, Y_start, Y_end for vertical alignment checking.
    - temp_h (DataFrame): DataFrame containing X_start, X_end, Y_start for horizontal alignment checking.
    - input_dxf_file (str): Path to the input DXF file.
    - output_dxf_file (str): Path to save the output DXF file with added boxes.
    - width (float): Width of the box to be drawn (default is 12).
    - height (float): Height of the box to be drawn (default is 12).
    - tolerance_v (float): Vertical tolerance for matching X coordinates.
    - tolerance_h (float): Horizontal tolerance for matching Y coordinates.
    - tolerance_y (float): Tolerance for matching Y coordinates with temp_h.
    - buffer (float): Additional spacing around each box to avoid overlap (default is 24).
    - radius (float): Minimum radius to avoid overlapping with nearby boxes.
    """
    
    # Load the input DXF file
    doc = ezdxf.readfile(input_dxf_file)
    msp = doc.modelspace()

    # To track created box positions and prevent overlapping
    created_boxes = []

    for idx, row in df_x_equal.iterrows():
        X1, X2, Y1, Y2 = row['X1'], row['X2'], row['Y1'], row.get('Y2', None)
        match_found = False

        # Step 1: Check X1 +5 and X1 +9 matches in `temp_v`
        for offset in [5, 9]:
            for _, temp_v_row in temp_v.iterrows():
                X_start_temp = temp_v_row['X_start']
                if abs((X1 + offset) - X_start_temp) <= tolerance_v:
                    match_found = True
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
                        break
                if match_found:
                    break

        # Only create the box if a match was found and no overlapping occurs
        if match_found:
            Y = min(Y1, Y2)
            X = X1
            center_point = (X + width / 2, Y + height / 2)
            box_points = [
                (X, Y),
                (X + width, Y),
                (X + width, Y + height),
                (X, Y + height)
            ]
            
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
                # Add the box to the DXF file
                points = box_points + [box_points[0]]  # Closing the loop for polyline
                msp.add_lwpolyline(points, close=True)
                
                hatch = msp.add_hatch(color=1)  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
                
                # Add box coordinates to created_boxes list
                created_boxes.append(box_points)

    # Save the updated DXF file to the specified output path
    doc.saveas(output_dxf_file)


# %%
import ezdxf
import math

def create_boxes_in_df_y_equal(input_dxf_path, output_dxf_path, df_y_equal, temp_h, temp_v, tolerance_h=0.5, tolerance_v=0.5, width=12, height=9, radius=12):
    """
    Matches rows in `df_y_equal` against `temp_h` with specified tolerance thresholds,
    and draws boxes based on matched conditions. Prevents overlapping boxes within
    a specified radius and ensures no box is created if neither check is fulfilled.
    
    Parameters:
    - input_dxf_path (str): Path to the input DXF file.
    - output_dxf_path (str): Path to save the output DXF file with added boxes.
    - df_y_equal (DataFrame): DataFrame containing Y1, X1, X2 columns for matching conditions.
    - temp_h (DataFrame): DataFrame containing Y_start for horizontal alignment checking.
    - temp_v (DataFrame): DataFrame containing X_start, Y_start, Y_end for vertical alignment checking.
    - tolerance_h (float): Horizontal tolerance for matching Y coordinates.
    - tolerance_v (float): Vertical tolerance for matching X coordinates.
    - width (float): Width of the box to be drawn (default is 12).
    - height (float): Height of the box to be drawn (default is 12).
    - radius (float): Minimum radius to avoid overlapping with nearby boxes.
    """
    
    # Open the input DXF file
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()

    # List to store coordinates of created boxes to prevent overlapping
    created_boxes = []

    # Iterate over each row in df_y_equal
    for idx, row in df_y_equal.iterrows():
        Y1, X1, X2 = row['Y1'], row['X1'], row['X2']
        match_found = False

        # First check: matches in temp_h for Y1 + 5 and Y1 + 9 within tolerance
        for offset in [5, 9]:
            for _, temp_h_row in temp_h.iterrows():
                Y_start_temp = temp_h_row['Y_start']
                if abs((Y1 + offset) - Y_start_temp) <= tolerance_h:
                    match_found = True
                    break
            if match_found:
                break

        # Second check: matches in temp_h for Y1 - 5 and Y1 - 9 within tolerance, if no match found
        if not match_found:
            for offset in [-5, -9]:
                for _, temp_h_row in temp_h.iterrows():
                    Y_start_temp = temp_h_row['Y_start']
                    if abs((Y1 + offset) - Y_start_temp) <= tolerance_h:
                        match_found = True
                        break
                if match_found:
                    break

        # Only create the box if a match was found and no overlapping occurs
        if match_found:
            X = min(X1, X2)
            Y = Y1
            center_point = (X + width / 2, Y + height / 2)
            box_points = [
                (X, Y),
                (X + width, Y),
                (X + width, Y + height),
                (X, Y + height)
            ]
            
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
                # Add the box to the DXF file
                points = [
                    (X, Y),
                    (X + width, Y),
                    (X + width, Y + height),
                    (X, Y + height),
                    (X, Y)  # Closing the loop for polyline
                ]
                
                msp.add_lwpolyline(points, close=True)
                
                hatch = msp.add_hatch(color=1)  # Red color
                hatch.paths.add_polyline_path(points, is_closed=True)
                
                # Add box coordinates to created_boxes list
                created_boxes.append(box_points)

    # Save the modified DXF file
    doc.saveas(output_dxf_path)


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
from math import sqrt

def create_boxes_in_df_other_groupA(df_other_groupA, width, height, input_dxf_file, output_dxf_file):
    """
    Adds boxes to a DXF file based on coordinates in df_other_groupA, ensuring no overlap within a 12-unit radius.

    Parameters:
    - df_other_groupA (DataFrame): DataFrame containing X1, Y1, X2, Y2 columns for box placement.
    - width (float): Width of each box.
    - height (float): Height of each box.
    - input_dxf_file (str): Path to the input DXF file.
    - output_dxf_file (str): Path to save the output DXF file with added boxes.
    """
    # Load the input DXF file
    doc = ezdxf.readfile(input_dxf_file)
    msp = doc.modelspace()

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

        # Check for overlap with existing boxes within a radius of 12
        overlapping = False
        for existing_center in created_boxes:
            distance = sqrt((box_center[0] - existing_center[0])**2 + (box_center[1] - existing_center[1])**2)
            if distance < 84:
                overlapping = True
                break

        # Only create the box if no overlap within the radius
        if not overlapping:
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True)

            # Add a hatch to fill the polyline with red color
            hatch = msp.add_hatch(color=1)  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Add the box center to created_boxes list to track position
            created_boxes.append(box_center)

    # Save the updated DXF file to the specified output path
    doc.saveas(output_dxf_file)


# %%
import math

def create_boxes_in_df_other_groupB(df_other_groupB, width, height, input_dxf_file, output_dxf_file):
    """
    Adds boxes to a DXF file based on coordinates in df_other_groupB, ensuring no overlap within a radius of 12 units.

    Parameters:
    - df_other_groupB (DataFrame): DataFrame containing X1, Y1, X2, Y2 columns for box placement.
    - width (float): Width of each box.
    - height (float): Height of each box.
    - input_dxf_file (str): Path to the input DXF file.
    - output_dxf_file (str): Path to save the output DXF file with added boxes.
    """
    # Load the input DXF file
    doc = ezdxf.readfile(input_dxf_file)
    msp = doc.modelspace()

    # List to store coordinates and centers of created boxes to prevent overlapping
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
            if distance < 84:
                overlapping = True
                break

        # Only create the box if no overlap within the 12-unit radius
        if not overlapping:
            # Create a closed polyline for the box
            points = [p1, p2, p3, p4, p1]
            msp.add_lwpolyline(points, close=True)

            # Add a hatch to fill the polyline with red color
            hatch = msp.add_hatch(color=1)  # Red color
            hatch.paths.add_polyline_path(points, is_closed=True)

            # Add the box coordinates and center to created_boxes list
            created_boxes.append({'points': [p1, p2, p3, p4], 'center': box_center})

    # Save the updated DXF file to the specified output path
    doc.saveas(output_dxf_file)


# %%
def detect_and_label_boxes(input_dxf, output_dxf, label_position='right', offset=1, text_height=1, text_color=7, shift=0):
    '''
    Detects boxes (closed polylines) in a DXF file, labels them around the box (right, left, or top),
    and saves the updated DXF file. Supports hypertuning for label position, text height, color, and shifts.

    Parameters:
    (1) dxf_input_path: The path to the input DXF file (string).
    (2) dxf_output_path: The path to save the output DXF file (string).
    (3) label_position: The position of the label relative to the box ('right', 'left', 'top').
    (4) offset: The distance from the box to the label (float, default is 1).
    (5) text_height: The height of the label text (float, default is 1).
    (6) text_color: The color of the label text (DXF color index, default is 7 - white).
    (7) shift: Shift the text position (float, default is 0, can adjust the left or right position).

    Returns:
    None: The function saves the updated DXF file with labeled boxes.
    '''
    # Load the input DXF file
    doc = ezdxf.readfile(input_dxf)
    msp = doc.modelspace()

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

    # Save the modified DXF file
    doc.saveas(output_dxf)

def add_mtext(msp, text, position, height=1, color=7):
    '''
    Adds MText (multiline text) to the DXF file at the specified position with adjustable height and color.

    Parameters:
    (1) msp: The modelspace of the DXF document (ezdxf.modelspace).
    (2) text: The text to add (string).
    (3) position: The position to place the text (tuple of (x, y)).
    (4) height: The height of the text (float, default is 1).
    (5) color: The color of the text (DXF color index, default is 7 - white).

    Returns:
    None: This function does not return any values.
    '''
    mtext = msp.add_mtext(text, dxfattribs={'char_height': height, 'color': color})
    mtext.set_location(position)


# %%
def create_column_schedule_dataframe(filename, max_along_x, max_along_y):
    """
    Creates a DataFrame for 'Schedule of Column at Ground Floor' with column names,
    dynamic length and width based on box center conditions, and height.

    Parameters:
        filename (str): The filename of the DXF file to analyze.
        max_along_x (float): Maximum x-coordinate value in the drawing.
        max_along_y (float): Maximum y-coordinate value in the drawing.

    Returns:
        DataFrame: A pandas DataFrame with the column schedule.
    """
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None

    msp = doc.modelspace()
    column_names = []
    x_centers = []
    y_centers = []
    lengths = []
    widths = []

    # Collect all labels that start with "C" (e.g., "C1", "C2", etc.)
    for entity in msp.query('MTEXT TEXT'):
        if entity.dxftype() in ['MTEXT', 'TEXT']:
            label = entity.dxf.text.strip()
            if label.startswith('C'):
                column_names.append(label)
                
                # Assuming `entity.dxf.insert` represents the starting point
                start_x, start_y = entity.dxf.insert.x, entity.dxf.insert.y

                # Define corners for the box based on start point
                p1 = (start_x, start_y)
                p2 = (p1[0] + 12, p1[1])   # Assume default box width of 12
                p3 = (p2[0], p1[1] - 9)    # Assume default box height of 9
                p4 = (p1[0], p3[1])

                # Calculate box width and height dynamically
                width = abs(p2[0] - p1[0])
                height = abs(p1[1] - p3[1])
                
                # Calculate the center of the box
                x_center = (p1[0] + p3[0]) / 2
                y_center = (p1[1] + p3[1]) / 2
                x_centers.append(x_center)
                y_centers.append(y_center)

                # Determine Length and Width based on center conditions
                if ((0 <= x_center <= 9 or max_along_x - 9 <= x_center <= max_along_x) and
                    (0 <= y_center <= 9 or max_along_y - 9 <= y_center <= max_along_y)):
                    lengths.append(9)
                    widths.append(12)
                else:
                    lengths.append(12)
                    widths.append(12)

    # Create a DataFrame with the column names and calculated dimensions
    df = pd.DataFrame({
        'Columns No': column_names,
        'Length': lengths,
        'Width': widths,
        'Height': 10  # Fixed height for all boxes
    })


    return df

# %%
def change_line_color_to_light_gray(filename, output_dxf):
    """
    Changes the color of all lines in a DXF file to extreme light gray.

    Parameters:
        filename (str): The path to the input DXF file.
        output_filename (str): The path to save the modified DXF file with updated line color.

    Returns:
        None
    """
    try:
        # Read the input DXF file
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return

    # Set the light gray color using RGB values (240, 240, 240)
    R, G, B = 240, 240, 240
    light_gray_color = (R << 16) | (G << 8) | B  # Convert RGB to a single integer

    # Modify the color of all lines in the modelspace
    msp = doc.modelspace()

    for entity in msp:
        if entity.dxftype() == 'LINE':
            # Set the color of the line to light gray (RGB value)
            entity.dxf.color = 256  # 256 corresponds to "BYLAYER" which will take the layer's color
            entity.dxf.true_color = light_gray_color  # Set the true color to light gray

    # Save the modified DXF file
    doc.saveas(output_dxf)


# %% [markdown]
# # For Beams

# %%
def extract_columns(filename):
    """
    Extract columns of vertically and horizontally aligned boxes based on a DXF file.
    Outputs four DataFrames: C1_ver_col, C2_hor_col, C3_ver_col, and C4_hor_col.

    Parameters:
        filename (str): The path to the input DXF file.

    Returns:
        tuple: A tuple containing four DataFrames (C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col).
    """
    try:
        # Read the DXF file
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None, None, None, None
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None, None, None, None

    msp = doc.modelspace()
    
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

# %%
import pandas as pd
import ezdxf
import math

def calculate_edges(entity):
    points = entity.get_points('xy') if entity.dxftype() == 'LWPOLYLINE' else [vertex.dxf.location for vertex in entity.vertices]
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    width = max_x - min_x
    top_edge = min([p for p in points if p[1] == max_y], key=lambda p: p[0])
    bottom_edge = min([p for p in points if p[1] == min_y], key=lambda p: p[0])

    return top_edge, bottom_edge, width

def connect_edges_vertically_boundary_1(input_file, output_dxf, aligned_boxes_df, beam_count=0):
    try:
        doc = ezdxf.readfile(input_file)
    except IOError:
        print(f"Cannot open file: {input_file}")
        return None, None
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {input_file}")
        return None, None

    msp = doc.modelspace()
    box_edges = {}
    beam_info_rows = []  # Initialize a list to collect rows for the beam info DataFrame

    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            top_edge, bottom_edge, width = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'top': top_edge, 'bottom': bottom_edge, 'width': width}

    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, None

    label_counter = 1
    label_count = 0

    for i in range(len(aligned_boxes_df) - 1):
        label_lower = aligned_boxes_df.iloc[i, 0]
        label_upper = aligned_boxes_df.iloc[i + 1, 0]

        upper_bottom_edge = box_edges[label_upper]['bottom']
        lower_top_edge = box_edges[label_lower]['top']
        width = box_edges[label_upper]['width']

        msp.add_line(upper_bottom_edge, lower_top_edge, dxfattribs={'color': 3})  # Green color
        upper_bottom_edge_offset = (upper_bottom_edge[0] + width, upper_bottom_edge[1])
        lower_top_edge_offset = (lower_top_edge[0] + width, lower_top_edge[1])
        msp.add_line(upper_bottom_edge_offset, lower_top_edge_offset, dxfattribs={'color': 3})  # Green color

        midpoint_x = upper_bottom_edge[0] - width * 1
        midpoint_y = (upper_bottom_edge[1] + lower_top_edge[1]) / 2
        label_text = f"B{label_counter}"

        text_entity = msp.add_text(label_text, dxfattribs={'color': 2, 'height': 0.5})
        text_entity.dxf.insert = (midpoint_x, midpoint_y)
        text_entity.dxf.rotation = 90

        length = math.sqrt((lower_top_edge[0] - upper_bottom_edge[0]) ** 2 + (lower_top_edge[1] - upper_bottom_edge[1]) ** 2)
        beam_info_rows.append({'beam names': label_text, 'length': length})

        label_counter += 1
        label_count += 1

    # Create the DataFrame once from the collected rows
    beam_info_df = pd.DataFrame(beam_info_rows)

    doc.saveas(output_dxf)

    return label_count, beam_info_df  # Return label count and the beam information DataFrame


# %%
import pandas as pd
import ezdxf
import math

def connect_edges_horizontally_boundary_2(input_file, output_dxf, aligned_boxes_df, beam_count, beam_info_df):
    """
    Connect horizontally aligned boxes by drawing green lines between their edge points and labeling pairs.

    Parameters:
        input_file (str): Path to the input DXF file.
        output_dxf (str): Path to save the modified DXF file.
        aligned_boxes_df (pd.DataFrame): DataFrame containing labels of horizontally aligned boxes.
        beam_count (int): Starting count for the beam labels.
        beam_info_df (pd.DataFrame): DataFrame to store beam information, with columns 'beam names' and 'length'.

    Returns:
        tuple: Updated beam count after the labels are added, and updated beam_info_df DataFrame.
    """
    try:
        doc = ezdxf.readfile(input_file)
    except IOError:
        print(f"Cannot open file: {input_file}")
        return None, beam_info_df
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {input_file}")
        return None, beam_info_df

    msp = doc.modelspace()
    box_edges = {}

    # Extract edges for each box
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            left_edge, right_edge, height = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'left': left_edge, 'right': right_edge, 'height': height}

    # Ensure the required labels exist in box_edges
    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, beam_info_df

    label_counter = beam_count + 1
    label_count = 0

    for i in range(len(aligned_boxes_df) - 1):
        label_left = aligned_boxes_df.iloc[i, 0]
        label_right = aligned_boxes_df.iloc[i + 1, 0]
        right_edge_left_box = box_edges[label_left]['right']
        left_edge_right_box = box_edges[label_right]['left']
        y_coord = right_edge_left_box[1]
        beam_length = abs(left_edge_right_box[0] - right_edge_left_box[0])

        # Draw the connecting lines
        line1 = msp.add_line((right_edge_left_box[0], y_coord), (left_edge_right_box[0], y_coord), dxfattribs={'color': 3})
        right_edge_left_box_offset = (right_edge_left_box[0], y_coord + height)
        left_edge_right_box_offset = (left_edge_right_box[0], y_coord + height)
        line2 = msp.add_line(right_edge_left_box_offset, left_edge_right_box_offset, dxfattribs={'color': 3})

        midpoint_x = (right_edge_left_box[0] + left_edge_right_box[0]) / 2
        midpoint_y = y_coord + height
        label_text = f"B{label_counter}"

        text_entity = msp.add_text(label_text, dxfattribs={'color': 2, 'height': 0.5})
        text_entity.dxf.insert = (midpoint_x, midpoint_y)

        # Update beam_info_df with pd.concat
        new_row = pd.DataFrame([[label_text, beam_length]], columns=['beam names', 'length'])
        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)

        label_counter += 1
        label_count += 1

    doc.saveas(output_dxf)
    return label_counter, beam_info_df


# %%
import pandas as pd
import ezdxf

def connect_edges_vertically_boundary_3(filename, output_dxf, aligned_boxes_df, beam_count, beam_info_df):
    """
    Connect vertically aligned boxes by drawing green lines between their edge points, labeling pairs, 
    and updating the beam_info_df with beam label and length.

    Parameters:
        filename (str): Path to the DXF file.
        output_dxf (str): Path to save the modified DXF file.
        aligned_boxes_df (pd.DataFrame): DataFrame containing labels of vertically aligned boxes.
        beam_count (int): Starting count for the beam labels.
        beam_info_df (pd.DataFrame): DataFrame to store information about each created beam.
    
    Returns:
        Tuple[int, pd.DataFrame]: Updated beam count and the modified beam_info_df.
    """
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None, beam_info_df
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None, beam_info_df

    msp = doc.modelspace()
    box_edges = {}

    # Extract edges for each box
    for entity in msp:
        if entity.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
            left_edge, right_edge, height = calculate_edges(entity)
            label = f"C{len(box_edges) + 1}"
            box_edges[label] = {'left': left_edge, 'right': right_edge, 'height': height}

    if not all(label in box_edges for label in aligned_boxes_df['connected columns']):
        return None, beam_info_df

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

        x_coord_left_line = right_edge_left_box[0] + 9
        x_coord_right_line = left_edge_right_box[0] - box_width

        # Draw the green vertical lines
        line1 = msp.add_line((x_coord_left_line, right_edge_left_box[1]), (x_coord_left_line, left_edge_right_box[1]), dxfattribs={'color': 3})
        line2 = msp.add_line((x_coord_right_line, right_edge_left_box[1]), (x_coord_right_line, left_edge_right_box[1]), dxfattribs={'color': 3})

        midpoint_y = (right_edge_left_box[1] + left_edge_right_box[1]) / 2
        label_text = f"B{label_counter}"

        text_entity = msp.add_text(label_text, dxfattribs={'color': 2, 'height': 0.5})
        text_entity.dxf.insert = (x_coord_left_line + 0.1, midpoint_y)

        beam_length = abs(right_edge_left_box[1] - left_edge_right_box[1])

        # Use pd.concat to update beam_info_df
        new_row = pd.DataFrame([[label_text, beam_length]], columns=['beam names', 'length'])
        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)

        label_counter += 1
        label_count += 1

    doc.saveas(output_dxf)
    return label_counter, beam_info_df


# %%
import pandas as pd
import ezdxf

def check_horizontal_alignment_boundary_1(filename, max_along_x, C1_ver_col, beam_count, tolerance=15, output_filename=None, beam_info_df=None):
    """
    Check horizontally aligned boxes and verify if any pair of parallel horizontal lines connect specific boxes.
    If a pair is found, draw green parallel horizontal lines on the DXF, label them, and update beam_count.

    Parameters:
        filename (str): The path to the input DXF file.
        max_along_x (int or float): The maximum x-coordinate to limit the search range.
        C1_ver_col (pd.DataFrame): DataFrame containing box coordinates and labels.
        beam_count (int): Initial beam count for labeling pairs of parallel lines.
        tolerance (int or float): The allowable deviation in y-coordinates for horizontal alignment.
        output_filename (str, optional): The name of the output DXF file with green lines added.
        beam_info_df (pd.DataFrame): DataFrame to store beam names and lengths.

    Returns:
        int: Updated beam count after labeling all detected pairs of parallel lines.
        pd.DataFrame: Updated beam_info_df with new rows for each beam.
    """
    try:
        # Read the DXF file
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None, beam_info_df
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None, beam_info_df

    msp = doc.modelspace()

    # Extract boxes and calculate their center points
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

    # Initialize list to store aligned box data
    alignment_data = []

    # Iterate over C1_ver_col from the second to the second-last row
    for i in range(1, len(C1_ver_col) - 1):
        current_label = C1_ver_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue  # Skip if label is not in centers

        current_center = centers[current_label]
        aligned_boxes = []

        # Check horizontally aligned boxes within the specified range and tolerance
        for label, center in centers.items():
            if label == current_label:
                continue  # Skip comparing the box with itself

            # Check if aligned within the x-range and y-tolerance
            if abs(center[1] - current_center[1]) <= tolerance and current_center[0] < center[0] <= max_along_x - 24:
                aligned_boxes.append((label, center[0]))  # Store label and x-coordinate for sorting

        # Sort aligned boxes by x-coordinate for left-to-right order
        aligned_labels = [label for label, _ in sorted(aligned_boxes, key=lambda x: x[1])]

        # Append to alignment_data list
        alignment_data.append({
            'Boundary column': current_label,
            'Horizontal aligned column': aligned_labels
        })

    # Convert to DataFrame
    Boundary_1_connection = pd.DataFrame(alignment_data)
    
    # Detect a single horizontal line between selected box pairs and search for a parallel line
    for index, row in Boundary_1_connection.iterrows():
        boundary_box = row['Boundary column']
        if row['Horizontal aligned column']:
            target_box = row['Horizontal aligned column'][0]  # Get the first aligned box
            
            if boundary_box in centers:
                midpoint_y = centers[boundary_box][1]  # Y coordinate of the boundary box's midpoint
                line_found = False
                first_line_y = None
                first_line_x_min = None
                first_line_x_max = None

                # Search for a single horizontal line that matches criteria
                for entity in msp:
                    if entity.dxftype() == 'LINE':
                        x_start, y_start = entity.dxf.start.x, entity.dxf.start.y
                        x_end, y_end = entity.dxf.end.x, entity.dxf.end.y

                        # Ensure the line is horizontal and check with tolerance
                        if abs(y_start - y_end) <= 0.01:
                            x_min, x_max = min(x_start, x_end), max(x_start, x_end)
                            y_coord = y_start  # since y_start == y_end for a horizontal line

                            # Check if the Y is within tolerance of midpoint_y
                            if abs(y_coord - midpoint_y) <= 10:
                                # Confirm the X range overlaps the expected position (around 9 ± 1)
                                if (abs(x_min - 9) <= 1 or abs(x_max - 9) <= 1):
                                    line_found = True
                                    first_line_y = y_coord
                                    first_line_x_min, first_line_x_max = x_min, x_max
                                    break
                                  
                # If a first line was detected, search for the second parallel line
                if line_found and first_line_y is not None:
                    parallel_line_found = False
                    for entity in msp:
                        if entity.dxftype() == 'LINE':
                            x_start, y_start = entity.dxf.start.x, entity.dxf.start.y
                            x_end, y_end = entity.dxf.end.x, entity.dxf.end.y

                            # Ensure the line is horizontal
                            if abs(y_start - y_end) <= 0.01:
                                x_min, x_max = min(x_start, x_end), max(x_start, x_end)
                                y_coord = y_start  # since y_start == y_end for a horizontal line

                                # Check for parallel conditions: X around 9 ± 1, Y difference of 5 or 9 with ±0.5 tolerance
                                if (abs(x_min - 9) <= 1 or abs(x_max - 9) <= 1) and \
                                   (abs(y_coord - first_line_y - 5) <= 0.5 or abs(y_coord - first_line_y - 9) <= 0.5):
                                    parallel_line_found = True
        
                                    # Draw the green parallel lines and add labels
                                    msp.add_line((first_line_x_min, first_line_y), (first_line_x_max, first_line_y), dxfattribs={'color': 3})
                                    msp.add_line((x_min, y_coord), (x_max, y_coord), dxfattribs={'color': 3})

                                   # Label the beam with the current beam count
                                    beam_label = f"B{beam_count}"
                                    label_x = (first_line_x_min + first_line_x_max) / 2
                                    label_y = (first_line_y + y_coord) / 2

                                    offset_y = 7  # Adjust this value to shift the text vertically

                                    msp.add_text(
                                        beam_label, 
                                        dxfattribs={'height': 2.5, 'color': 3}
                                    ).set_dxf_attrib("insert", (label_x, label_y + offset_y))

                                    # Use pd.concat to update beam_info_df
                                    new_row = pd.DataFrame([[beam_label, abs(first_line_x_max - first_line_x_min)]], columns=['beam names', 'length'])
                                    beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)

                                    # Increment the beam count for the next pair
                                    beam_count += 1
                                    break
                    if not parallel_line_found:
                        pass

    # Save the modified DXF with added green lines and labels
    doc.saveas(output_filename)
    
    return beam_count, beam_info_df


# %%
def check_vertical_alignment_boundary_2(filename, C2_hor_col, max_along_y, beam_count, tolerance=15, output_filename=None, beam_info_df=None):
    """
    Check vertically aligned boxes from the second row to the second-last row in C2_hor_col.
    Sketch two parallel vertical lines connecting aligned boxes and label the beams.
    
    Parameters:
        filename (str): The path to the input DXF file.
        C2_hor_col (pd.DataFrame): DataFrame containing box coordinates and labels.
        max_along_y (float): The maximum Y value to calculate the Y-condition for parallel lines.
        beam_count (int): Initial count for labeling beams.
        tolerance (int or float): The allowable deviation in X-coordinates for vertical alignment.
        output_filename (str): The filename to save the modified DXF file.
        beam_info_df (pd.DataFrame): DataFrame to store beam names and lengths.
        
    Returns:
        tuple: Updated beam count, the output filename with the modified DXF, and updated beam_info_df.
    """
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None, None, beam_info_df
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None, None, beam_info_df

    msp = doc.modelspace()
    
    # Step 1: Extract centers from DXF data
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
    
    # Step 2: Iterate over C2_hor_col and detect vertical lines
    for i in range(1, len(C2_hor_col) - 1):
        current_label = C2_hor_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue  # Skip if label is not in centers

        current_center = centers[current_label]
        aligned_boxes = []

        # Check for vertically aligned boxes based on X-coordinate within tolerance
        for label, center in centers.items():
            if label != current_label and abs(center[0] - current_center[0]) <= tolerance:
                # Calculate distance between current box and aligned box
                distance = sqrt((center[0] - current_center[0])**2 + (center[1] - current_center[1])**2)
                aligned_boxes.append((label, center[1], distance))

        if aligned_boxes:
            # Sort aligned boxes by distance (near to far) and pick the nearest one
            aligned_boxes_sorted = sorted(aligned_boxes, key=lambda x: x[2])  # Sort by distance

            # Use the nearest aligned box for detecting and drawing vertical lines
            target_label, target_y, _ = aligned_boxes_sorted[0]
            if current_label in centers:
                midpoint_x = centers[current_label][0]
                first_line_x = midpoint_x

                # Step 3: Determine Y-coordinates to draw lines based only on box Y-ranges
                min_y = min(current_center[1], target_y)
                max_y = max(current_center[1], target_y)

                # Draw two green parallel vertical lines connecting boxes
                msp.add_line((first_line_x, min_y), (first_line_x, max_y), dxfattribs={'color': 3})
                msp.add_line((first_line_x + 5, min_y), (first_line_x + 5, max_y), dxfattribs={'color': 3})
                
                # Label the beam with 90-degree rotation at calculated midpoint
                beam_label = f"B{beam_count}"
                label_offset_x = 15  # Adjust this value as needed for the leftward shift
                midpoint_y = (min_y + max_y) / 2  # Calculated midpoint Y for label position
                text_entity = msp.add_text(beam_label, dxfattribs={
                    'color': 2,  # Optional: specify a different color for the text
                    'height': 0.5  # Adjust text size as needed
                })
                text_entity.dxf.insert = (midpoint_x - label_offset_x, midpoint_y)
                text_entity.dxf.rotation = 90
                
                
                # Add the new beam information to the DataFrame
                beam_length = max_y - min_y  # Calculate the length of the beam (the vertical line length)
                new_row = pd.DataFrame([[beam_label,beam_length]], columns = ['beam names', 'length'])
                beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)

                # Increment beam count
                beam_count += 1
    
    # Save the modified DXF file with added lines and labels
    if output_filename:
        doc.saveas(output_filename)
    
    return beam_count, beam_info_df


# %%
def check_horizontal_alignment_boundary_3(filename, max_along_x, C3_ver_col, beam_count, output_filename, beam_info_df, tolerance=10): 
    """
    Check horizontally aligned boxes and verify if any pair of parallel horizontal lines connect specific boxes.
    If a pair is found, draw green parallel horizontal lines on the DXF, label them, and update beam_count.

    Parameters:
        filename (str): The path to the input DXF file.
        max_along_x (int or float): The maximum x-coordinate to limit the search range.
        C3_ver_col (pd.DataFrame): DataFrame containing box coordinates and labels.
        beam_count (int): Initial beam count for labeling pairs of parallel lines.
        tolerance (int or float): The allowable deviation in y-coordinates for horizontal alignment.
        output_filename (str): The name of the output DXF file with green lines added.
        beam_info_df (pd.DataFrame): DataFrame to store information about created beams.

    Returns:
        int: Updated beam count after labeling all detected pairs of parallel lines.
    """
    try:
        # Read the DXF file
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None

    msp = doc.modelspace()

    # Extract boxes and calculate their center points
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

    # Initialize list to store aligned box data
    alignment_data = []

    # Iterate over C3_ver_col from the second to the second-last row
    for i in range(1, len(C3_ver_col) - 1):
        current_label = C3_ver_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue  # Skip if label is not in centers

        current_center = centers[current_label]
        aligned_boxes = []

        # Check horizontally aligned boxes within the specified range and tolerance
        for label, center in centers.items():
            if label == current_label:
                continue  # Skip comparing the box with itself

            # Check if aligned within the x-range and y-tolerance
            if abs(center[1] - current_center[1]) <= tolerance and 24 < center[0] <= max_along_x - 24:
                aligned_boxes.append((label, center[0]))  # Store label and x-coordinate for sorting

        # Sort aligned boxes by x-coordinate for right-to-left order
        aligned_labels = [label for label, _ in sorted(aligned_boxes, key=lambda x: x[1], reverse=True)]

        # Append to alignment_data list
        alignment_data.append({
            'Boundary column': current_label,
            'Horizontal aligned column': aligned_labels
        })

    # Convert to DataFrame
    Boundary_3_connection = pd.DataFrame(alignment_data)

    # Detect a single horizontal line between selected box pairs and search for a parallel line
    for index, row in Boundary_3_connection.iterrows():
        boundary_box = row['Boundary column']
        if row['Horizontal aligned column']:
            target_box = row['Horizontal aligned column'][0]  # Get the first aligned box

            if boundary_box in centers:
                midpoint_y = centers[boundary_box][1]  # Y coordinate of the boundary box's midpoint
                line_found = False
                first_line_y = None
                first_line_x_min = None
                first_line_x_max = None

                # Search for a single horizontal line that matches criteria
                for entity in msp:
                    if entity.dxftype() == 'LINE':
                        x_start, y_start = entity.dxf.start.x, entity.dxf.start.y
                        x_end, y_end = entity.dxf.end.x, entity.dxf.end.y

                        # Ensure the line is horizontal and check with tolerance
                        if abs(y_start - y_end) <= 0.01:
                            x_min, x_max = min(x_start, x_end), max(x_start, x_end)
                            y_coord = y_start  # since y_start == y_end for a horizontal line

                            # Check if the Y is within tolerance of midpoint_y
                            if abs(y_coord - midpoint_y) <= 10:
                                if (abs(x_min - (max_along_x - 9)) <= 1 or abs(x_max - (max_along_x - 9)) <= 1):
                                    line_found = True
                                    first_line_y = y_coord
                                    first_line_x_min, first_line_x_max = x_min, x_max
        
                                    break

                # If a first line was detected, search for the second parallel line
                if line_found and first_line_y is not None:
                    parallel_line_found = False  # Initialize here to avoid UnboundLocalError
                    for entity in msp:
                        if entity.dxftype() == 'LINE':
                            x_start, y_start = entity.dxf.start.x, entity.dxf.start.y
                            x_end, y_end = entity.dxf.end.x, entity.dxf.end.y

                            # Ensure the line is horizontal
                            if abs(y_start - y_end) <= 0.01:
                                x_min, x_max = min(x_start, x_end), max(x_start, x_end)
                                y_coord = y_start  # since y_start == y_end for a horizontal line

                                # Check for parallel conditions: X around (max_along_x - 9) ±1, Y difference of 5 or 9 with ±1 tolerance
                                if (abs(x_start - (max_along_x - 9)) <= 1 or abs(x_end - (max_along_x - 9)) <= 1) and \
                                   (abs(y_coord - first_line_y - 5) <= 1 or abs(y_coord - first_line_y - 9) <= 1):
                                    parallel_line_found = True
    

                                    # Draw the green parallel lines and add labels
                                    msp.add_line((first_line_x_min, first_line_y), (first_line_x_max, first_line_y), dxfattribs={'color': 3})
                                    msp.add_line((x_min, y_coord), (x_max, y_coord), dxfattribs={'color': 3})

                                    # Label the beam with the current beam count
                                    beam_label = f"B{beam_count}"
                                    label_x = (first_line_x_min + first_line_x_max) / 2
                                    label_y = (first_line_y + y_coord) / 2

                                    # Create the text label with the desired attributes and position
                                    text_entity = msp.add_text(beam_label, dxfattribs={'color': 2, 'height': 0.5})
                                    text_entity.dxf.insert = (label_x, label_y + 7)  # Apply the offset to y-coordinate
                                    text_entity.dxf.rotation = 90

                                    # Increment the beam count for the next pair
                                    beam_count += 1

                                    # Add beam information to the DataFrame
                                    length = x_max - x_min  # Calculate the length of the beam (horizontal line)
                                    beam_info_df = pd.concat([beam_info_df, pd.DataFrame([{
                                        'beam names': beam_label,
                                        'length': length
                                    }])], ignore_index=True)

                                    break

                if not parallel_line_found:
                    pass
                
    # Save the modified DXF with added green lines and labels
    doc.saveas(output_filename)
    
    
    # Return the updated beam_count and the updated beam_info_df
    return beam_count, beam_info_df


# %%
import pandas as pd
from math import sqrt
import ezdxf

def check_vertical_alignment_boundary_4(filename, C4_hor_col, max_along_y, beam_count, beam_info_df, tolerance=15, output_filename=None):
    """
    Check vertically aligned boxes from the second row to the second-last row in C4_hor_col.
    Detect vertical lines between aligned boxes and label the beams if parallel lines are found.

    Parameters:
        filename (str): The path to the input DXF file.
        C4_hor_col (pd.DataFrame): DataFrame containing box coordinates and labels.
        max_along_y (float): The maximum Y value to calculate the Y-condition for parallel lines.
        beam_count (int): Initial count for labeling beams.
        beam_info_df (pd.DataFrame): DataFrame containing beam information (names and lengths).
        tolerance (int or float): The allowable deviation in X-coordinates for vertical alignment.
        output_filename (str): The filename to save the modified DXF file.

    Returns:
        tuple: Updated beam count and the updated `beam_info_df` with new beam information.
    """
    try:
        doc = ezdxf.readfile(filename)
    except IOError:
        print(f"Cannot open file: {filename}")
        return None, None
    except ezdxf.DXFStructureError:
        print(f"Invalid or corrupt DXF file: {filename}")
        return None, None

    msp = doc.modelspace()
    
    # Step 1: Extract centers from DXF data
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
    
    # Step 2: Iterate over C4_hor_col and detect vertical lines
    for i in range(1, len(C4_hor_col) - 1):
        current_label = C4_hor_col.iloc[i]['connected columns']
        if current_label not in centers:
            continue  # Skip if label is not in centers

        current_center = centers[current_label]
        aligned_boxes = []

        # Check for vertically aligned boxes based on X-coordinate within tolerance
        for label, center in centers.items():
            if label != current_label and abs(center[0] - current_center[0]) <= tolerance:
                # Calculate distance between current box and aligned box
                distance = sqrt((center[0] - current_center[0])**2 + (center[1] - current_center[1])**2)
                aligned_boxes.append((label, center[1], distance))

        if aligned_boxes:
            # Sort aligned boxes by distance (near to far) and pick the nearest one
            aligned_boxes_sorted = sorted(aligned_boxes, key=lambda x: x[2])  # Sort by distance

            # Use the nearest aligned box for detecting and drawing vertical lines
            target_label, target_y, _ = aligned_boxes_sorted[0]
            if current_label in centers:
                midpoint_x = centers[current_label][0]
                first_line_x = midpoint_x

                # Step 3: Determine Y-coordinates to draw lines based only on box Y-ranges
                min_y = min(current_center[1], target_y)
                max_y = max(current_center[1], target_y)

                # Step 4: Check for the existence of vertical parallel lines
                parallel_lines = []
                for entity in msp:
                    if entity.dxftype() == 'LINE':
                        if abs(entity.dxf.start.x - entity.dxf.end.x) <= tolerance:
                            # Check if line is vertical
                            if abs(entity.dxf.start.x - first_line_x) <= tolerance:
                                parallel_lines.append(entity)

                # If at least one parallel line exists, continue drawing
                if len(parallel_lines) == 1:
                    # Find the second parallel line and check for matching criteria
                    second_line_x = first_line_x + 5  # Example, shift by 5 units
                    parallel_line_found = False
                    for entity in msp:
                        if entity.dxftype() == 'LINE':
                            if abs(entity.dxf.start.x - second_line_x) <= tolerance and abs(entity.dxf.end.x - second_line_x) <= tolerance:
                                # Check horizontal condition
                                if (abs(entity.dxf.start.y - 0) <= 1 or abs(entity.dxf.start.y - 9) <= 1 or
                                    abs(entity.dxf.end.y - 0) <= 1 or abs(entity.dxf.end.y - 9) <= 1):
                                    parallel_line_found = True
                                    break
                    
                    if parallel_line_found:
                        # Draw two green parallel vertical lines connecting boxes
                        msp.add_line((first_line_x, min_y), (first_line_x, max_y), dxfattribs={'color': 3})
                        msp.add_line((first_line_x + 5, min_y), (first_line_x + 5, max_y), dxfattribs={'color': 3})
                        
                        # Label the beam with 90-degree rotation at calculated midpoint
                        beam_label = f"B{beam_count}"
                        label_offset_x = 15  # Adjust this value as needed for the leftward shift
                        midpoint_y = (min_y + max_y) / 2  # Calculated midpoint Y for label position
                        text_entity = msp.add_text(beam_label, dxfattribs={
                            'color': 2,  # Optional: specify a different color for the text
                            'height': 0.5  # Adjust text size as needed
                        })
                        text_entity.dxf.insert = (midpoint_x - label_offset_x, midpoint_y)
                        text_entity.dxf.rotation = 90
                        
                        # Update beam_info_df with the new beam label and length using pd.concat
                        beam_length = max_y - min_y  # Calculate the length of the beam
                        new_row = pd.DataFrame([[beam_label, beam_length]], columns=['beam_names', 'length'])
                        beam_info_df = pd.concat([beam_info_df, new_row], ignore_index=True)
                        
                        # Increment beam count for the next beam
                        beam_count += 1
    
    # Save the modified DXF file with added lines and labels
    if output_filename:
        doc.saveas(output_filename)
    
    return beam_count, beam_info_df


# %% [markdown]
# # Main function 

# %%
def pipeline_main(input_file, output_filename):
    """
    Main pipeline function to process a DXF file through multiple steps.
    
    Parameters:
    (1) input_file: Path to the input DXF file.
    (2) output_file: Path to save the modified DXF file after processing.
    """

    # Define the number of intermediate files
    num_intermediates = 22
    intermediate_files = [f'SMH_Single_Floor_DXF1.{i}.dxf' for i in range(num_intermediates + 1)]

    # Step 1: adjust dxf coordinates to zero
    adjust_dxf_coordinates_to00(input_file, intermediate_files[0])
    
    # Step 2: Convert DXF to DataFrame
    df = Dxf_to_DF(intermediate_files[0])
    
    # Step 3: Calculate max_along_x and max_along_y
    max_along_x, max_along_y = calculate_max_along_x_y(df)
    
    # Step 4-28: Pipeline steps using intermediate_files[i]
    four_corners(intermediate_files[0], intermediate_files[1], max_along_x, max_along_y, width=9, height=12)
    Boundary_1(intermediate_files[1], intermediate_files[2], target_x=9, tolerance=1, width=9, height=12, max_along_y=max_along_y)
    Boundary_2(intermediate_files[2], intermediate_files[3], width=12, height=9, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    Boundary_3(intermediate_files[3], intermediate_files[4], width=9, height=12, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    Boundary_4(intermediate_files[4], intermediate_files[5], width=12, height=9, tolerance=1, max_along_x=max_along_x, max_along_y=max_along_y)
    process_single_walls_left(df, max_along_x, max_along_y, intermediate_files[5], intermediate_files[6], width=12, height=9)
    process_single_walls_right(df, max_along_x, max_along_y, intermediate_files[6], intermediate_files[7], width=12, height=9)
    single_wall_up(df, max_along_x, max_along_y, intermediate_files[7], intermediate_files[8])

    pipeline_results = semi_main_columns(df, max_along_x, max_along_y)
    df_x_equal = pipeline_results['df_x_equal']
    df_y_equal = pipeline_results['df_y_equal']
    df_other = pipeline_results['df_other']
    temp_h = pipeline_results['temp_h']
    temp_v = pipeline_results['temp_v']

    create_boxes_in_df_x_equal(df_x_equal, temp_v, temp_h, intermediate_files[8], intermediate_files[9], width=9, height=12, tolerance_v=0.5, tolerance_h=0.5, tolerance_y=0.1, buffer=24, radius=84)
    create_boxes_in_df_y_equal(intermediate_files[9], intermediate_files[10], df_y_equal, temp_h, temp_v, tolerance_h=0.5, tolerance_v=0.5, width=12, height=9,radius=84)
    
    df_other_groupA, df_other_groupB = group_by_x(df_other)
    create_boxes_in_df_other_groupA(df_other_groupA, width=12, height=9, input_dxf_file=intermediate_files[10], output_dxf_file=intermediate_files[11])
    create_boxes_in_df_other_groupB(df_other_groupB, width=12, height=9, input_dxf_file=intermediate_files[11], output_dxf_file=intermediate_files[12])

    detect_and_label_boxes(intermediate_files[12], intermediate_files[13], label_position='right', offset=1, text_height=2, text_color=3, shift=0.5)
    
    column_info_df = create_column_schedule_dataframe(intermediate_files[13], max_along_x, max_along_y)
    change_line_color_to_light_gray(intermediate_files[13], intermediate_files[14])
    C1_ver_col, C2_hor_col, C3_ver_col, C4_hor_col = extract_columns(intermediate_files[14])

    beam_count = 0
    beam_count, beam_info_df = connect_edges_vertically_boundary_1(intermediate_files[14], intermediate_files[15], C1_ver_col, beam_count)
    
    beam_count, beam_info_df = connect_edges_horizontally_boundary_2(intermediate_files[15], intermediate_files[16], C2_hor_col, beam_count, beam_info_df)
    beam_count, beam_info_df = connect_edges_vertically_boundary_3(intermediate_files[16], intermediate_files[17], C3_ver_col, beam_count, beam_info_df)
    
    beam_count, beam_info_df = check_horizontal_alignment_boundary_1(
        filename=intermediate_files[17],      
        max_along_x=max_along_x,              
        C1_ver_col=C1_ver_col,         
        beam_count=beam_count,       
        tolerance=15,                  
        output_filename=intermediate_files[18],  
        beam_info_df=beam_info_df
    )
    beam_count, beam_info_df = check_vertical_alignment_boundary_2(
        filename=intermediate_files[18],
        C2_hor_col=C2_hor_col,
        max_along_y=max_along_y, 
        beam_count=beam_count, 
        tolerance=15, 
        output_filename=intermediate_files[19], 
        beam_info_df=beam_info_df
    )
    beam_count, beam_info_df = check_horizontal_alignment_boundary_3(
        filename=intermediate_files[19], 
        max_along_x=max_along_x, 
        C3_ver_col=C3_ver_col, 
        beam_count=beam_count, 
        output_filename=intermediate_files[20], 
        beam_info_df=beam_info_df, 
        tolerance=15
    )
    beam_count, beam_info_df = check_vertical_alignment_boundary_4(
        filename= intermediate_files[20], 
        C4_hor_col = C4_hor_col, 
        max_along_y = max_along_y, 
        beam_count=beam_count, 
        beam_info_df=beam_info_df, 
        tolerance=15, 
        output_filename=output_filename
    )
    
    print(f"Processing complete. The output file is saved as {output_filename}")
    
    # Return both output_filename and column_info_df
    return column_info_df, beam_info_df



# %%
