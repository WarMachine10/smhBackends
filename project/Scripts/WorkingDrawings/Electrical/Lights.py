
import ezdxf
import pandas as pd
from shapely.geometry import Polygon, LineString, JOIN_STYLE
import matplotlib.pyplot as plt
import math

# %%
import ezdxf
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import Point
import pandas as pd
import math

import sys
import os
import ezdxf
import matplotlib.pyplot as plt
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend


# Function to calculate the Euclidean distance between two points
def calculate_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate the length of a polyline
def calculate_length(coords):
    length = 0.0
    for i in range(1, len(coords)):
        length += calculate_distance(coords[i - 1], coords[i])
    return length

# Convert DXF file into a DataFrame with coordinates and lengths
def dxf_to_dataframe(dxf_file):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()
    polyline_data = []

    for entity in msp.query('LWPOLYLINE POLYLINE'):
        coords = [(point[0], point[1]) for point in entity.get_points()]
        length = calculate_length(coords)
        polyline_data.append({
            'Layer': entity.dxf.layer,
            'Coordinates': coords,
            'Length': length,
            'Closed': entity.is_closed
        })

    df = pd.DataFrame(polyline_data)
    return df

# Preprocess polylines into structured columns for vertices and lengths
def preprocess_polylines(df):
    processed_data = []

    for index, row in df.iterrows():
        data = {'layer': row['Layer']}
        coords = row['Coordinates']
        
        for i, (x, y) in enumerate(coords):
            data[f'vertex_{i+1}_x_coordinate'] = x
            data[f'vertex_{i+1}_y_coordinate'] = y
            if i < len(coords) - 1:
                next_x, next_y = coords[i + 1]
                data[f'length{i+1}'] = calculate_distance((x, y), (next_x, next_y))
        
        processed_data.append(data)
        
    processed_df = pd.DataFrame(processed_data)
    return processed_df


def calculate_distance(point1, point2):
    """ Calculate Euclidean distance between two points. """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def create_offset(df, offset_distance=2):
    offset_entities = []
    offsets_data = []

    def generate_offset(polyline_coords, offset_distance, layer):
        polygon = Polygon(polyline_coords)
        if polygon.is_valid:
            # If it's a complex shape, create the offset polygon
            offset_polygon = polygon.buffer(-offset_distance, resolution=16, join_style=JOIN_STYLE.mitre)
            if offset_polygon.is_valid:
                return offset_polygon
            else:
                # If offset creates invalid geometry, return original polygon
                return polygon
        return None

    def break_into_rectangles(polygon):
        """ Break a complex polygon (more than 4 sides) into simpler boxes/rectangles """
        if isinstance(polygon, MultiPolygon):
            return [p for p in polygon]
        
        # If polygon has more than 4 points, approximate by breaking into rectangles
        if len(polygon.exterior.coords) > 5:
            return polygon.minimum_rotated_rectangle
        else:
            return polygon

    def move_side_lines_away(polygon, side_lines, offset_distance, direction='right'):
        """
        Move side lines away from the original polyline if they are too close (either parallel or perpendicular).
        If direction is 'left', move left. If direction is 'right', move right.
        """
        new_side_lines = []
        for line in side_lines:
            # Check the distance between the line and the polygon
            if polygon.distance(line) < offset_distance:
                # Move the line away from the original polygon
                # Use the parallel_offset method to move the line
                new_line = line.parallel_offset(offset_distance, side=direction)
                new_side_lines.append(new_line)
            else:
                # If line is already far enough, leave it as is
                new_side_lines.append(line)
        return new_side_lines

    for index, row in df.iterrows():
        if row['Layer'] == 'Staircase_outerwall':
            continue
        
        coords = row['Coordinates']
        layer = row['Layer']
        polygon = Polygon(coords)
        bounding_box = polygon.bounds
        horizontal_length = bounding_box[2] - bounding_box[0]
        vertical_length = bounding_box[3] - bounding_box[1]
        
        if horizontal_length > offset_distance * 2 and vertical_length > offset_distance * 2:
            offset_polygon = generate_offset(coords, offset_distance, layer)
            if offset_polygon:
                # If the offset polygon is a complex shape (more than 4 sides), break it into rectangles
                if len(offset_polygon.exterior.coords) > 4:
                    offset_polygon = break_into_rectangles(offset_polygon)
                
                if isinstance(offset_polygon, MultiPolygon):
                    for p in offset_polygon:
                        # Get the side lines of the offset polygon
                        side_lines = [LineString([p.exterior.coords[i], p.exterior.coords[i+1]]) for i in range(len(p.exterior.coords)-1)]
                        
                        # Move side lines away if they are too close to the original polygon
                        side_lines = move_side_lines_away(polygon, side_lines, offset_distance, direction='right')  # You can change to 'left' if needed
                        
                        offset_entities.append((p, layer))
                        offset_data = {'layer': layer, 'Type': 'Offset'}
                        offset_coords = list(p.exterior.coords)
                        for i, (x, y) in enumerate(offset_coords):
                            offset_data[f'vertex_{i+1}_x_coordinate'] = x
                            offset_data[f'vertex_{i+1}_y_coordinate'] = y
                            if i < len(offset_coords) - 1:
                                next_x, next_y = offset_coords[i + 1]
                                offset_data[f'length{i+1}'] = calculate_distance((x, y), (next_x, next_y))
                        offsets_data.append(offset_data)
                else:
                    # For simple polygon (not MultiPolygon), create side lines and move if needed
                    side_lines = [LineString([offset_polygon.exterior.coords[i], offset_polygon.exterior.coords[i+1]]) for i in range(len(offset_polygon.exterior.coords)-1)]
                    side_lines = move_side_lines_away(polygon, side_lines, offset_distance, direction='right')  # Adjust direction as necessary
                    
                    offset_entities.append((offset_polygon, layer))
                    offset_data = {'layer': layer, 'Type': 'Offset'}
                    offset_coords = list(offset_polygon.exterior.coords)
                    for i, (x, y) in enumerate(offset_coords):
                        offset_data[f'vertex_{i+1}_x_coordinate'] = x
                        offset_data[f'vertex_{i+1}_y_coordinate'] = y
                        if i < len(offset_coords) - 1:
                            next_x, next_y = offset_coords[i + 1]
                            offset_data[f'length{i+1}'] = calculate_distance((x, y), (next_x, next_y))
                    offsets_data.append(offset_data)
        else:
            # If the shape is simple (bounding box), create a line at its center
            mid_point = polygon.centroid
            if horizontal_length <= offset_distance * 2:
                line = LineString([(mid_point.x, bounding_box[1] + offset_distance), 
                                   (mid_point.x, bounding_box[3] - offset_distance)])
            elif vertical_length <= offset_distance * 2:
                line = LineString([(bounding_box[0] + offset_distance, mid_point.y),
                                   (bounding_box[2] - offset_distance, mid_point.y)])
            offset_entities.append((line, layer))  # Store the line with its layer
            line_coords = list(line.coords)
            offset_data = {'layer': layer, 'Type': 'Single Line'}
            for i, (x, y) in enumerate(line_coords):
                offset_data[f'vertex_{i+1}_x_coordinate'] = x
                offset_data[f'vertex_{i+1}_y_coordinate'] = y
                if i < len(line_coords) - 1:
                    next_x, next_y = line_coords[i + 1]
                    offset_data[f'length{i+1}'] = calculate_distance((x, y), (next_x, next_y))
            offsets_data.append(offset_data)

    offsets_df = pd.DataFrame(offsets_data)
    return offset_entities, offsets_df

def save_offsets_to_dxf(offset_entities, doc):
    msp = doc.modelspace()

    for offset_entity, layer in offset_entities:  # Unpack the entity and its layer
        if isinstance(offset_entity, Polygon):
            coords = list(offset_entity.exterior.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            polyline = msp.add_lwpolyline(coords)
            polyline.dxf.layer = layer  # Set the layer to the original layer
        elif isinstance(offset_entity, LineString):
            line = msp.add_line(offset_entity.coords[0], offset_entity.coords[1])
            line.dxf.layer = layer  # Set the layer to the original layer
            line.dxf.color = 5  # Set line color to blue for single lines

    doc.saveas('offset_dxf.dxf')

# Plot DXF file contents with offsets
def plot_dxf(dxf_file):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()
    fig, ax = plt.subplots(figsize=(18, 15))

    for entity in msp.query('LWPOLYLINE'):
        points = entity.get_points()
        x, y = zip(*[(point[0], point[1]) for point in points])
        ax.plot(x, y, color='blue' if entity.dxf.color == 5 else 'black')

    for entity in msp.query('POLYLINE'):
        points = entity.get_points()
        x, y = zip(*[(point[0], point[1]) for point in points])
        ax.plot(x, y, color='green')

    for entity in msp.query('LINE'):
        x, y = zip((entity.dxf.start.x, entity.dxf.start.y), (entity.dxf.end.x, entity.dxf.end.y))
        ax.plot(x, y, color='blue' if entity.dxf.color == 5 else 'red')

    ax.set_aspect('equal', 'box')
    ax.set_title('DXF Plot')
    plt.show()

# Main function to run the process
def main(dxf_file, offset_distance=2):
    # Step 1: Convert DXF to DataFrame
    df = dxf_to_dataframe(dxf_file)
    
    # Preprocess the DataFrame to add vertices and lengths
    processed_df = preprocess_polylines(df)
    
    # Step 2: Create offsets with structured columns
    offsets, offsets_df = create_offset(df, offset_distance)
    
    # Step 3: Save offsets to a new DXF file
    doc = ezdxf.readfile(dxf_file)
    save_offsets_to_dxf(offsets, doc)

# Specify the DXF file and offset distance
dxf_file = 'Test6.dxf'  # Update with your DXF file path
main(dxf_file, offset_distance=24)


# First, create or load the necessary dataframes
df = dxf_to_dataframe(dxf_file)  # Assuming df is created from the DXF file

# Preprocess the polylines to get the processed dataframe
processed_df = preprocess_polylines(df)

# Generate offsets and their dataframe
offsets, offsets_df = create_offset(df, offset_distance=24)

import pandas as pd

def calculate_polygon_area(row):
    # Extract the coordinates dynamically based on the column pattern
    coordinates = []
    for i in range(1, 10):  # Since you have vertex_1_x to vertex_9_x
        x_coord = row.get(f'vertex_{i}_x_coordinate', None)
        y_coord = row.get(f'vertex_{i}_y_coordinate', None)
        
        if pd.notna(x_coord) and pd.notna(y_coord):
            coordinates.append((x_coord, y_coord))

    if len(coordinates) < 3:  # Need at least 3 vertices for a polygon
        return 0

    # Apply Shoelace Theorem to calculate the area of the polygon
    n = len(coordinates)
    area = 0
    for i in range(n):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2.0
    
    # Convert area to square feet (if coordinates are in inches)
    area_in_sqfeet = area / 144  # If coordinates are in inches, divide by 144 to convert to square feet
    
    return area_in_sqfeet

# Assuming your dataframe is named 'processed_df'
processed_df['Carpet Area'] = processed_df.apply(calculate_polygon_area, axis=1)

#print(processed_df[['layer', 'Carpet Area']])


def calculate_bulbs_required(row):
    carpet_area = row['Carpet Area']
    lux_requirement = 500  # average lux requirement for each space
    lumens_per_bulb = 1300  # lumens for each 15W LED bulb

    # Calculate required lumens based on carpet area
    required_lumens = carpet_area * lux_requirement * 0.092903

    # Calculate number of bulbs required, rounding up to ensure enough light
    bulbs_required = required_lumens / lumens_per_bulb
    return int(bulbs_required) if bulbs_required.is_integer() else int(bulbs_required) + 1

# Apply the function to processed_df and create a new column for bulbs required
processed_df['Bulbs Required'] = processed_df.apply(calculate_bulbs_required, axis=1)

# Display the layer, carpet area, and bulbs required
#print(processed_df[['layer', 'Carpet Area', 'Bulbs Required']])

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to calculate midpoint of two points
def midpoint(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Function to plot the offsets with vertices, midpoints, and lengths in feet
def plot_offsets_with_points(offsets_df, min_distance=0.1):
    fig, ax = plt.subplots(figsize=(18, 15))
    point_counter = 1  # Counter for numbering the points
    labels_data = []  # List to store label data for DataFrame
    
    # Function to check if a new label overlaps with existing labels
    def is_overlapping(new_x, new_y):
        for _, existing_x, existing_y, _ in labels_data:
            if euclidean_distance(new_x, new_y, existing_x, existing_y) < min_distance:
                return True
        return False

    # Iterate through each row in the dataframe
    for idx, row in offsets_df.iterrows():
        if row['Type'] == 'Offset':
            # Collect coordinates and lengths for vertices dynamically
            x_coords = []
            y_coords = []
            lengths = []
            i = 1
            while f'vertex_{i}_x_coordinate' in row and not pd.isna(row[f'vertex_{i}_x_coordinate']):
                x_coords.append(row[f'vertex_{i}_x_coordinate'])
                y_coords.append(row[f'vertex_{i}_y_coordinate'])
                if f'length{i}' in row and not pd.isna(row[f'length{i}']):
                    lengths.append(row[f'length{i}'] / 12)  # Convert from inches to feet
                i += 1
            
            # Plot the polyline
            ax.plot(x_coords, y_coords, label=row['layer'], color='blue', lw=2)

            # Check if the shape is a closed 4-sided shape with narrow sides
            if len(lengths) == 4:
                narrow_sides = [length < 2 for length in lengths]
                if sum(narrow_sides) == 2 and narrow_sides.count(True) == 2:
                    narrow_indices = [i for i, is_narrow in enumerate(narrow_sides) if is_narrow]
                    midpoints = []
                    for idx in narrow_indices:
                        x1, y1 = x_coords[idx], y_coords[idx]
                        x2, y2 = x_coords[(idx + 1) % 4], y_coords[(idx + 1) % 4]
                        mx, my = midpoint(x1, y1, x2, y2)
                        if not is_overlapping(mx, my):
                            ax.text(mx, my, str(point_counter), color='green', fontsize=10, ha='center')
                            labels_data.append((point_counter, mx, my, 'midpoint'))
                            point_counter += 1
                        midpoints.append((mx, my))
                    
                    # Label imaginary midpoint if midpoints are more than 6 feet apart
                    if euclidean_distance(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1]) > 6:
                        imx, imy = midpoint(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1])
                        if not is_overlapping(imx, imy):
                            ax.text(imx, imy, str(point_counter), color='orange', fontsize=10, ha='center')
                            labels_data.append((point_counter, imx, imy, 'imaginary midpoint'))
                            point_counter += 1
                    continue
            
            # Label each edge and midpoint based on the updated requirements
            for i in range(len(lengths)):
                x1, y1 = x_coords[i], y_coords[i]
                x2, y2 = x_coords[(i + 1) % len(x_coords)], y_coords[(i + 1) % len(y_coords)]
                length_ft = lengths[i]
                mx, my = midpoint(x1, y1, x2, y2)
                
                # Label the starting vertex of each segment
                if not is_overlapping(x1, y1):
                    ax.text(x1, y1, str(point_counter), color='red', fontsize=10, ha='center')
                    labels_data.append((point_counter, x1, y1, 'vertex'))
                    point_counter += 1
                
                # Label midpoint if segment length > 6 feet
                if length_ft > 6 and not is_overlapping(mx, my):
                    ax.text(mx, my, str(point_counter), color='green', fontsize=10, ha='center')
                    labels_data.append((point_counter, mx, my, 'midpoint'))
                    point_counter += 1

        elif row['Type'] == 'Single Line':
            # For single line, use vertex_1 and vertex_2 coordinates and length1
            x1, y1 = row['vertex_1_x_coordinate'], row['vertex_1_y_coordinate']
            x2, y2 = row['vertex_2_x_coordinate'], row['vertex_2_y_coordinate']
            length_ft = row['length1'] / 12  # Convert from inches to feet
            mx, my = midpoint(x1, y1, x2, y2)
            
            # Plot the single line
            ax.plot([x1, x2], [y1, y2], label=row['layer'], color='red', lw=2)
            
            # Display length on the plot
            ax.text(mx, my, f"{length_ft:.2f} ft", color='purple', fontsize=8, ha='center')

            # Label edge points and midpoint for single lines based on length
            if length_ft < 1.5:
                if not is_overlapping(mx, my):
                    ax.text(mx, my, str(point_counter), color='green', fontsize=10, ha='center')
                    labels_data.append((point_counter, mx, my, 'midpoint'))
                    point_counter += 1
            else:
                if not is_overlapping(x1, y1):
                    ax.text(x1, y1, str(point_counter), color='red', fontsize=10, ha='center')
                    labels_data.append((point_counter, x1, y1, 'vertex'))
                    point_counter += 1
                if not is_overlapping(x2, y2):
                    ax.text(x2, y2, str(point_counter), color='red', fontsize=10, ha='center')
                    labels_data.append((point_counter, x2, y2, 'vertex'))
                    point_counter += 1

                # Mark midpoint if length > 6 feet
                if length_ft > 6 and not is_overlapping(mx, my):
                    ax.text(mx, my, str(point_counter), color='green', fontsize=10, ha='center')
                    labels_data.append((point_counter, mx, my, 'midpoint'))
                    point_counter += 1

    # Convert labels data to DataFrame
    labels_df = pd.DataFrame(labels_data, columns=['Label', 'X_Coordinate', 'Y_Coordinate', 'Point_Type'])
    
    # Adding labels and title
    ax.set_title('Offset Polylines with Marked Points and Lengths (Non-overlapping Labels)', fontsize=16)
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.legend()

    plt.grid(True)
    plt.show()
    
    return labels_df  # Return the DataFrame containing non-overlapping label coordinates

# Example usage with your offsets_df
labels_df = plot_offsets_with_points(offsets_df)

# Display the cleaned labels DataFrame
#print(labels_df)


# %%


# %%
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Load the source DXF (bulbs) and target DXF (lines)
source_dxf = ezdxf.readfile("15W.dxf")
target_dxf = ezdxf.readfile("Test6.dxf")  # Replace with the actual target file name

# Define the block name for bulbs
bulb_block_name = "15w"

# Ensure the "15w" layer exists in the target DXF
if "15w" not in target_dxf.layers:
    target_dxf.layers.new(name="15w", dxfattribs={'color': 7})  # Adjust color if needed

# Check if the bulb block exists in the source DXF
if bulb_block_name in source_dxf.blocks:
    # Copy the block definition to the target DXF if it doesn't exist
    if bulb_block_name not in target_dxf.blocks:
        target_dxf.blocks.new(name=bulb_block_name)
        for entity in source_dxf.blocks[bulb_block_name]:
            target_dxf.blocks[bulb_block_name].add_entity(entity.copy())
else:
    print(f"Block '{bulb_block_name}' not found in the source DXF.")

# Insert bulbs at each position in labels_df (assumes labels_df is already available)
target_msp = target_dxf.modelspace()
for index, row in labels_df.iterrows():
    x, y = row['X_Coordinate'], row['Y_Coordinate']  # Use correct column names
    bulb_ref = target_msp.add_blockref(bulb_block_name, insert=(x, y))
    bulb_ref.dxf.layer = "15w"  # Explicitly set the layer to "15w"

# Now, let's load the switch block from another DXF file and place switches
sw_doc = ezdxf.readfile("SW.dxf")
if "SW" not in sw_doc.blocks:
    raise ValueError("The block 'SW' was not found in the SW.dxf file.")
sw_block = sw_doc.blocks.get("SW")

# Ensure the switch block definition exists in the main document
if "SW" not in target_dxf.blocks:
    target_dxf.blocks.new(name="SW")
    for entity in sw_block:
        target_dxf.blocks["SW"].add_entity(entity.copy())

# Distance thresholds in inches
distances = [5 / 12, 9 / 12]  # Convert inches to feet for consistency

# Track layers where a switch has been placed
placed_layers = set()

# Helper functions for parallelism, rotation, and position adjustment
def are_parallel(line1, line2, distance):
    if abs(line1[0][0] - line1[1][0]) < 1e-6 and abs(line2[0][0] - line2[1][0]) < 1e-6:
        return abs(line1[0][0] - line2[0][0]) <= distance
    elif abs(line1[0][1] - line1[1][1]) < 1e-6 and abs(line2[0][1] - line2[1][1]) < 1e-6:
        return abs(line1[0][1] - line2[0][1]) <= distance
    return False

def calculate_rotation(line_start, line_end):
    if abs(line_start[1] - line_end[1]) < 1e-6:
        return 0  # Horizontal line
    elif abs(line_start[0] - line_end[0]) < 1e-6:
        return 90  # Vertical line
    return 0

def calculate_adjusted_position(layer_line, non_layer_line, midpoint):
    layer_mid_y = (layer_line[0][1] + layer_line[1][1]) / 2
    non_layer_mid_y = (non_layer_line[0][1] + non_layer_line[1][1]) / 2
    layer_mid_x = (layer_line[0][0] + layer_line[1][0]) / 2
    non_layer_mid_x = (non_layer_line[0][0] + non_layer_line[1][0]) / 2
    
    offset_distance = 0.5  # Adjust the distance as needed
    if abs(layer_mid_y - non_layer_mid_y) > abs(layer_mid_x - non_layer_mid_x):
        if non_layer_mid_y > layer_mid_y:
            return (midpoint[0], midpoint[1] - offset_distance)  # Place switch below
        else:
            return (midpoint[0], midpoint[1] + offset_distance)  # Place switch above
    else:
        if non_layer_mid_x > layer_mid_x:
            return (midpoint[0] - offset_distance, midpoint[1])  # Place switch to the left
        else:
            return (midpoint[0] + offset_distance, midpoint[1])  # Place switch to the right

# Loop through each polyline entity to place switches
for polyline in target_msp.query("POLYLINE LWPOLYLINE"):
    polyline_layer = polyline.dxf.layer
    if polyline_layer in placed_layers:
        continue

    same_layer_lines = [line for line in target_msp.query("LINE") if line.dxf.layer == polyline_layer]
    other_layer_lines = [line for line in target_msp.query("LINE") if line.dxf.layer != polyline_layer]

    switch_placed = False
    for line in same_layer_lines:
        start1, end1 = (line.dxf.start.x, line.dxf.start.y), (line.dxf.end.x, line.dxf.end.y)
        
        for other_line in other_layer_lines:
            start2, end2 = (other_line.dxf.start.x, other_line.dxf.start.y), (other_line.dxf.end.x, other_line.dxf.end.y)
            
            if any(are_parallel((start1, end1), (start2, end2), d) for d in distances):
                midpoint_x = (start1[0] + end1[0]) / 2
                midpoint_y = (start1[1] + end1[1]) / 2
                midpoint = (midpoint_x, midpoint_y)
                
                # Calculate rotation and position
                rotation = calculate_rotation(start1, end1)
                adjusted_position = calculate_adjusted_position((start1, end1), (start2, end2), midpoint)

                # Place the switch with calculated position and rotation
                block_ref = target_msp.add_blockref("SW", adjusted_position)
                block_ref.dxf.layer = "SW"
                block_ref.dxf.rotation = rotation

                placed_layers.add(polyline_layer)
                switch_placed = True
                break
        if switch_placed:
            break

# Save the modified DXF file with both bulbs and switches
target_dxf.saveas("Electrical.dxf")
print("DXF file saved with bulbs and switches.")

# Plotting the updated DXF file with both bulbs and switches
fig, ax = plt.subplots(figsize=(20, 15))
ax.set_aspect('equal')
ax.set_title('DXF Plot with Bulbs and Switches')

# Plot the lines
for entity in target_msp.query("LINE"):
    x_coords = [entity.dxf.start.x, entity.dxf.end.x]
    y_coords = [entity.dxf.start.y, entity.dxf.end.y]
    ax.plot(x_coords, y_coords, color="black")

# Plot the bulbs (as blue circles) and switches (as red squares)
for block_ref in target_msp.query("INSERT"):
    if block_ref.dxf.name == bulb_block_name:
        mid_x, mid_y = block_ref.dxf.insert.x, block_ref.dxf.insert.y
        ax.plot(mid_x, mid_y, 'bo', markersize=8)  # Blue circles for bulbs
    elif block_ref.dxf.name == "SW":
        mid_x, mid_y = block_ref.dxf.insert.x, block_ref.dxf.insert.y
        ax.plot(mid_x, mid_y, 'rs', markersize=8)  # Red squares for switches

plt.show()



def extract_polylines_and_blocks(dxf_path):
    # Load the DXF file
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Prepare lists to hold polyline and block data
    polylines_data = []
    blocks_data = []

    # Extract POLYLINE entities
    for polyline in msp.query("LWPOLYLINE"):
        vertices = [(point[0], point[1]) for point in polyline]
        polygon = Polygon(vertices)
        polylines_data.append({
            "Entity": "Polyline",
            "Layer": polyline.dxf.layer,
            "Color": polyline.dxf.color,
            "Closed": polyline.closed,
            "Polygon": polygon  # Store the Polygon object for boundary checks
        })

    # Extract BLOCKS (INSERT entities)
    for block in msp.query("INSERT"):
        block_data = {
            "Entity": "Block",
            "Layer": block.dxf.layer,
            "Block Name": block.dxf.name,
            "Insert Point": (block.dxf.insert.x, block.dxf.insert.y),
            "Rotation": block.dxf.rotation,
            "Scale X": block.dxf.xscale,
            "Scale Y": block.dxf.yscale,
        }
        blocks_data.append(block_data)

    return pd.DataFrame(polylines_data), pd.DataFrame(blocks_data)

def extract_and_group_blocks(dxf_path, polyline_df, block_df, tolerance=0.01):
    # Group blocks by the polyline they fall within
    results = []
    for _, poly in polyline_df.iterrows():
        poly_polygon = poly["Polygon"]
        sw_block_coords = None
        light_15w_coords = []

        for _, block in block_df.iterrows():
            block_point = Point(block["Insert Point"])
            if block["Layer"] == "SW":
                nearest_point_on_polyline = nearest_points(poly_polygon, block_point)[0]
                if block_point.distance(nearest_point_on_polyline) <= tolerance:
                    sw_block_coords = block["Insert Point"]
            elif block["Layer"] == "15w":
                if block_point.within(poly_polygon):
                    light_15w_coords.append(block["Insert Point"])
        
        results.append({
            "Polyline Layer": poly["Layer"],
            "Polygon": poly_polygon,
            "SW Block Coordinates": sw_block_coords,
            "15w Block Coordinates": light_15w_coords,
            "Number of 15w Blocks": len(light_15w_coords)
        })

    return pd.DataFrame(results)

def create_bulb_to_switch_connections(dxf_path, output_path, df):
    # Load the original DXF file
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # Create a new layer for wires with green color and line weight of 3 mm
    if "Wires" not in doc.layers:
        doc.layers.new(name="Wires", dxfattribs={"color": 3, "lineweight": 30})

    def create_straight_line_within_boundary(start, end, boundary, layer_name):
        # Create a LineString from the start to end points
        line = LineString([start, end])

        # Check if the entire line is within the boundary polygon
        if boundary.contains(line):
            # Draw the line in DXF if it fits within the boundary
            msp.add_line(start, end, dxfattribs={"layer": layer_name})
            return True
        else:
            return False

    # Iterate through each row of the DataFrame to connect bulbs to switches
    for _, row in df.iterrows():
        sw_coords = row["SW Block Coordinates"]
        poly_boundary = row["Polygon"]
        
        # Only proceed if there's an SW block in the polyline
        if sw_coords is not None:
            for bulb_coords in row["15w Block Coordinates"]:
                # Attempt to create a straight line within the boundary
                create_straight_line_within_boundary(bulb_coords, sw_coords, poly_boundary, "Wires")

    # Save the modified DXF file
    doc.saveas(output_path)

def main(dxf_path, output_path):
    # Extract polylines and blocks data
    polyline_df, block_df = extract_polylines_and_blocks(dxf_path)
   # print(polyline_df)
   # print(block_df)

    # Group blocks by polylines
    grouped_df = extract_and_group_blocks(dxf_path, polyline_df, block_df)
    #print(grouped_df)

    # Create bulb-to-switch connections and save to new DXF
    create_bulb_to_switch_connections(dxf_path, output_path, grouped_df)
   # print(f"Connections saved in {output_path}")

# Path to your DXF file and output file
dxf_path = "Electrical.dxf"
output_path = "output_with_lines.dxf"
main(dxf_path, output_path)




try:
    dxf_file = 'Test3.dxf'  
    doc, auditor = recover.readfile(dxf_file)
except IOError:
    print(f'Not a DXF file or a generic I/O error.')
    sys.exit(1)
except ezdxf.DXFStructureError:
    print(f'Invalid or corrupted DXF file.')
    sys.exit(2)


if not auditor.has_errors:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    
   
    png_file = os.path.splitext(dxf_file)[0] + '.png'
    
    
    ezdxf.addons.drawing.matplotlib.qsave(layout=doc.modelspace(), 
                                          filename=png_file, 
                                          bg='#FFFFFF',  # White background color
                                          dpi=300)


img = plt.imread(png_file)


plt.figure(figsize=(50, 40))  


plt.imshow(img, alpha=1.0) 
plt.axis('off')  
plt.show()

# %%


# %%



