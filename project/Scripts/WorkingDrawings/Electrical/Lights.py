# %%
# Required Libraries:
# ezdxf==0.15.2 (for DXF file reading and manipulation)
# pandas==1.5.3 (for data manipulation in DataFrames)
# shapely==2.0.1 (for geometric operations on shapes and lines)
# matplotlib==3.7.2 (for plotting DXF file contents)
from shapely.geometry import LineString,Polygon
import os
from django.conf import settings
import ezdxf  # For reading and writing DXF files
from loguru import logger
import pandas as pd  # For organizing polyline data in a DataFrame
from shapely.geometry import Polygon, LineString, MultiPolygon, JOIN_STYLE  # For geometry manipulations
from shapely.ops import unary_union  # For geometry operations
import matplotlib.pyplot as plt  # For visualizing DXF contents
import math  # For calculating Euclidean distances
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    # Query all polyline entities (LWPOLYLINE and POLYLINE)
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
        # Add coordinates for each vertex and lengths between consecutive vertices
        for i, (x, y) in enumerate(coords):
            data[f'vertex_{i+1}_x_coordinate'] = x
            data[f'vertex_{i+1}_y_coordinate'] = y
            if i < len(coords) - 1:
                next_x, next_y = coords[i + 1]
                data[f'length{i+1}'] = calculate_distance((x, y), (next_x, next_y))
        
        processed_data.append(data)
        
    processed_df = pd.DataFrame(processed_data)
    
    # Ensure it's a DataFrame with at least one row, even if empty
    if processed_df.empty:
        processed_df = pd.DataFrame(columns=['layer'] + [f'vertex_{i+1}_x_coordinate' for i in range(len(coords))] + [f'vertex_{i+1}_y_coordinate' for i in range(len(coords))])
    
    return processed_df


from shapely.geometry import Point


def calculate_distance(point1, point2):
    """ Calculate Euclidean distance between two points. """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
# Create offset polygons/lines from input DataFrame containing polyline coordinates
def create_offset(df, offset_distance=4):
    offset_entities = []
    offsets_data = []
 # Generates offset geometry from a polyline (Polygon or LineString) for a specified distance
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
    # Splits complex polygons into multiple rectangles (or retains as single rectangle)
    def break_into_rectangles(polygon):
        """ Break a complex polygon (more than 4 sides) into simpler boxes/rectangles """
        if isinstance(polygon, MultiPolygon):
            return [p for p in polygon]
        
        # If polygon has more than 4 points, approximate by breaking into rectangles
        if len(polygon.exterior.coords) > 5:
            return polygon.minimum_rotated_rectangle
        else:
            return polygon
     # Moves side lines away if too close to original polyline
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
# Function to save offset entities (polygons/lines) into a DXF file
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

# %%
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
#labels_df = plot_offsets_with_points(offsets_df)

# Display the cleaned labels DataFrame
#print(labels_df)


# %%
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Helper function to ensure layers and blocks exist
def ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name):
    if "15w" not in target_dxf.layers:
        target_dxf.layers.new(name="15w", dxfattribs={'color': 7})
    
    if bulb_block_name in source_dxf.blocks and bulb_block_name not in target_dxf.blocks:
        target_dxf.blocks.new(name=bulb_block_name)
        for entity in source_dxf.blocks[bulb_block_name]:
            target_dxf.blocks[bulb_block_name].add_entity(entity.copy())

def ensure_switch_block(target_dxf, sw_doc):
    if "SW" not in sw_doc.blocks:
        raise ValueError("The block 'SW' was not found in the SW.dxf file.")
    sw_block = sw_doc.blocks.get("SW")
    
    if "SW" not in target_dxf.blocks:
        target_dxf.blocks.new(name="SW")
        for entity in sw_block:
            target_dxf.blocks["SW"].add_entity(entity.copy())

# Bulb placement function
def place_bulbs(target_msp, labels_df, bulb_block_name):
    for index, row in labels_df.iterrows():
        x, y = row['X_Coordinate'], row['Y_Coordinate']
        bulb_ref = target_msp.add_blockref(bulb_block_name, insert=(x, y))
        bulb_ref.dxf.layer = "15w"

# Switch placement helper functions
def are_parallel(line1, line2, distance):
    if abs(line1[0][0] - line1[1][0]) < 1e-6 and abs(line2[0][0] - line2[1][0]) < 1e-6:
        return abs(line1[0][0] - line2[0][0]) <= distance
    elif abs(line1[0][1] - line1[1][1]) < 1e-6 and abs(line2[0][1] - line2[1][1]) < 1e-6:
        return abs(line1[0][1] - line2[0][1]) <= distance
    return False

def calculate_rotation(line_start, line_end):
    if abs(line_start[1] - line_end[1]) < 1e-6:
        return 0
    elif abs(line_start[0] - line_end[0]) < 1e-6:
        return 90
    return 0

def calculate_adjusted_position(layer_line, non_layer_line, midpoint):
    layer_mid_y = (layer_line[0][1] + layer_line[1][1]) / 2
    non_layer_mid_y = (non_layer_line[0][1] + non_layer_line[1][1]) / 2
    layer_mid_x = (layer_line[0][0] + layer_line[1][0]) / 2
    non_layer_mid_x = (non_layer_line[0][0] + non_layer_line[1][0]) / 2
    
    offset_distance = 0.5
    if abs(layer_mid_y - non_layer_mid_y) > abs(layer_mid_x - non_layer_mid_x):
        return (midpoint[0], midpoint[1] - offset_distance) if non_layer_mid_y > layer_mid_y else (midpoint[0], midpoint[1] + offset_distance)
    else:
        return (midpoint[0] - offset_distance, midpoint[1]) if non_layer_mid_x > layer_mid_x else (midpoint[0] + offset_distance, midpoint[1])

# Switch placement function
def place_switches(target_msp, distances):
    placed_layers = set()
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
                    
                    rotation = calculate_rotation(start1, end1)
                    adjusted_position = calculate_adjusted_position((start1, end1), (start2, end2), midpoint)

                    block_ref = target_msp.add_blockref("SW", adjusted_position)
                    block_ref.dxf.layer = "SW"
                    block_ref.dxf.rotation = rotation

                    placed_layers.add(polyline_layer)
                    switch_placed = True
                    break
            if switch_placed:
                break

# Function to save and plot the updated DXF
def save_and_plot_dxf(target_dxf, target_msp, bulb_block_name):
    try:
        # Save the DXF file
        savePath=os.path.join(settings.BASE_DIR,'Temp','crap','Electrical.dxf')
        target_dxf.saveas(savePath)
        print("DXF file saved as 'Electrical.dxf'.")
        return target_dxf
    except Exception as e:
        print(f"Error saving DXF file: {e}")
        # return target_dxf

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


# %%
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
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

def extract_and_group_blocks(dxf_path, polyline_df, block_df):
    # Group blocks by the polyline they fall within
    results = []
    for _, poly in polyline_df.iterrows():
        poly_polygon = poly["Polygon"]
        sw_block_coords = None
        light_15w_coords = []

        for _, block in block_df.iterrows():
            block_point = Point(block["Insert Point"])

            # For "SW" blocks, snap to the polyline boundary if close enough
            if block["Layer"] == "SW":
                nearest_point_on_polyline = nearest_points(poly_polygon, block_point)[0]

                # Place the SW block exactly on the boundary if it's not already there
                if not block_point.equals(nearest_point_on_polyline):
                    # Snap the SW block to the nearest point on the polyline boundary
                    sw_block_coords = (nearest_point_on_polyline.x, nearest_point_on_polyline.y)
                else:
                    # Keep original insertion point if it's already on the boundary
                    sw_block_coords = block["Insert Point"]

            # For "15w" blocks, retain the original within check
            elif block["Layer"] == "15w" and block_point.within(poly_polygon):
                light_15w_coords.append(block["Insert Point"])

        # Store the results for this polyline
        results.append({
            "Polyline Layer": poly["Layer"],
            "Polygon": poly_polygon,
            "SW Block Coordinates": sw_block_coords,
            "15w Block Coordinates": light_15w_coords,
            "Number of 15w Blocks": len(light_15w_coords)
        })

    return pd.DataFrame(results)

def extract_and_group_blocks(dxf_path, polyline_df, block_df):
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
                if block_point.distance(nearest_point_on_polyline) <= 0.01:  # Tolerance distance check
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

    # Gather all SW switch coordinates in a list for finding nearest ones when needed
    all_switch_coords = [row["SW Block Coordinates"] for _, row in df.iterrows() if row["SW Block Coordinates"] is not None]

    # Iterate through each row of the DataFrame to connect bulbs to switches
    for _, row in df.iterrows():
        sw_coords = row["SW Block Coordinates"]
        bulbs_coords = row["15w Block Coordinates"]

        if sw_coords is not None:
            # Connect each bulb to the existing switch in this polyline
            for bulb_coords in bulbs_coords:
                msp.add_line(bulb_coords, sw_coords, dxfattribs={"layer": "Wires"})
        else:
            # If no SW switch is found, find the nearest SW switch from all other polylines
            if all_switch_coords:
                # Find the single nearest switch to all the bulbs in this polyline
                nearest_sw_coords = min(
                    all_switch_coords,
                    key=lambda sw: min([Point(bulb_coords).distance(Point(sw)) for bulb_coords in bulbs_coords])
                )
                
                # Connect all the bulbs in this polyline to the nearest switch
                for bulb_coords in bulbs_coords:
                    msp.add_line(bulb_coords, nearest_sw_coords, dxfattribs={"layer": "Wires"})

    # Save the modified DXF file
    doc.saveas(output_path)

    
def main_final(dxf_file, offset_distance=24):
    fifteenW = os.path.join(settings.BASE_DIR, 'assets', '15W.dxf')
    dynamicSW = os.path.join(settings.BASE_DIR, 'assets', 'SW.dxf')
    
    df = dxf_to_dataframe(dxf_file)
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    processed_df = preprocess_polylines(df)

    # Ensure "is_closed" column exists if needed
    if 'is_closed' not in processed_df.columns:
        processed_df['is_closed'] = processed_df.apply(
            lambda row: getattr(row.geometry, 'is_ring', False) if hasattr(row, 'geometry') else False, axis=1
        )

    # Offset creation without skipping
    offsets, offsets_df = [], pd.DataFrame()
    for _, row in processed_df.iterrows():
        offset, offset_df = create_offset(row, offset_distance)
        if isinstance(offset_df, pd.Series):
            offset_df = offset_df.to_frame().T
        offsets.append(offset)
        offsets_df = pd.concat([offsets_df, offset_df])

    labels_df = plot_offsets_with_points(offsets_df)


    # Load DXF documents and continue processing
    target_dxf = ezdxf.readfile(dxf_file)
    source_dxf = ezdxf.readfile(fifteenW)
    sw_doc = ezdxf.readfile(dynamicSW)
    bulb_block_name = "15w"
    target_msp = target_dxf.modelspace()

    save_and_plot_dxf(target_dxf, target_msp, bulb_block_name)

    # Define paths for output
    savePath = os.path.join(settings.BASE_DIR, 'Temp', 'crap', 'Electrical.dxf')
    dxf_path_electrical = savePath
    output_path_final = "Electrical_drawing.dxf"

    ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name)
    ensure_switch_block(target_dxf, sw_doc)

    # Place bulbs and switches
    place_bulbs(target_msp, labels_df, bulb_block_name)
    place_switches(target_msp, distances=[5 / 12, 9 / 12])

    # Process and create final output
    polyline_df, block_df = extract_polylines_and_blocks(dxf_path_electrical)
    grouped_df = extract_and_group_blocks(dxf_path_electrical, polyline_df, block_df)
    create_bulb_to_switch_connections(dxf_path_electrical, output_path_final, grouped_df)

    # Count block occurrences
    dwg = ezdxf.readfile(dxf_path_electrical)
    target_blocks = ["15w", "SW"]
    block_counts = {name: 0 for name in target_blocks}
    for entity in dwg.modelspace():
        if entity.dxftype() == "INSERT":
            block_name = entity.dxf.name
            if block_name in block_counts:
                block_counts[block_name] += 1

    for block, count in block_counts.items():
        print(f"{block}: {count} occurrences")
    
    # Load and match materials from Excel
    excel_file_path = os.path.join(settings.BASE_DIR, 'assets', 'price_of_electrical.xlsx')
    df = pd.read_excel(excel_file_path)
    df.columns = df.columns.str.strip()
    df['Material_Name_lower'] = df['Material Name'].str.lower()
    block_counts_lower = {key.lower(): count for key, count in block_counts.items()}
    result = {}
    for block, count in block_counts_lower.items():
        match = df[df['Material_Name_lower'].str.contains(block)]
        if not match.empty:
            material_id = match.iloc[0]['Material ID']
            result[material_id] = count

    
    # Display the result in the desired format
    print("Material ID and Counts:", result)

    return output_path_final, result
