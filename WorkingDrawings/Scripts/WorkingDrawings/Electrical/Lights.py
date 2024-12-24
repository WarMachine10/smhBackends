
import ezdxf  # For reading and writing DXF files
import pandas as pd  # For organizing polyline data in a DataFrame
from shapely.geometry import Polygon, LineString, MultiPolygon, JOIN_STYLE  # For geometry manipulations
from shapely.ops import unary_union  # For geometry operations
import matplotlib.pyplot as plt  # For visualizing DXF contents
import math  # For calculating Euclidean distances
import numpy as np
import matplotlib
matplotlib.use('agg')
def calculate_distance_polyline(coord1, coord2):
    try:
        x1, y1 = coord1
        x2, y2 = coord2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    except Exception as e:
        print(f"Error calculating distance between polylines: {e}")
        return 0

def calculate_length(coords):
    length = 0.0
    try:
        for i in range(1, len(coords)):
            length += calculate_distance_polyline(coords[i - 1], coords[i])
    except Exception as e:
        print(f"Error calculating polyline length: {e}")
    return length

def dxf_to_dataframe(dxf_file):
    try:
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
    except Exception as e:
        print(f"Error reading DXF file or creating DataFrame: {e}")
        return pd.DataFrame()

def preprocess_polylines(df):
    processed_data = []
    try:
        for index, row in df.iterrows():
            data = {'layer': row['Layer']}
            coords = row['Coordinates']
            for i, (x, y) in enumerate(coords):
                data[f'vertex_{i+1}_x_coordinate'] = x
                data[f'vertex_{i+1}_y_coordinate'] = y
                if i < len(coords) - 1:
                    next_x, next_y = coords[i + 1]
                    data[f'length{i+1}'] = calculate_distance_polyline((x, y), (next_x, next_y))
            processed_data.append(data)
        processed_df = pd.DataFrame(processed_data)
        return processed_df
    except Exception as e:
        print(f"Error processing polylines: {e}")
        return pd.DataFrame()

def calculate_distance_offset(point1, point2):
    try:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    except Exception as e:
        print(f"Error calculating distance offset: {e}")
        return 0

def create_offset(df, offset_distance=2):
    offset_entities = []
    offsets_data = []
    
    def generate_offset(polyline_coords, offset_distance, layer):
        try:
            polygon = Polygon(polyline_coords)
            if polygon.is_valid:
                offset_polygon = polygon.buffer(-offset_distance, resolution=16, join_style=JOIN_STYLE.mitre)
                return offset_polygon if offset_polygon.is_valid else polygon
        except Exception as e:
            print(f"Error generating offset for polyline: {e}")
        return None
    
    def break_into_rectangles(polygon):
        try:
            if isinstance(polygon, MultiPolygon):
                return [p for p in polygon]
            if len(polygon.exterior.coords) > 5:
                return polygon.minimum_rotated_rectangle
            else:
                return polygon
        except Exception as e:
            print(f"Error breaking polygon into rectangles: {e}")
            return polygon
    
    def move_side_lines_away(polygon, side_lines, offset_distance, direction='right'):
        new_side_lines = []
        try:
            for line in side_lines:
                if polygon.distance(line) < offset_distance:
                    new_line = line.parallel_offset(offset_distance, side=direction)
                    new_side_lines.append(new_line)
                else:
                    new_side_lines.append(line)
        except Exception as e:
            print(f"Error moving side lines: {e}")
        return new_side_lines

    try:
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
                    offset_polygon = break_into_rectangles(offset_polygon)
                    if isinstance(offset_polygon, MultiPolygon):
                        for p in offset_polygon:
                            side_lines = [LineString([p.exterior.coords[i], p.exterior.coords[i+1]]) for i in range(len(p.exterior.coords)-1)]
                            side_lines = move_side_lines_away(polygon, side_lines, offset_distance)
                            offset_entities.append((p, layer))
                            offset_data = {'layer': layer, 'Type': 'Offset'}
                            offset_coords = list(p.exterior.coords)
                            for i, (x, y) in enumerate(offset_coords):
                                offset_data[f'vertex_{i+1}_x_coordinate'] = x
                                offset_data[f'vertex_{i+1}_y_coordinate'] = y
                                if i < len(offset_coords) - 1:
                                    next_x, next_y = offset_coords[i + 1]
                                    offset_data[f'length{i+1}'] = calculate_distance_offset((x, y), (next_x, next_y))
                            offsets_data.append(offset_data)
                    else:
                        side_lines = [LineString([offset_polygon.exterior.coords[i], offset_polygon.exterior.coords[i+1]]) for i in range(len(offset_polygon.exterior.coords)-1)]
                        side_lines = move_side_lines_away(polygon, side_lines, offset_distance)
                        offset_entities.append((offset_polygon, layer))
                        offset_data = {'layer': layer, 'Type': 'Offset'}
                        offset_coords = list(offset_polygon.exterior.coords)
                        for i, (x, y) in enumerate(offset_coords):
                            offset_data[f'vertex_{i+1}_x_coordinate'] = x
                            offset_data[f'vertex_{i+1}_y_coordinate'] = y
                            if i < len(offset_coords) - 1:
                                next_x, next_y = offset_coords[i + 1]
                                offset_data[f'length{i+1}'] = calculate_distance_offset((x, y), (next_x, next_y))
                        offsets_data.append(offset_data)
            else:
                mid_point = polygon.centroid
                if horizontal_length <= offset_distance * 2:
                    line = LineString([(mid_point.x, bounding_box[1] + offset_distance), 
                                       (mid_point.x, bounding_box[3] - offset_distance)])
                elif vertical_length <= offset_distance * 2:
                    line = LineString([(bounding_box[0] + offset_distance, mid_point.y),
                                       (bounding_box[2] - offset_distance, mid_point.y)])
                offset_entities.append((line, layer))
                line_coords = list(line.coords)
                offset_data = {'layer': layer, 'Type': 'Single Line'}
                for i, (x, y) in enumerate(line_coords):
                    offset_data[f'vertex_{i+1}_x_coordinate'] = x
                    offset_data[f'vertex_{i+1}_y_coordinate'] = y
                    if i < len(line_coords) - 1:
                        next_x, next_y = line_coords[i + 1]
                        offset_data[f'length{i+1}'] = calculate_distance_offset((x, y), (next_x, next_y))
                offsets_data.append(offset_data)
    except Exception as e:
        print(f"Error creating offsets: {e}")
    
    offsets_df = pd.DataFrame(offsets_data)
    return offset_entities, offsets_df

def save_offsets_to_dxf(offset_entities, doc):
    try:
        msp = doc.modelspace()
        for offset_entity, layer in offset_entities:
            if isinstance(offset_entity, Polygon):
                coords = list(offset_entity.exterior.coords)
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polyline = msp.add_lwpolyline(coords)
                polyline.dxf.layer = layer
            elif isinstance(offset_entity, LineString):
                line = msp.add_line(offset_entity.coords[0], offset_entity.coords[1])
                line.dxf.layer = layer
                line.dxf.color = 5
        doc.saveas('offset_dxf.dxf')
    except Exception as e:
        print(f"Error saving offsets to DXF file: {e}")

def visualize_dxf(df):
    try:
        fig, ax = plt.subplots()
        for i, row in df.iterrows():
            coords = row['Coordinates']
            x, y = zip(*coords)
            ax.plot(x, y, label=row['Layer'])
        ax.legend()
        # plt.show()
    except Exception as e:
        print(f"Error visualizing DXF content: {e}")

# %%
# Function to calculate midpoint of two points
def midpoint(x1, y1, x2, y2):
    try:
        return (x1 + x2) / 2, (y1 + y2) / 2
    except Exception as e:
        print(f"Error calculating midpoint: {e}")
        return None, None

# Function to calculate Euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    try:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except Exception as e:
        print(f"Error calculating Euclidean distance: {e}")
        return None

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
        try:
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
                            if mx is not None and my is not None and not is_overlapping(mx, my):
                                ax.text(mx, my, str(point_counter), color='green', fontsize=10, ha='center')
                                labels_data.append((point_counter, mx, my, 'midpoint'))
                                point_counter += 1
                            midpoints.append((mx, my))

                        # Label imaginary midpoint if midpoints are more than 6 feet apart
                        if euclidean_distance(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1]) > 6:
                            imx, imy = midpoint(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1])
                            if imx is not None and imy is not None and not is_overlapping(imx, imy):
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

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    # Convert labels data to DataFrame
    labels_df = pd.DataFrame(labels_data, columns=['Label', 'X_Coordinate', 'Y_Coordinate', 'Point_Type'])

    # Adding labels and title
    ax.set_title('Offset Polylines with Marked Points and Lengths (Non-overlapping Labels)', fontsize=16)
    ax.set_xlabel('X Coordinates')
    ax.set_ylabel('Y Coordinates')
    ax.legend()

    plt.grid(True)
    # plt.show()

    return labels_df  # Return the DataFrame containing non-overlapping label coordinates

# Example usage with your offsets_df
# labels_df = plot_offsets_with_points(offsets_df)

# Display the cleaned labels DataFrame
# print(labels_df)


# %%
import ezdxf
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Helper function to ensure layers and blocks exist
def ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name):
    try:
        if "15w" not in target_dxf.layers:
            target_dxf.layers.new(name="15w", dxfattribs={'color': 7})
        
        if bulb_block_name in source_dxf.blocks and bulb_block_name not in target_dxf.blocks:
            target_dxf.blocks.new(name=bulb_block_name)
            for entity in source_dxf.blocks[bulb_block_name]:
                target_dxf.blocks[bulb_block_name].add_entity(entity.copy())
    except Exception as e:
        print(f"Error in ensure_layers_and_blocks: {e}")

def ensure_switch_block(target_dxf, sw_doc):
    try:
        if "SW" not in sw_doc.blocks:
            raise ValueError("The block 'SW' was not found in the SW.dxf file.")
        sw_block = sw_doc.blocks.get("SW")
        
        if "SW" not in target_dxf.blocks:
            target_dxf.blocks.new(name="SW")
            for entity in sw_block:
                target_dxf.blocks["SW"].add_entity(entity.copy())
    except Exception as e:
        print(f"Error in ensure_switch_block: {e}")

# Bulb placement function
def place_bulbs(target_msp, labels_df, bulb_block_name):
    try:
        for index, row in labels_df.iterrows():
            x, y = row['X_Coordinate'], row['Y_Coordinate']
            bulb_ref = target_msp.add_blockref(bulb_block_name, insert=(x, y))
            bulb_ref.dxf.layer = "15w"
    except Exception as e:
        print(f"Error in place_bulbs: {e}")

# Switch placement helper functions
def are_parallel(line1, line2, distance):
    try:
        if abs(line1[0][0] - line1[1][0]) < 1e-6 and abs(line2[0][0] - line2[1][0]) < 1e-6:
            return abs(line1[0][0] - line2[0][0]) <= distance
        elif abs(line1[0][1] - line1[1][1]) < 1e-6 and abs(line2[0][1] - line2[1][1]) < 1e-6:
            return abs(line1[0][1] - line2[0][1]) <= distance
        return False
    except Exception as e:
        print(f"Error in are_parallel: {e}")
        return False

def calculate_rotation(line_start, line_end):
    try:
        if abs(line_start[1] - line_end[1]) < 1e-6:
            return 0
        elif abs(line_start[0] - line_end[0]) < 1e-6:
            return 90
        return 0
    except Exception as e:
        print(f"Error in calculate_rotation: {e}")
        return 0

def calculate_adjusted_position(layer_line, non_layer_line, midpoint):
    try:
        layer_mid_y = (layer_line[0][1] + layer_line[1][1]) / 2
        non_layer_mid_y = (non_layer_line[0][1] + non_layer_line[1][1]) / 2
        layer_mid_x = (layer_line[0][0] + layer_line[1][0]) / 2
        non_layer_mid_x = (non_layer_line[0][0] + non_layer_line[1][0]) / 2

        offset_distance = 0.5
        if abs(layer_mid_y - non_layer_mid_y) > abs(layer_mid_x - non_layer_mid_x):
            return (midpoint[0], midpoint[1] - offset_distance) if non_layer_mid_y > layer_mid_y else (midpoint[0], midpoint[1] + offset_distance)
        else:
            return (midpoint[0] - offset_distance, midpoint[1]) if non_layer_mid_x > layer_mid_x else (midpoint[0] + offset_distance, midpoint[1])
    except Exception as e:
        print(f"Error in calculate_adjusted_position: {e}")
        return midpoint

# Switch placement function
def place_switches(target_msp, distances):
    placed_layers = set()
    try:
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
    except Exception as e:
        print(f"Error in place_switches: {e}")

def save_and_plot_dxf(target_dxf, target_msp, bulb_block_name):
    # Saving the DXF file
    try:
        target_dxf.saveas("Electrical.dxf")
        print("DXF file saved as 'Electrical.dxf'.")
    except Exception as e:
        print(f"Error saving DXF file: {e}")
        return 

    # Plotting the updated DXF file with both bulbs and switches
    try:
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.set_aspect('equal')
        ax.set_title('DXF Plot with Bulbs and Switches')

        # Plot the lines in the model space
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

        # plt.show()
    except Exception as e:
        print(f"Error plotting DXF file: {e}")

from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

def extract_polylines_and_blocks(dxf_path):
    try:
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
    except Exception as e:
        print(f"Error in extract_polylines_and_blocks: {e}")
        return pd.DataFrame(), pd.DataFrame()

def extract_and_group_blocks(dxf_path, polyline_df, block_df):
    results = []
    try:
        # Group blocks by the polyline they fall within
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
    except Exception as e:
        print(f"Error in extract_and_group_blocks: {e}")
        return pd.DataFrame()

def create_bulb_to_switch_connections(dxf_path, output_path, df):
    try:
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
        print(f"DXF file saved as '{output_path}'")
    except Exception as e:
        print(f"Error in create_bulb_to_switch_connections: {e}")

import ezdxf
import pandas as pd

def main_final(input_dxf_file, source_dxf_path, sw_dxf_path, dxf_path_electrical, excel_file_path, output_path_final, offset_distance=24):
    # Load target DXF files
    target_dxf = ezdxf.readfile(input_dxf_file)
    source_dxf = ezdxf.readfile(source_dxf_path)
    sw_doc = ezdxf.readfile(sw_dxf_path)
    
    bulb_block_name = "15w"

    # Convert DXF to DataFrame and preprocess
    df = dxf_to_dataframe(input_dxf_file)
    processed_df = preprocess_polylines(df)
    offsets, offsets_df = create_offset(df, offset_distance=offset_distance)
    labels_df = plot_offsets_with_points(offsets_df)

    # Ensure necessary layers and blocks are set
    ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name)
    ensure_switch_block(target_dxf, sw_doc)

    # Place bulbs and switches in the target DXF modelspace
    target_msp = target_dxf.modelspace()
    place_bulbs(target_msp, labels_df, bulb_block_name)
    place_switches(target_msp, distances=[5 / 12, 9 / 12])

    # Save the target DXF with new entities
    save_and_plot_dxf(target_dxf, target_msp, bulb_block_name)

    # Extract and process polylines and blocks
    polyline_df, block_df = extract_polylines_and_blocks(dxf_path_electrical)
    grouped_df = extract_and_group_blocks(dxf_path_electrical, polyline_df, block_df)
    create_bulb_to_switch_connections(dxf_path_electrical, output_path_final, grouped_df)

    # Load the final processed DXF file
    dwg = ezdxf.readfile(dxf_path_electrical)

    # Define block names to detect
    target_blocks = ["15w", "SW"]

    # Count occurrences of target blocks in modelspace
    block_counts = {name: 0 for name in target_blocks}
    modelspace = dwg.modelspace()
    for entity in modelspace:
        if entity.dxftype() == "INSERT":
            block_name = entity.dxf.name
            if block_name in block_counts:
                block_counts[block_name] += 1

    # Load Excel file for material matching
    df = pd.read_excel(excel_file_path)
    df.columns = df.columns.str.strip()
    df['Material_Name_lower'] = df['Material Name'].str.lower()

    # Process block counts
    block_counts_lower = {key.lower(): count for key, count in block_counts.items()}

    # Prepare result dictionary to store Material IDs and counts
    result = {}
    for block, count in block_counts_lower.items():
        match = df[df['Material_Name_lower'].str.contains(block)]
        if not match.empty:
            material_id = match.iloc[0]['Material ID']
            result[material_id] = count

    # Display the result
    print("Material ID and Counts:", result)

    return output_path_final, result

