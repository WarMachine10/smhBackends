
import math  
from math import atan2, degrees  
import logging 
import re 
from loguru import logger
import numpy as np 
import pandas as pd
import os
from shapely.geometry import (Polygon, LineString, MultiPolygon, Point, JOIN_STYLE)  
from shapely.ops import unary_union, nearest_points 
import ezdxf
from ezdxf.math import (area as calculate_polyline_area, Vec3)
from ezdxf.addons.drawing import (RenderContext, Frontend) 
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt 
from concurrent.futures import ThreadPoolExecutor
from itertools import product 

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
            # Handle MultiPolygon case
            if isinstance(polygon, MultiPolygon):
                return [p for p in polygon]
                
            # Input validation
            if not hasattr(polygon, 'exterior'):
                print(f"Invalid polygon object without exterior: {type(polygon)}")
                return polygon
                
            # Get coordinates and validate
            coords = list(polygon.exterior.coords)
            if not coords:
                print("Polygon has no coordinates")
                return polygon
                
            print(f"Processing polygon with {len(coords)} points")
            
            # Handle polygons with more than 5 points
            if len(coords) > 5:
                # Define threshold for large gaps (adjust as needed)
                LARGE_GAP_THRESHOLD = 100.0  # units
                
                print("Gap Analysis:")
                gaps = []
                for i in range(len(coords)-1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    dx = abs(p1[0] - p2[0])
                    dy = abs(p1[1] - p2[1])
                    gap_size = (dx**2 + dy**2)**0.5
                    
                    if gap_size > LARGE_GAP_THRESHOLD:
                        gaps.append({
                            'start_idx': i,
                            'end_idx': i+1,
                            'start_point': p1,
                            'end_point': p2,
                            'size': gap_size
                        })
                        print(f"Large gap found: {gap_size:.2f} units between points {i} and {i+1}")
                        print(f"  Start: {p1}")
                        print(f"  End: {p2}")
                
                if gaps:
                    # Sort gaps by size
                    gaps.sort(key=lambda x: x['size'], reverse=True)
                    largest_gap = gaps[0]
                    
                    print(f"\nClosing largest gap ({largest_gap['size']:.2f} units)")
                    print(f"From: {largest_gap['start_point']}")
                    print(f"To: {largest_gap['end_point']}")
                    
                    # Create new coordinates with gap closed
                    new_coords = coords[:]
                    start_idx = largest_gap['start_idx']
                    end_idx = largest_gap['end_idx']
                    
                    # Close the gap by averaging the points
                    avg_x = (largest_gap['start_point'][0] + largest_gap['end_point'][0]) / 2
                    avg_y = (largest_gap['start_point'][1] + largest_gap['end_point'][1]) / 2
                    new_point = (avg_x, avg_y)
                    
                    # Insert intermediate point
                    new_coords.insert(end_idx, new_point)
                    
                    # Ensure proper closure
                    if new_coords[0] != new_coords[-1]:
                        new_coords.append(new_coords[0])
                    
                    print(f"Created intermediate point at: {new_point}")
                    
                    try:
                        # Create new polygon with modified coordinates
                        closed_polygon = Polygon(new_coords)
                        if closed_polygon.is_valid:
                            print("Successfully created closed polygon")
                            # Create minimum rectangle
                            try:
                                rect = closed_polygon.envelope
                                if rect and rect.is_valid:
                                    print("Successfully created bounding rectangle")
                                    return rect
                            except Exception as e:
                                print(f"Error creating rectangle: {e}")
                        else:
                            print("Created polygon is invalid")
                    except Exception as e:
                        print(f"Error creating polygon: {e}")
                
                # Fallback to original envelope if gap closing fails
                return polygon.envelope
            
            return polygon
                
        except Exception as e:
            print(f"Error in break_into_rectangles: {e}")
            return polygon
    def oriented_minimum_bounding_box(polygon):
        """Alternative implementation of minimum rotated rectangle."""
        try:
            # Get the convex hull first
            hull = polygon.convex_hull
            
            # If hull is invalid, try to fix
            if not hull.is_valid:
                hull = hull.buffer(0)
                
            # Get hull coordinates
            coords = list(hull.exterior.coords)
            
            # Find all possible angles from hull edges
            angles = []
            for i in range(len(coords) - 1):
                dx = coords[i+1][0] - coords[i][0]
                dy = coords[i+1][1] - coords[i][1]
                angles.append(atan2(dy, dx))
                
            # Find minimum area rectangle
            min_area = float('inf')
            min_rect = None
            
            for angle in angles:
                # Rotate polygon
                cos_a = cos(-angle)
                sin_a = sin(-angle)
                
                rotated_coords = []
                for x, y in coords:
                    rx = x * cos_a - y * sin_a
                    ry = x * sin_a + y * cos_a
                    rotated_coords.append((rx, ry))
                    
                # Get bounding box
                xs = [p[0] for p in rotated_coords]
                ys = [p[1] for p in rotated_coords]
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                
                # Rotate back
                cos_a = cos(angle)
                sin_a = sin(angle)
                
                rect_coords = [
                    (minx * cos_a - miny * sin_a, minx * sin_a + miny * cos_a),
                    (maxx * cos_a - miny * sin_a, maxx * sin_a + miny * cos_a),
                    (maxx * cos_a - maxy * sin_a, maxx * sin_a + maxy * cos_a),
                    (minx * cos_a - maxy * sin_a, minx * sin_a + maxy * cos_a)
                ]
                
                # Close the polygon
                rect_coords.append(rect_coords[0])
                
                rect = Polygon(rect_coords)
                area = rect.area
                
                if area < min_area:
                    min_area = area
                    min_rect = rect
                    
            return min_rect
            
        except Exception as e:
            print(f"Error in oriented_minimum_bounding_box: {e}")
            return None
    
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
            if row['Layer'] == 'Staircase_outertwall':
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
        plt.show()
    except Exception as e:
        print(f"Error visualizing DXF content: {e}")

def midpoint(x1, y1, x2, y2):
    try:
        return (x1 + x2) / 2, (y1 + y2) / 2
    except Exception as e:
        print(f"Error calculating midpoint: {e}")
        return None, None

def euclidean_distance(x1, y1, x2, y2):
    try:
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    except Exception as e:
        print(f"Error calculating Euclidean distance: {e}")
        return None

def generate_labels_from_offsets(offsets_df, min_distance=0.1):
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
                                labels_data.append((point_counter, mx, my, 'midpoint'))
                                point_counter += 1
                            midpoints.append((mx, my))

                        # Label imaginary midpoint if midpoints are more than 6 feet apart
                        if euclidean_distance(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1]) > 6:
                            imx, imy = midpoint(midpoints[0][0], midpoints[0][1], midpoints[1][0], midpoints[1][1])
                            if imx is not None and imy is not None and not is_overlapping(imx, imy):
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
                        labels_data.append((point_counter, x1, y1, 'vertex'))
                        point_counter += 1

                    # Label midpoint if segment length > 6 feet
                    if length_ft > 6 and not is_overlapping(mx, my):
                        labels_data.append((point_counter, mx, my, 'midpoint'))
                        point_counter += 1

            elif row['Type'] == 'Single Line':
                # For single line, use vertex_1 and vertex_2 coordinates and length1
                x1, y1 = row['vertex_1_x_coordinate'], row['vertex_1_y_coordinate']
                x2, y2 = row['vertex_2_x_coordinate'], row['vertex_2_y_coordinate']
                length_ft = row['length1'] / 12  # Convert from inches to feet
                mx, my = midpoint(x1, y1, x2, y2)

                # Label edge points and midpoint for single lines based on length
                if length_ft < 1.5:
                    if not is_overlapping(mx, my):
                        labels_data.append((point_counter, mx, my, 'midpoint'))
                        point_counter += 1
                else:
                    if not is_overlapping(x1, y1):
                        labels_data.append((point_counter, x1, y1, 'vertex'))
                        point_counter += 1
                    if not is_overlapping(x2, y2):
                        labels_data.append((point_counter, x2, y2, 'vertex'))
                        point_counter += 1

                    # Mark midpoint if length > 6 feet
                    if length_ft > 6 and not is_overlapping(mx, my):
                        labels_data.append((point_counter, mx, my, 'midpoint'))
                        point_counter += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")

    # Convert labels data to DataFrame
    labels_df = pd.DataFrame(labels_data, columns=['Label', 'X_Coordinate', 'Y_Coordinate', 'Point_Type'])

    return labels_df  


def ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name):
    try:
        if "CeilingLight" not in target_dxf.layers:
            target_dxf.layers.new(name="CeilingLight", dxfattribs={'color': 7})
        
        if bulb_block_name in source_dxf.blocks and bulb_block_name not in target_dxf.blocks:
            target_dxf.blocks.new(name=bulb_block_name)
            for entity in source_dxf.blocks[bulb_block_name]:
                target_dxf.blocks[bulb_block_name].add_entity(entity.copy())
    except Exception as e:
        print(f"Error in ensure_layers_and_blocks: {e}")

def place_bulbs(target_msp, labels_df, bulb_block_name):
    try:
        for index, row in labels_df.iterrows():
            x, y = row['X_Coordinate'], row['Y_Coordinate']
            bulb_ref = target_msp.add_blockref(bulb_block_name, insert=(x, y))
            bulb_ref.dxf.layer = "CeilingLight"
    except Exception as e:
        print(f"Error in place_bulbs: {e}")

def main1(
    input_file,
    light_dxf,
    output_path_final,
    offset_distance=24
):
    try:
        # Step 1: Load the target and source DXF files
        target_dxf = ezdxf.readfile(input_file)  # Load architectural drawing
        source_dxf = ezdxf.readfile(light_dxf)  # Load light fixture block
        bulb_block_name = "SM_L"  # Specify the light fixture block name
        # Step 2: Convert the input DXF to a DataFrame for easier manipulation
        df = dxf_to_dataframe(input_file)  # Convert DXF to a DataFrame
        processed_df = preprocess_polylines(df)  # Process polylines for further use
        # Step 3: Generate offsets for placement and label the coordinates
        offsets, offsets_df = create_offset(df, offset_distance=offset_distance)  # Generate offsets for fixtures
        labels_df = generate_labels_from_offsets(offsets_df)  # Prepare labeled points for placement
        # Step 4: Ensure required layers and blocks exist in the target DXF
        ensure_layers_and_blocks(target_dxf, source_dxf, bulb_block_name)  # Ensure light block exists in target DXF
        # Step 5: Place bulbs in the target DXF model space
        target_msp = target_dxf.modelspace()  # Access model space for placement
        place_bulbs(target_msp, labels_df, bulb_block_name)  # Place bulbs at predefined positions
        # Step 16: Save the modified DXF file
        target_dxf.saveas(output_path_final)  # Save the modified DXF file
        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred

def ensure_switch_block(target_dxf, sw_doc):
    try:
        if "M_SW" not in sw_doc.blocks:
            raise ValueError("The block 'M_SW' was not found in the M_SW.dxf file.")
        sw_block = sw_doc.blocks.get("M_SW")
        
        if "M_SW" not in target_dxf.blocks:
            target_dxf.blocks.new(name="M_SW")
            for entity in sw_block:
                target_dxf.blocks["M_SW"].add_entity(entity.copy())
    except Exception as e:
        print(f"Error in ensure_switch_block: {e}")

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
    
def place_switches(target_msp, distances):
    placed_layers = set()
    try:
        for polyline in target_msp.query("POLYLINE LWPOLYLINE"):
            polyline_layer = polyline.dxf.layer
            if polyline_layer in placed_layers:
                continue

            same_layer_lines = [line for line in target_msp.query("LINE") if line.dxf.layer == polyline_layer]
            other_layer_lines = [line for line in target_msp.query("LINE") if line.dxf.layer != polyline_layer]
            text_entities = [text for text in target_msp.query("TEXT MTEXT") if text.dxf.layer == polyline_layer]

            switch_placed = False
            for line in same_layer_lines:
                start1, end1 = (line.dxf.start.x, line.dxf.start.y), (line.dxf.end.x, line.dxf.end.y)
                line_midpoint = ((start1[0] + end1[0]) / 2, (start1[1] + end1[1]) / 2)

                for other_line in other_layer_lines:
                    start2, end2 = (other_line.dxf.start.x, other_line.dxf.start.y), (other_line.dxf.end.x, other_line.dxf.end.y)

                    if any(are_parallel((start1, end1), (start2, end2), d) for d in distances):
                        # Determine the base rotation for alignment (downline alignment)
                        if abs(start1[1] - end1[1]) < 1e-6:  # Horizontal line
                            base_rotation = 0
                        elif abs(start1[0] - end1[0]) < 1e-6:  # Vertical line
                            base_rotation = 90
                        else:
                            continue  # Skip non-horizontal/vertical lines

                        # Find the closest text entity to determine block orientation
                        closest_text = min(
                            text_entities,
                            key=lambda text: ((text.dxf.insert.x - line_midpoint[0]) ** 2 + (text.dxf.insert.y - line_midpoint[1]) ** 2),
                            default=None
                        )

                        if closest_text:
                            text_position = (closest_text.dxf.insert.x, closest_text.dxf.insert.y)

                            # Adjust rotation based on the polyline's orientation and text position
                            if base_rotation == 0:  # Horizontal line
                                if text_position[1] > line_midpoint[1]:
                                    rotation = 0  # Upper line faces upwards (text is above)
                                else:
                                    rotation = 180  # Upper line faces downwards (text is below)
                            elif base_rotation == 90:  # Vertical line
                                if text_position[0] > line_midpoint[0]:
                                    rotation = 270  # Upper line faces right (text is to the right)
                                else:
                                    rotation = 90  # Upper line faces left (text is to the left)

                        # Place the block at the adjusted position
                        adjusted_position = calculate_adjusted_position((start1, end1), (start2, end2), line_midpoint)
                        block_ref = target_msp.add_blockref("M_SW", adjusted_position)
                        block_ref.dxf.layer = "M_SW"
                        block_ref.dxf.rotation = rotation

                        placed_layers.add(polyline_layer)
                        switch_placed = True
                        break
                if switch_placed:
                    break
    except Exception as e:
        print(f"Error in place_switches: {e}")


def main2(input_file, switch_dxf, output_path_final):
    try:
        # Load the input DXF file
        target_dxf = ezdxf.readfile(input_file)
        
        # Load the switch block file
        sw_doc = ezdxf.readfile(switch_dxf)
        
        # Ensure the switch block is present in the target DXF
        ensure_switch_block(target_dxf, sw_doc)
        
        # Access the modelspace of the target DXF
        target_msp = target_dxf.modelspace()
        
        # Place switches at calculated positions
        place_switches(target_msp, distances=[5 / 12, 9 / 12])
        
        # Save the modified DXF file
        target_dxf.saveas(output_path_final)
        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred


# %%
import logging
import ezdxf  # Ensure ezdxf is imported for handling DXF files
from shapely.geometry import Polygon

def calculate_area_and_center_by_diagonal(polyline):
    try:
        points = [(v[0], v[1]) for v in polyline.lwpoints]
        if not polyline.is_closed:
            points.append(points[0])
        polygon = Polygon(points)
        area = polygon.area / 144
        if len(points) >= 4:
            x1, y1 = points[0]  
            x2, y2 = points[2] 
            midpoint_x = (x1 + x2) / 2
            midpoint_y = (y1 + y2) / 2
            return round(area, 2), (round(midpoint_x, 2), round(midpoint_y, 2))

        # Fallback: Use the centroid of the polygon if diagonal intersection is not applicable.
        centroid = polygon.centroid
        return round(area, 2), (round(centroid.x, 2), round(centroid.y, 2))
    except Exception:
        # Return zero area and an origin center in case of an error.
        return 0, (0, 0)

def clear_existing_blocks(doc, block_name="CHND_L"):
    try:
        # Access the model space of the DXF document.
        msp = doc.modelspace()

        # Query and delete all instances of the specified block from the model space.
        for entity in msp.query(f"INSERT[name=='{block_name}']"):
            msp.delete_entity(entity)

        # Log a success message once all blocks are cleared.
        logging.info(f"Cleared all existing '{block_name}' blocks.")
    except Exception as e:
        # Log an error message if something goes wrong.
        logging.error(f"Error clearing blocks: {e}")

def import_blocks(source_file, target_doc):
    try:
        # Read the source DXF file.
        source_doc = ezdxf.readfile(source_file)

        # Iterate through all blocks in the source document.
        for block in source_doc.blocks:
            # Skip anonymous and already existing blocks in the target document.
            if not block.name.startswith("*") and block.name not in target_doc.blocks:
                # Create a new block in the target document with the same name.
                new_block = target_doc.blocks.new(name=block.name)

                # Copy each entity from the source block to the new block.
                for entity in block:
                    copied_entity = entity.copy()
                    # Retain the original layer name of the entity.
                    copied_entity.dxf.layer = entity.dxf.layer
                    new_block.add_entity(copied_entity)

        # Log a success message once all blocks are imported.
        logging.info(f"Blocks from {source_file} imported successfully.")
    except Exception as e:
        # Log an error message if something goes wrong.
        logging.error(f"Error importing blocks: {e}")

def place_fans(doc, fan_placements, fan_block_name="Fan_C", fan_dxf="Fan_C.dxf"):
    try:
        # Import the fan block from the specified DXF file.
        logging.debug("Importing fan block...")
        import_blocks(fan_dxf, doc)

        # Access the model space of the DXF document.
        msp = doc.modelspace()

        # Ensure the fan block exists in the document after importing.
        if fan_block_name not in doc.blocks:
            logging.error(f"Block '{fan_block_name}' is missing in the document.")
            return

        # Initialize a flag to track if any fans were placed.
        fans_placed = False
        logging.debug("Processing polylines for fan placement...")

        # Iterate through the specified fan placements for each layer.
        for layer_name, polylines in fan_placements.items():
            if fan_placements[layer_name]:  # Check if there are polylines available for this layer.
                # Identify the largest polyline by area.
                largest_polyline = max(polylines, key=lambda p: calculate_area_and_center_by_diagonal(p)[0])

                # Calculate the centroid of the largest polyline for placement.
                _, centroid = calculate_area_and_center_by_diagonal(largest_polyline)

                # Add a block reference to the model space at the centroid location.
                msp.add_blockref(fan_block_name, insert=centroid)  # Layer is determined by the block definition.

                # Mark that a fan has been placed and log the placement details.
                fans_placed = True
                logging.info(f"Fan placed on layer '{layer_name}' at {centroid}.")

    except Exception as e:
        # Log and print error details if fan placement fails.
        logging.error(f"Error placing fans: {e}")
        print("An error occurred while placing fans. Check the logs for details.")

def main3(
    input_file,
    fan_dxf,
    output_path_final
):
    try:
        # Step 1: Load the input DXF file
        target_dxf = ezdxf.readfile(input_file)  # Read the input DXF file

        msp = target_dxf.modelspace()
        layer_polylines = {}

        # Step 2: Organize polylines by layer
        for polyline in msp.query("LWPOLYLINE"):
            layer_name = polyline.dxf.layer
            if layer_name not in layer_polylines:
                layer_polylines[layer_name] = []
            layer_polylines[layer_name].append(polyline)

        # Step 3: Predefined layers for fan placement
        predefined_layers = ["BedRoom1", "Bedroom2", "Bedroom3", "LivingRoom1", "LivingRoom2", "LivingRoom", "DiningRoom", ]
        fan_placements = {layer_name: layer_polylines[layer_name] for layer_name in predefined_layers if layer_name in layer_polylines}

        # Place fans only on the largest polyline of each layer
        place_fans(target_dxf, fan_placements, fan_block_name="Fan_C", fan_dxf=fan_dxf)

        # Step 4: Save the modified DXF file
        target_dxf.saveas(output_path_final)  # Save the modified DXF file
        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred

def get_midpoints(geometry):
    """Calculate midpoints of lines, polylines, or arcs."""
    midpoints = []
    if geometry.dxftype() == 'LINE':
        start, end = Vec3(geometry.dxf.start), Vec3(geometry.dxf.end)
        mid = Vec3((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
        midpoints.append(mid)
    elif geometry.dxftype() in ['LWPOLYLINE', 'POLYLINE']:
        points = geometry.get_points()
        for i in range(len(points) - 1):
            start, end = Vec3(points[i]), Vec3(points[i + 1])
            mid = Vec3((start.x + end.x) / 2, (start.y + end.y) / 2, (start.z + end.z) / 2)
            midpoints.append(mid)
    elif geometry.dxftype() == 'ARC':
        center = Vec3(geometry.dxf.center)
        start_angle = geometry.dxf.start_angle
        end_angle = geometry.dxf.end_angle
        mid_angle = (start_angle + end_angle) / 2
        mid_x = center.x + geometry.dxf.radius * math.cos(math.radians(mid_angle))
        mid_y = center.y + geometry.dxf.radius * math.sin(math.radians(mid_angle))
        midpoints.append(Vec3(mid_x, mid_y, center.z))
    return midpoints
def find_nearest_geometry(midpoint, geometries):
    """Find the nearest geometry to a given point."""
    min_distance = float('inf')
    nearest_geometry = None

    for geometry in geometries:
        points = get_midpoints(geometry) if geometry.dxftype() in ['LINE', 'LWPOLYLINE', 'ARC'] else []
        for point in points:
            distance = (Vec3(point) - Vec3(midpoint)).magnitude
            if distance < min_distance:
                min_distance = distance
                nearest_geometry = geometry
    return nearest_geometry
def snap_to_orthogonal(angle):
    """Snap the given angle to the nearest orthogonal direction (90°, 180°, 270°, 360°)."""
    return round(angle / 90) * 90
def compute_angle(from_point, to_point):
    """Calculate angle (in degrees) from one point to another and snap to orthogonal."""
    dx = to_point.x - from_point.x
    dy = to_point.y - from_point.y
    angle = math.degrees(math.atan2(dy, dx))
    # Reverse the angle to ensure proper alignment
    angle = (angle + 180) % 360
    # Snap to the nearest orthogonal angle
    return snap_to_orthogonal(angle)
def copy_and_place_wall_light(wall_light, target_path, outer_midpoints, inner_geometries):
    """Copy and place Wall_light block with updated placement logic."""
    source_doc = ezdxf.readfile(wall_light)
    if 'ACC_L' not in source_doc.blocks:
        raise ValueError("Block 'ACC_L' not found in source file.")

    # Copy the block definition to the target document
    if 'ACC_L' not in target_path.blocks:
        target_path.blocks.new(name='ACC_L', base_point=Vec3(0, 0, 0))
        for entity in source_doc.blocks['ACC_L']:
            target_path.blocks['ACC_L'].add_entity(entity.copy())

    msp = target_path.modelspace()
    for midpoint in outer_midpoints:
        # Find nearest geometry on the inner wall
        nearest_geometry = find_nearest_geometry(midpoint, inner_geometries)
        if nearest_geometry:
            inner_midpoints = get_midpoints(nearest_geometry)
            nearest_point = min(inner_midpoints, key=lambda p: (Vec3(p) - Vec3(midpoint)).magnitude)
            rotation_angle = compute_angle(midpoint, nearest_point)
        else:
            rotation_angle = 0  # Default to no rotation if no inner wall geometry found

        # Place and rotate the block
        msp.add_blockref('ACC_L', midpoint, dxfattribs={'rotation': rotation_angle})
def main4(
    input_file,
    output_path_final,
    wall_light,
):
    try:
        # Load architectural drawing
        target_dxf = ezdxf.readfile(input_file)
        msp = target_dxf.modelspace()  # Initialize msp

        # Step 11: Wall Light Placement Logic
        outer_midpoints = []
        inner_geometries = []
        for entity in msp:
            if entity.dxf.layer == 'Staircase_outerwall' and entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC']:
                outer_midpoints.extend(get_midpoints(entity))  # Get midpoints of outer walls
            elif entity.dxf.layer == 'Staircase_innerwall' and entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE', 'ARC']:
                inner_geometries.append(entity)  # Get inner wall geometries

        # Step 12: Detect wall light locations and place the wall light
        if outer_midpoints and inner_geometries:
            copy_and_place_wall_light(wall_light, target_dxf, outer_midpoints, inner_geometries)
        
        # Save the modified DXF file
        target_dxf.saveas(output_path_final)
        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred


# %%
def clean_layer_name(name):
    """Standardizes layer names for matching by removing numbers, special characters, and extra spaces."""
    return re.sub(r'[^a-z]', '', name.replace(" ", "").lower())
def is_target_layer(layer_name):
    """Check if the layer name matches target rooms, accounting for typos."""
    cleaned_name = clean_layer_name(layer_name)

    # Define all acceptable layer names and typo variations
    target_variations = {
        'bedroom': [
            'BEDROOM', 'Bedroom', 'BedRoom', 'bedRoom', 'b edroom', 'b  edroom', 'b-edroom', 'b3droom', 'b_edroom',
            'bbedroom', 'bderoom', 'bdroom', 'be droom', 'be  droom', 'be-droom', 'be_droom', 'bed room', 'bed  room',
            'bed-room', 'bed_room', 'beddroom', 'bedoom', 'bedorom', 'bedr oom', 'bedr  oom', 'bedr-oom', 'bedr0om',
            'bedr_oom', 'bedro om', 'bedro  om', 'bedro-om', 'bedro0m', 'bedro_om', 'bedrom', 'bedromo', 'bedroo',
            'bedroo m', 'bedroo  m', 'bedroo-m', 'bedroo_m', 'bedroom ', ' bedroom', 'bedroom  ', 'bedroomm',
            'bedrooom', 'bedrroom', 'beedroom', 'berdoom', 'beroom', 'ebdroom', 'edroom'
        ],
        'diningroom': [
            'DININGROOM', 'Diningroom', 'DiningRoom', 'diningRoom', 'd iningroom', 'd  iningroom', 'd-iningroom',
            'd1ningroom', 'd_iningroom', 'ddiningroom', 'di ningroom', 'di  ningroom', 'di-ningroom', 'di_ningroom',
            'diingroom', 'diiningroom', 'diinngroom', 'din ingroom', 'din  ingroom', 'din-ingroom', 'din1ngroom',
            'din_ingroom', 'dini ngroom', 'dini  ngroom', 'dini-ngroom', 'dini_ngroom', 'dinignroom', 'dinigroom',
            'diniingroom', 'dinin groom', 'dinin  groom', 'dinin-groom', 'dinin_groom', 'dining room', 'dining  room',
            'dining-room', 'dining_room', 'dininggroom', 'diningoom', 'diningorom', 'diningr oom', 'diningr  oom',
            'diningr-oom', 'diningr0om', 'diningr_oom', 'diningro om', 'diningro  om', 'diningro-om', 'diningro0m',
            'diningro_om', 'diningrom', 'diningromo', 'diningroo', 'diningroo m', 'diningroo  m', 'diningroo-m',
            'diningroo_m', 'diningroom ', ' diningroom', 'diningroom  ', 'diningroomm', 'diningrooom', 'diningrroom',
            'dininngroom', 'dininrgoom', 'dininroom', 'dinngroom', 'dinnigroom', 'dinningroom', 'dniingroom',
            'dningroom', 'idningroom', 'iningroom'
        ],
        'livingroom': [
            '1ivingroom', 'LIVINGROOM', 'Livingroom', 'LivingRoom', 'livingRoom', 'ilvingroom', 'ivingroom',
            'l ivingroom', 'l  ivingroom', 'l-ivingroom', 'l1vingroom', 'l_ivingroom', 'li vingroom', 'li  vingroom',
            'li-vingroom', 'li_vingroom', 'liingroom', 'liivingroom', 'liivngroom', 'liv ingroom', 'liv  ingroom',
            'liv-ingroom', 'liv1ngroom', 'liv_ingroom', 'livi ngroom', 'livi  ngroom', 'livi-ngroom', 'livi_ngroom',
            'livignroom', 'livigroom', 'liviingroom', 'livin groom', 'livin  groom', 'livin-groom', 'livin_groom',
            'living room', 'living  room', 'living-room', 'living_room', 'livinggroom', 'livingoom', 'livingorom',
            'livingr oom', 'livingr  oom', 'livingr-oom', 'livingr0om', 'livingr_oom', 'livingro om', 'livingro  om',
            'livingro-om', 'livingro0m', 'livingro_om', 'livingrom', 'livingromo', 'livingroo', 'livingroo m',
            'livingroo  m', 'livingroo-m', 'livingroo_m', 'livingroom ', ' livingroom', 'livingroom  ', 'livingroomm',
            'livingrooom', 'livingrroom', 'livinngroom', 'livinrgoom', 'livinroom', 'livngroom', 'livnigroom',
            'livvingroom', 'llivingroom', 'lviingroom', 'lvingroom'
        ]
    }

    # Check if the cleaned layer name matches any of the variations
    for target, variations in target_variations.items():
        if any(cleaned_name.startswith(clean_layer_name(variation)) for variation in variations):
            return True

    return False
def calculate_centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    centroid_x = sum(x_coords) / len(points)
    centroid_y = sum(y_coords) / len(points)
    return centroid_x, centroid_y
def are_lines_parallel(line1, line2, angle_tolerance=10, distance_tolerance=10):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    angle1 = degrees(atan2(y2 - y1, x2 - x1)) % 180
    angle2 = degrees(atan2(y4 - y3, x4 - x3)) % 180

    if abs(angle1 - angle2) > angle_tolerance:
        return False

    def point_to_line_distance(px, py, ax, ay, bx, by):
        return abs((by - ay) * px - (bx - ax) * py + bx * ay - by * ax) / (((by - ay)**2 + (bx - ax)**2)**0.5)

    distance1 = point_to_line_distance(x3, y3, x1, y1, x2, y2)
    distance2 = point_to_line_distance(x4, y4, x1, y1, x2, y2)

    return distance1 <= distance_tolerance and distance2 <= distance_tolerance
def has_parallel_line(polyline, all_polylines, angle_tolerance=10, distance_tolerance=10):
    """Check if the polyline has at least one valid parallel line."""
    for other_polyline in all_polylines:
        if polyline is other_polyline:
            continue

        for i in range(len(polyline) - 1):
            for j in range(len(other_polyline) - 1):
                line1 = (polyline[i], polyline[i + 1])
                line2 = (other_polyline[j], other_polyline[j + 1])

                if are_lines_parallel(line1, line2, angle_tolerance, distance_tolerance):
                    return True

    return False
def get_largest_polylines_with_rotation(msp):
    layer_polylines = {}
    all_polylines = []

    for entity in msp.query('LWPOLYLINE'):
        layer = entity.dxf.layer
        if is_target_layer(layer):
            points = list(entity.get_points('xy'))
            all_polylines.append(points)

            length = 0
            for i in range(len(points) - 1):
                start, end = points[i], points[i + 1]
                length += ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5

            if layer not in layer_polylines or length > layer_polylines[layer]['length']:
                layer_polylines[layer] = {'polyline': points, 'length': length}

    midpoints_with_angles = []
    centroids = []

    for data in layer_polylines.values():
        polyline = data['polyline']

        if not has_parallel_line(polyline, all_polylines):
            continue

        centroids.append(calculate_centroid(polyline))

        longest_length = 0
        longest_segment = None

        for i in range(len(polyline) - 1):
            start_point, end_point = polyline[i], polyline[i + 1]
            length = ((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5
            if length > longest_length:
                longest_length = length
                longest_segment = (start_point, end_point)

        if longest_segment:
            start_point, end_point = longest_segment
            midpoint = Vec3(
                (start_point[0] + end_point[0]) / 2,
                (start_point[1] + end_point[1]) / 2,
                0
            )
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            angle = degrees(atan2(dy, dx))
            midpoints_with_angles.append((midpoint, angle))

    return midpoints_with_angles, centroids
def adjust_ac_rotation_for_orientation(angle, midpoint, centroid):
# Calculate angle between the midpoint and the centroid
    dx = centroid[0] - midpoint.x
    dy = centroid[1] - midpoint.y
    angle_to_centroid = degrees(atan2(dy, dx))

    # Normalize to nearest 90° increment (cardinal directions)
    cardinal_angles = [0, 90, 180, 270, 360]
    adjusted_angle = min(cardinal_angles, key=lambda x: abs((angle_to_centroid - x) % 360))

    # Adjust rotation to align downside based on polyline orientation
    if 45 <= angle % 360 < 135 or 225 <= angle % 360 < 315:
        # Vertical alignment (90° or 270°)
        if adjusted_angle in [0, 180]:
            adjusted_angle = 90 if adjusted_angle < 180 else 270
    else:
        # Horizontal alignment (0° or 180°)
        if adjusted_angle in [90, 270]:
            adjusted_angle = 0 if adjusted_angle < 180 else 180

    return adjusted_angle
def copy_and_place_ac_block(ac_dxf, target_plan, midpoints_with_angles, centroids):
    
    source_doc = ezdxf.readfile(ac_dxf)
    if 'AC' not in source_doc.blocks:
        raise ValueError("Block 'AC' not found in source file.")

    # Copy the block definition to the target document
    if 'AC' not in target_plan.blocks:
        target_plan.blocks.new(name='AC', base_point=Vec3(0, 0, 0))
        for entity in source_doc.blocks['AC']:
            target_plan.blocks['AC'].add_entity(entity.copy())

    msp = target_plan.modelspace()

    for (midpoint, angle), centroid in zip(midpoints_with_angles, centroids):
        # Adjust rotation to align AC downside to vertical/horizontal and face centroid
        adjusted_angle = adjust_ac_rotation_for_orientation(angle, midpoint, centroid)

        # Add block reference with the adjusted rotation angle
        msp.add_blockref('AC', midpoint, dxfattribs={'rotation': adjusted_angle})
def main5(
    input_file,
    output_path_final,
    ac_dxf
):
    try:
       
        target_dxf = ezdxf.readfile(input_file)  # Load architectural drawing
        msp = target_dxf.modelspace()  # Initialize msp
        # Step 13: AC Placement Logic
        midpoints_with_angles, centroids = get_largest_polylines_with_rotation(msp)
        if midpoints_with_angles:
            copy_and_place_ac_block(ac_dxf, target_dxf, midpoints_with_angles, centroids)  # Place AC blocks
        # Step 16: Save the modified DXF file
        target_dxf.saveas(output_path_final)  # Save the modified DXF file

        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred


# %%
def calculate_midpoint(start, end):
    return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
def calculate_angle(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle
def ensure_mbd_block(target_dxf, mbd_dxf_path):
    try:
        # Load the MBD DXF file
        mbd_dxf = ezdxf.readfile(mbd_dxf_path)
        
        # Check if the MBD block exists in the source DXF
        if "MBD" not in mbd_dxf.blocks:
            raise ValueError("MBD block not found in the provided MBD DXF file.")
        
        # Check if the MBD block already exists in the target DXF
        if "MBD" not in target_dxf.blocks:
            # If not, add it to the target DXF
            target_block = target_dxf.blocks.new(name="MBD")
            for entity in mbd_dxf.blocks["MBD"]:
                target_block.add_entity(entity.copy())
    except Exception as e:
        raise RuntimeError(f"Failed to ensure MBD block: {e}")
def main6(
    input_file,
    output_path_final,
    MBD_dxf,
):
    try:
        # Step 1: Load the target and source DXF files
        target_dxf = ezdxf.readfile(input_file)  # Load architectural drawing
        msp = target_dxf.modelspace()  # Initialize msp

        # Step 14: MBD Placement Logic
        foyer_lines = [entity for entity in msp.query('LINE') if entity.dxf.layer == "Foyer"]
        if foyer_lines:
            shortest_line = min(foyer_lines, key=lambda line: math.dist(line.dxf.start, line.dxf.end))  # Find the shortest foyer line
            midpoint = calculate_midpoint(shortest_line.dxf.start, shortest_line.dxf.end)  # Calculate midpoint of the line
            angle = calculate_angle(shortest_line.dxf.start, shortest_line.dxf.end)  # Calculate the angle of placement
            angle = round(angle / 90) * 90 if angle % 90 != 0 else angle  # Round angle to the nearest 90°

            block_doc = ezdxf.readfile(MBD_dxf)  # Load MBD block
            if "MBD" not in block_doc.blocks:
                print("Block 'MBD' not found in the block file.")
                return None

            block_definition = block_doc.blocks["MBD"]  # Get MBD block definition
            if "MBD" not in target_dxf.blocks:
                new_block = target_dxf.blocks.new(name="MBD")  # Create new MBD block if not present
                for entity in block_definition:
                    new_block.add_entity(entity.copy())

            block_ref = msp.add_blockref("MBD", insert=midpoint, dxfattribs={"layer": "MBD"})  # Place MBD block
            block_ref.dxf.rotation = angle  # Set MBD block rotation angle

  
        target_dxf.saveas(output_path_final)  # Save the modified DXF file

        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred

# %%
import ezdxf
from ezdxf.math import Vec3
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt
import math

def get_largest_line_midpoint_and_angle(doc, layer_name="Parking"):
    """Finds the largest line in the specified layer and returns its midpoint and angle with the X-axis."""
    largest_line = None
    max_length = 0
    line_angle = 0  # Default angle in degrees

    for entity in doc.modelspace().query(f"LINE[layer=='{layer_name}']"):
        line = entity
        start = Vec3(line.dxf.start)
        end = Vec3(line.dxf.end)
        length = (start - end).magnitude
        if length > max_length:
            max_length = length
            largest_line = line

    if largest_line:
        start = Vec3(largest_line.dxf.start)
        end = Vec3(largest_line.dxf.end)
        midpoint = start.lerp(end)

        # Calculate the angle of the line with respect to the X-axis
        delta = end - start
        line_angle = math.degrees(math.atan2(delta.y, delta.x))

        return midpoint, line_angle
    return None, None

def find_nearest_text_angle(doc, reference_point, target_text="Parking"):
    """Finds the nearest text with the target content and calculates its angle relative to the reference point."""
    min_distance = float("inf")
    nearest_text = None
    nearest_angle = 0

    for entity in doc.modelspace().query("TEXT"):
        if entity.dxf.text.lower() == target_text.lower():
            text_pos = Vec3(entity.dxf.insert)
            distance = (text_pos - reference_point).magnitude
            if distance < min_distance:
                min_distance = distance
                nearest_text = entity
                # Calculate the angle from the reference point to the text
                delta = text_pos - reference_point
                nearest_angle = math.degrees(math.atan2(delta.y, delta.x))

    return nearest_angle

def import_and_paste_block(doc, EvSwitch_dxf, EV_block_name, EV_layer_name, insert_point, rotation_angle):
    """Imports a block from another DXF file and pastes it at the specified point with rotation."""
    try:
        source_doc = ezdxf.readfile(EvSwitch_dxf)

        if EV_block_name not in source_doc.blocks:
            raise ValueError(f"Block '{EV_block_name}' not found in {EvSwitch_dxf}")

        # Copy block definition to the target document
        source_block = source_doc.blocks.get(EV_block_name)
        if EV_block_name not in doc.blocks:
            new_block = doc.blocks.new(name=EV_block_name)
            for entity in source_block:
                new_block.add_entity(entity.copy())

        # Ensure the layer exists
        if EV_layer_name not in doc.layers:
            doc.layers.add(name=EV_layer_name)

        # Insert block reference in the target document
        doc.modelspace().add_blockref(
            EV_block_name,
            insert_point,
            dxfattribs={"layer": EV_layer_name, "rotation": rotation_angle},
        )

    except Exception as e:
        pass

def main7(
    input_file,
    EvSwitch_dxf,
    output_path_final,
    user_input="yes"
):
    try:
       
        target_dxf = ezdxf.readfile(input_file)  # Load architectural drawing
    
        # Step 15: EV Switch Placement Logic
        midpoint, line_angle = get_largest_line_midpoint_and_angle(target_dxf)
        if midpoint is not None:
            text_angle = find_nearest_text_angle(target_dxf, midpoint, target_text="Parking")
            rotation_angle = (text_angle - line_angle) % 360
            if user_input == "yes":
                EV_block_name = "EvSwitch"
                EV_layer_name = "EvSwitch"
                import_and_paste_block(target_dxf, EvSwitch_dxf, EV_block_name, EV_layer_name, midpoint, rotation_angle)

        # Step 16: Save the modified DXF file
        target_dxf.saveas(output_path_final)  # Save the modified DXF fil
        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred


def main_process(input_file, light_dxf, switch_dxf, fan_dxf, wall_light, ac_dxf, MBD_dxf, EvSwitch_dxf, output_file_final,user_input):
    """
    Main function to handle sequential DXF processing and clean up intermediate files.

    Args:
        input_file (str): The initial input DXF file.
        light_dxf (str): DXF file for ceiling lights.
        switch_dxf (str): DXF file for switches.
        fan_dxf (str): DXF file for fans.
        wall_light (str): DXF file for wall lights.
        ac_dxf (str): DXF file for air conditioners.
        MBD_dxf (str): DXF file for MBD.
        EvSwitch_dxf (str): DXF file for EV switches.
        output_file_final (str): Final output DXF file name.
    """
    # Ask the user about EV vehicles
    

    intermediate_files = [
        "Intermediate1.dxf", "Intermediate2.dxf", "Intermediate3.dxf",
        "Intermediate4.dxf", "Intermediate5.dxf", "Intermediate6.dxf"
    ]

    # Sequential processing using provided main functions
    main1(input_file=input_file, light_dxf=light_dxf, output_path_final=intermediate_files[0])
    main2(input_file=intermediate_files[0], switch_dxf=switch_dxf, output_path_final=intermediate_files[1])
    main3(input_file=intermediate_files[1], fan_dxf=fan_dxf, output_path_final=intermediate_files[2])
    main4(input_file=intermediate_files[2], wall_light=wall_light, output_path_final=intermediate_files[3])
    main5(input_file=intermediate_files[3], ac_dxf=ac_dxf, output_path_final=intermediate_files[4])
    main6(input_file=intermediate_files[4], MBD_dxf=MBD_dxf, output_path_final=intermediate_files[5])
    main7(input_file=intermediate_files[5], EvSwitch_dxf=EvSwitch_dxf, output_path_final=output_file_final, user_input=user_input)

    # Organize and delete intermediate files
    files_structure = {"Intermediate Files": intermediate_files}
    print("Files structure:", files_structure)

    for file in intermediate_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"File not found: {file}")

# Example usage
main_process(
    input_file="Test6.dxf",
    light_dxf="Ceiling_Lights.dxf",
    switch_dxf="M_SW.dxf",
    fan_dxf="Fan_C.dxf",
    wall_light="Wall_Lights.dxf",
    ac_dxf="AC.dxf",
    MBD_dxf="MBD.dxf",
    EvSwitch_dxf="EvSwitch.dxf",
    output_file_final="Electrical_Drawing.dxf",
    user_input = "yes"
)

