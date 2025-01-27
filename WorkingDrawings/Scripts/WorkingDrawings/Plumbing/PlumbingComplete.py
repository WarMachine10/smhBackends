
import ezdxf
from ezdxf.addons import Importer
from ezdxf.math import Vec2, intersection_line_line_2d
from ezdxf.math import Vec2
import pandas as pd
import ezdxf
from fuzzywuzzy import fuzz
import ezdxf
from ezdxf.math import Vec2
import os

def copy_and_place_blocks(block_file_IC_and_GT, input_dxf1, output_dxf):
    try:
        # Load the block file and input DXF file
        block_doc = ezdxf.readfile(block_file_IC_and_GT)
        input_doc = ezdxf.readfile(input_dxf1)

        # Extract modelspace of the input DXF
        input_msp = input_doc.modelspace()

        # Import blocks (IC and GT)
        blocks_to_import = {"IC": 1, "GT": 0.8}
        importer = Importer(block_doc, input_doc)

        for block_name, scale in blocks_to_import.items():
            if block_name not in block_doc.blocks:
                print(f"Error: Block '{block_name}' not found in block file!")
                return
            importer.import_block(block_name)
        importer.finalize()
        #print("Blocks imported successfully.")

        # Step 1: Place IC and GT in Garden, WashArea, Setback layers
        layer_priority_1 = ["Garden", "Garden1", "Garden2", "WashArea", "WashArea1", "WashArea2", "Setback", "Setback1", "Setback2"]

        for layer in layer_priority_1:
            if layer in input_doc.layers:
                entities = [
                    e for e in input_msp if e.dxf.layer == layer and e.dxftype() == "MTEXT"
                ]
                if entities:
                    for i, (block_name, scale) in enumerate(blocks_to_import.items()):
                        position = (entities[0].dxf.insert[0] + i * 10, entities[0].dxf.insert[1])
                        input_msp.add_blockref(
                            block_name, position, dxfattribs={'xscale': scale, 'yscale': scale}
                        )
                else:
                    #print(f"No MTEXT in '{layer}', checking for lines.")
                    lines = [
                        e for e in input_msp if e.dxf.layer == layer and e.dxftype() in ["LINE", "LWPOLYLINE"]
                    ]
                    if lines:
                        bbox = {
                            "min_x": min(line.dxf.start.x for line in lines),
                            "max_x": max(line.dxf.end.x for line in lines),
                            "min_y": min(line.dxf.start.y for line in lines),
                            "max_y": max(line.dxf.end.y for line in lines),
                        }
                        center = ((bbox["min_x"] + bbox["max_x"]) / 2, (bbox["min_y"] + bbox["max_y"]) / 2)
                        input_msp.add_blockref(
                            "IC", center, dxfattribs={'xscale': 1, 'yscale': 1}
                        )
                        input_msp.add_blockref(
                            "GT", (center[0] + 10, center[1]), dxfattribs={'xscale': 0.8, 'yscale': 0.8}
                        )

        # Step 2: Place IC block in hierarchical layers
        layer_hierarchy_2 = ["Parking", "WashArea", "Setback", "BathRoom", "Kitchen", "DiningRoom", "LivingRoom"]

        block_placed = False
        for layer in layer_hierarchy_2:
            if block_placed:
                break

            if layer in input_doc.layers:
                entities = [
                    e for e in input_msp if e.dxf.layer == layer and e.dxftype() == "MTEXT"
                ]
                if entities:
                    input_msp.add_blockref(
                        "IC", entities[0].dxf.insert, dxfattribs={'xscale': 1, 'yscale': 1}
                    )
                    block_placed = True
                else:
                    #print(f"No MTEXT in '{layer}', checking for lines.")
                    lines = [
                        e for e in input_msp if e.dxf.layer == layer and e.dxftype() in ["LINE", "LWPOLYLINE"]
                    ]
                    if lines:
                        bbox = {
                            "min_x": min(line.dxf.start.x for line in lines),
                            "max_x": max(line.dxf.end.x for line in lines),
                            "min_y": min(line.dxf.start.y for line in lines),
                            "max_y": max(line.dxf.end.y for line in lines),
                        }
                        center = ((bbox["min_x"] + bbox["max_x"]) / 2, (bbox["min_y"] + bbox["max_y"]) / 2)
                        input_msp.add_blockref(
                            "IC", center, dxfattribs={'xscale': 1, 'yscale': 1}
                        )
                        block_placed = True

        if not block_placed:
            print("No suitable placement found for IC block.")

        # Save the mzodified DXF file
        input_doc.saveas(output_dxf)
        #print(f"Blocks placed and saved to '{output_dxf}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


def main1(input_dxf1, block_file_IC_and_GT, output_dxf):
    copy_and_place_blocks(block_file_IC_and_GT, input_dxf1, output_dxf)


def read_dxf(file_path):
    try:
        doc = ezdxf.readfile(file_path)
        #print(f"DXF file '{file_path}' loaded successfully.")
        return doc
    except Exception as e:
        #print(f"Error reading DXF file: {e}")
        return None

# Step 4: Process DXF - Add FT Blocks
def process_dxf1(block_file_FT, input_dxf2, output_dxf):
    block_doc = read_dxf(block_file_FT)
    floor_plan_doc = read_dxf(input_dxf2)

    if not block_doc or not floor_plan_doc:
        return

    floor_plan_msp = floor_plan_doc.modelspace()

    # Copy FT block
    if "FT" not in block_doc.blocks:
        raise ValueError("FT block is missing in the block file.")

    if "FT" not in floor_plan_doc.blocks:
        ft_block = block_doc.blocks.get("FT")
        new_block = floor_plan_doc.blocks.new(name="FT")
        for entity in ft_block:
            new_block.add_entity(entity.copy())
    
        #print("FT block already exists in the floor plan.")

    # Identify Shower and Toilet sheet insertion points
    target_blocks = ["Shower", "Toilet sheet", "WC"]
    insertion_points = [
        (entity.dxf.insert, entity.dxf.name) for entity in floor_plan_msp.query("INSERT")
        if entity.dxf.name in target_blocks
    ]

    #print(f"Insertion Points for target blocks: {insertion_points}")

    # Offset placement (move FT block 10 units to the left of insertion points)
    offset_distance = 1  # Negative value moves left

    # Updated list of BathRoom layers
    bathroom_layers = [
        "BathRoom1", "BathRoom1a", "BathRoom2", "BathRoom2a", "BathRoom2b", "BathRoom2c",
        "BathRoom1", "BathRoom2", "BathRoom3", "BathRoom4", "Common_BathRoom",
        "Common_BathRoom1", "Common_BathRoom2", "Master_BathRoom", "Master_BathRoom1",
        "Master_BathRoom2", "Powder_Room", "Powder_Room1", "Powder_Room2"
        
    ]

    # Get BathRoom layer boundaries
    bathroom_lines = [
        line for line in floor_plan_msp.query("LINE") if line.dxf.layer in bathroom_layers
    ]

    if not bathroom_lines:
        #print("No lines found in BathRoom layers. Cannot place FT blocks.")
        return

    #print(f"Found {len(bathroom_lines)} lines in BathRoom layers.")

    def is_point_inside_bathroom(point):
        """Check if a point is inside the BathRoom layer boundaries."""
        intersections = 0
        test_x, test_y = point
        for line in bathroom_lines:
            x1, y1 = line.dxf.start.x, line.dxf.start.y
            x2, y2 = line.dxf.end.x, line.dxf.end.y

            if y1 == y2:  # Horizontal line
                if test_y == y1 and min(x1, x2) <= test_x <= max(x1, x2):
                    return True  # On the boundary
            elif min(y1, y2) < test_y <= max(y1, y2):
                x_intersect = x1 + (test_y - y1) * (x2 - x1) / (y2 - y1)
                if test_x < x_intersect:
                    intersections += 1

        return intersections % 2 == 1  # Odd intersections mean inside

    def find_valid_offset_point(point):
        """Find a valid offset point for the FT block inside the BathRoom boundaries."""
        for dx in range(-20, 21):
            for dy in range(-20, 21):
                offset_point = (point[0] + dx, point[1] + dy)
                if is_point_inside_bathroom(offset_point):
                    return offset_point
        return None

    # Place FT block for each target block's insertion point
    for point, block_name in insertion_points:
        #print(f"Processing {block_name} at {point}.")
        valid_point = find_valid_offset_point(point)

        if valid_point:
            #print(f"Placing FT block at {valid_point} near {block_name}.")
            floor_plan_msp.add_blockref("FT", valid_point, dxfattribs={"xscale": 0.02, "yscale": 0.02})
        
            #print(f"No valid position found for FT block near {block_name} at {point}. Skipping placement.")

    # Save the updated DXF
    floor_plan_doc.saveas(output_dxf)
    #print(f"Updated DXF saved as {output_dxf}")


def main2(input_dxf2, block_file_FT, output_dxf):
    process_dxf1(block_file_FT, input_dxf2, output_dxf)


def find_intersection_points(horizontal_lines, vertical_lines):
    """Find intersection points of horizontal and vertical lines."""
    intersection_points = []
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            intersect = intersection_line_line_2d(h_line, v_line)
            if intersect:
                intersection_points.append(Vec2(intersect))
    return intersection_points

def get_lines_in_layer(entities, layer_name):
    """Categorize lines into horizontal and vertical within the specified layer."""
    horizontal = []
    vertical = []
    for line in entities:
        if line.dxf.layer == layer_name:  # Ensure the line belongs to the given layer
            start, end = Vec2(line.dxf.start), Vec2(line.dxf.end)
            if abs(start.x - end.x) < 1e-6:  # Vertical line
                vertical.append((start, end))
            elif abs(start.y - end.y) < 1e-6:  # Horizontal line
                horizontal.append((start, end))
    return horizontal, vertical

def is_point_inside_boundary(point, horizontal_lines, vertical_lines):
    """Check if a point is inside the boundary defined by horizontal and vertical lines."""
    min_x = min(line[0].x for line in vertical_lines)
    max_x = max(line[0].x for line in vertical_lines)
    min_y = min(line[0].y for line in horizontal_lines)
    max_y = max(line[0].y for line in horizontal_lines)
    return min_x <= point.x <= max_x and min_y <= point.y <= max_y

def filter_by_distance(points, min_distance):
    """Filter points to ensure a minimum distance between them."""
    filtered_points = []
    for point in points:
        if not any((point - p).magnitude < min_distance for p in filtered_points):
            filtered_points.append(point)
    return filtered_points

def place_ft_blocks(layer_name, msp, ft_block, target_dxf, max_blocks=None, min_distance=None):
    """Place FT blocks in the specified layer within the layer boundary."""
    # Get all entities in the modelspace
    layer_entities = msp.query("LINE")

    # Categorize lines into horizontal and vertical, filtering by layer name
    horizontal_lines, vertical_lines = get_lines_in_layer(layer_entities, layer_name)

    # Find intersection points
    intersection_points = find_intersection_points(horizontal_lines, vertical_lines)

    # Filter intersection points to ensure they are inside the layer boundary
    valid_points = [
        point for point in intersection_points
        if is_point_inside_boundary(point, horizontal_lines, vertical_lines)
    ]

    # Apply minimum distance filter if specified
    if min_distance:
        valid_points = filter_by_distance(valid_points, min_distance)

    # Limit the number of blocks if max_blocks is specified
    if max_blocks is not None:
        valid_points = valid_points[:max_blocks]

    # Add FT block to the target DXF if not already present
    if "FT" not in target_dxf.blocks:
        target_block = target_dxf.blocks.new(name="FT")
        for entity in ft_block:
            target_block.add_entity(entity.copy())  # Clone the entity before adding

    # Place FT block at each valid intersection point
    for point in valid_points:
        block_ref = msp.add_blockref(
            name="FT",
            insert=(point.x, point.y),
            dxfattribs={"xscale": 1 / 25.4, "yscale": 1 / 25.4}  # Scale to 2 inches
        )

        # Ensure the block stays completely inside the boundary
        if not is_point_inside_boundary(Vec2(block_ref.dxf.insert), horizontal_lines, vertical_lines):
            msp.delete_entity(block_ref)  # Remove block if it is outside the boundary

def make_ft_placement(block_file_FT, input_dxf3, output_dxf, layers):
    """Perform FT block placement in the specified layers."""
    # Open the source DXF and get the block definition
    source_dxf = ezdxf.readfile(block_file_FT)
    ft_block = source_dxf.blocks.get("FT")

    # Open the target DXF
    target_dxf = ezdxf.readfile(input_dxf3)  # Corrected this line
    msp = target_dxf.modelspace()

    # Place FT blocks in each layer with specific rules
    for layer_name in layers:
        if layer_name in ["Balcony", "Mumty", "Terrace"]:  # Ensure layer name matches specified layers
            if layer_name == "Balcony":
                # Place up to 4 FT blocks with a minimum distance of 6 feet
                place_ft_blocks(layer_name, msp, ft_block, target_dxf, max_blocks=4, min_distance=6 * 12)
            elif layer_name == "Mumty":
                # Place up to 5 FT blocks with a minimum distance of 6 feet
                place_ft_blocks(layer_name, msp, ft_block, target_dxf, max_blocks=5, min_distance=6 * 12)
            elif layer_name == "Terrace":
                # Place FT blocks with a minimum 6 feet distance
                place_ft_blocks(layer_name, msp, ft_block, target_dxf, min_distance=6 * 12)  # 6 feet in inches

    # Save the modified DXF
    target_dxf.saveas(output_dxf)
    #print(f"Modified DXF saved as {output_dxf}")


def main3(input_dxf3, block_file_FT, output_dxf, layers):
    make_ft_placement(block_file_FT, input_dxf3, output_dxf, layers)

def add_arrow_near_ft(input_dxf4, output_dxf):
    try:
        # Load the DXF file
        doc = ezdxf.readfile(input_dxf4)
        msp = doc.modelspace()
        
        # Define the arrow size and offset
        arrow_length = 6.0  # Length of the arrow shaft
        arrow_head_size = 3.0  # Size of the arrowhead
        arrow_offset = 3.0  # Offset from the FT block

        # Find all `FT` blocks in the drawing
        for entity in msp.query('INSERT'):
            if entity.dxf.name.upper() == "FT":
                # Get the insertion point of the FT block
                x, y, z = entity.dxf.insert

                # Define arrow start and end points (diagonal line)
                arrow_start = (x + arrow_offset, y + arrow_offset)
                arrow_end = (x + arrow_offset + arrow_length, y + arrow_offset + arrow_length)

                # Add the arrow shaft (line)
                msp.add_line(arrow_start, arrow_end)

                # Define arrowhead points (triangle)
                arrowhead = [
                    (arrow_end[0], arrow_end[1]),  # Tip of the arrow
                    (arrow_end[0] - arrow_head_size, arrow_end[1] - 0.5 * arrow_head_size),  # Left side
                    (arrow_end[0] - arrow_head_size, arrow_end[1] + 0.5 * arrow_head_size),  # Right side
                ]
                msp.add_lwpolyline(arrowhead, close=True)

        # Save the modified DXF file
        doc.saveas(output_dxf)
        #print(f"Arrows added successfully. Output saved as '{output_dxf}'.")

    except Exception as e:
        print(f"An error occurred: {e}")

def main4(input_dxf4, output_dxf):
    add_arrow_near_ft(input_dxf4, output_dxf)

def copy_block_from_source_to_target(source_file, block_name, target_file):
    source_doc = ezdxf.readfile(source_file)
    target_doc = ezdxf.readfile(target_file)
    block = source_doc.blocks.get(block_name)
    if block_name not in target_doc.blocks:
        target_block = target_doc.blocks.new(name=block_name)
        for entity in block:
            target_block.add_entity(entity.copy())
    target_doc.saveas(target_file)

def find_midpoint_of_line(line):
    start = Vec2(line.dxf.start)
    end = Vec2(line.dxf.end)
    return (start + end) / 2

def find_nearest_layer_line(msp, target_layer, nearby_layers):
    target_lines = msp.query(f'LINE[layer=="{target_layer}"]')
    if not target_lines:
        return None
    target_midpoints = [find_midpoint_of_line(line) for line in target_lines]
    nearest_line = None
    min_distance = float("inf")
    for layer in nearby_layers:
        layer_lines = msp.query(f'LINE[layer=="{layer}"]')
        for line in layer_lines:
            line_midpoint = find_midpoint_of_line(line)
            for target_midpoint in target_midpoints:
                distance = (line_midpoint - target_midpoint).magnitude
                if distance < min_distance:
                    min_distance = distance
                    nearest_line = line
    if nearest_line:
        return find_midpoint_of_line(nearest_line)
    return None

def find_nearest_line_outside(msp, layer_name):
    layer_lines = msp.query(f'LINE[layer=="{layer_name}"]')
    if not layer_lines:
        return None
    all_lines = msp.query("LINE")
    nearest_line = None
    min_distance = float("inf")
    layer_midpoints = [find_midpoint_of_line(line) for line in layer_lines]
    for line in all_lines:
        if line.dxf.layer != layer_name:
            line_midpoint = find_midpoint_of_line(line)
            for layer_midpoint in layer_midpoints:
                distance = (line_midpoint - layer_midpoint).magnitude
                if distance < min_distance:
                    min_distance = distance
                    nearest_line = line
    if nearest_line:
        return find_midpoint_of_line(nearest_line)
    return None

def place_block1(msp, block_name, location, scale=(0.05, 0.05, 0.05)):
    if location:
        block_ref = msp.add_blockref(block_name, location)
        block_ref.dxf.xscale, block_ref.dxf.yscale, block_ref.dxf.zscale = scale

def process_dxf2(block_file_WPDT, input_dxf5, output_dxf):
    copy_block_from_source_to_target(block_file_WPDT, "WPDT", input_dxf5)
    doc = ezdxf.readfile(input_dxf5)
    msp = doc.modelspace()
    block_scale = (0.05, 0.05, 0.05)

    bathroom_layers = ["BathRoom1", "BathRoom2","BathRoom3", "BathRoom4", "Common_BathRoom", "Master_BathRoom",
                       "Powder_Room","Powder_Room1","Powder_Room2","Balcony","Balcony1", "Balcony2", "Balcony3","Terrace","Mumty"]
    
    nearby_layers = ["Garden", "Setback", "Passage", "WashArea", "Boundary", "Balcony", 
                 "Balcony1", "Balcony2", "Balcony3", "O.T.S", "O.T.S1", "O.T.S2", 
                    ]


    terrace_location = find_nearest_line_outside(msp,[ "Terrace","Mumty","Balcony"])
    if terrace_location:
        place_block1(msp, "WPDT", terrace_location, scale=block_scale)

    for layer in bathroom_layers:
        location = find_nearest_layer_line(msp, layer, nearby_layers)
        if location:
            place_block1(msp, "WPDT", location, scale=block_scale)
        else:
            location = find_nearest_line_outside(msp, layer)
            if location:
                place_block1(msp, "WPDT", location, scale=block_scale)

    doc.saveas(output_dxf)


def main5(input_dxf5, block_file_WPDT, output_dxf):
    process_dxf2(block_file_WPDT, input_dxf5, output_dxf)

def find_wpdt_blocks(msp, wpdt_block_name):
    """Find all WPDT block references in the model space."""
    wpdt_locations = []
    for block_ref in msp.query(f'INSERT[name=="{wpdt_block_name}"]'):
        wpdt_locations.append(block_ref.dxf.insert)
    return wpdt_locations

def find_line_for_wpdt(msp, wpdt_location):
    """Find a line near a WPDT block location."""
    line_entities = msp.query('LINE')
    for line in line_entities:
        line_start = Vec2(line.dxf.start)
        line_end = Vec2(line.dxf.end)

        # Check if the WPDT insert point lies on or near the line
        if line_start.x <= wpdt_location.x <= line_end.x or line_start.y <= wpdt_location.y <= line_end.y:
            # Check if WPDT is aligned with the line
            if abs((wpdt_location - line_start).magnitude + (wpdt_location - line_end).magnitude - (line_end - line_start).magnitude) < 1e-3:
                return line

    return None

def place_spdt_near_wpdt(msp, wpdt_block_name, spdt_block_name, distance_in_inches=6):
    """Place SPDT blocks near WPDT blocks."""
    wpdt_locations = find_wpdt_blocks(msp, wpdt_block_name)
    restricted_layers = ["Terrace", "Mumty", "Balcony"]

    for wpdt_location in wpdt_locations:
        wpdt_layer = None
        for block_ref in msp.query(f'INSERT[name=="{wpdt_block_name}"]'):
            if block_ref.dxf.insert == wpdt_location:
                wpdt_layer = block_ref.dxf.layer
                break

        if wpdt_layer in restricted_layers:
            continue

        line = find_line_for_wpdt(msp, Vec2(wpdt_location))
        if line:
            line_start = Vec2(line.dxf.start)
            line_end = Vec2(line.dxf.end)

            # Determine direction of the line
            direction = (line_end - line_start).normalize()

            # Calculate offset position
            offset = direction * distance_in_inches
            spdt_location = Vec2(wpdt_location.x, wpdt_location.y) + offset

            # Place the SPDT block
            msp.add_blockref(spdt_block_name, (spdt_location.x, spdt_location.y))

def define_spdt_block(doc, spdt_block_name):
    """Define the SPDT block if it does not already exist."""
    if spdt_block_name not in doc.blocks:
        spdt_block = doc.blocks.new(name=spdt_block_name)
        spdt_block.add_circle(center=(0, 0), radius=2)  # Example: a circle with radius 2
        spdt_block.add_text("SPDT", height=0.5, dxfattribs={"insert": (0, -2)})

def process_dxf3(doc, wpdt_block_name, spdt_block_name):
    """Process the DXF file by placing SPDT blocks near WPDT blocks."""
    msp = doc.modelspace()
    define_spdt_block(doc, spdt_block_name)
    place_spdt_near_wpdt(msp, wpdt_block_name, spdt_block_name, distance_in_inches=6)

def main6(input_dxf6, wpdt_block_name, spdt_block_name, output_dxf):
    """Main function to read, process, and save the DXF file."""
    try:
        doc = ezdxf.readfile(input_dxf6)
        process_dxf3(doc, wpdt_block_name, spdt_block_name)
        doc.saveas(output_dxf)
    except Exception as e:
        pass

def debug_layer_content(doc, layer_name):
    """Print all entities in a given layer for debugging."""
    msp = doc.modelspace()
    entities = msp.query(f'*[layer=="{layer_name}"]')

def place_septic_tank_block(doc, layers, block_size=50, offset=10):
    try:
        msp = doc.modelspace()
        layer_to_use = None

        # Iterate over layers in preference order
        for layer_name in layers:
            try:
                if doc.layers.get(layer_name):
                    lines = msp.query(f'LINE[layer=="{layer_name}"]')
                    if lines:  # Check if layer has any lines
                        layer_to_use = layer_name
                        break
            except Exception:
                continue  # Move to the next layer

        if not layer_to_use:
            return False

        # Find all lines in the selected layer
        lines = msp.query(f'LINE[layer=="{layer_to_use}"]')
        if not lines:
            return False

        # Separate vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []

        for line in lines:
            start = line.dxf.start
            end = line.dxf.end
            if start.x == end.x:  # Vertical line
                vertical_lines.append(line)
            elif start.y == end.y:  # Horizontal line
                horizontal_lines.append(line)

        if not vertical_lines or not horizontal_lines:
            return False

        # Find intersections between vertical and horizontal lines
        intersections = []
        for v_line in vertical_lines:
            v_x = v_line.dxf.start.x
            for h_line in horizontal_lines:
                h_y = h_line.dxf.start.y
                if (
                    min(h_line.dxf.start.x, h_line.dxf.end.x) <= v_x <= max(h_line.dxf.start.x, h_line.dxf.end.x)
                    and min(v_line.dxf.start.y, v_line.dxf.end.y) <= h_y <= max(v_line.dxf.start.y, v_line.dxf.end.y)
                ):
                    intersections.append((v_x, h_y))

        if not intersections:
            return False

        # Sort intersections to find the best placement
        intersections.sort(key=lambda point: (point[0], point[1]))
        for intersection in intersections:
            x, y = intersection

            # Determine the quadrant of the angle
            vertical_line = next(v for v in vertical_lines if v.dxf.start.x == x)
            horizontal_line = next(h for h in horizontal_lines if h.dxf.start.y == y)

            if vertical_line.dxf.start.y < y:  # Vertical line is below the intersection
                if horizontal_line.dxf.start.x < x:  # Horizontal line is to the left
                    insert_point = (x - block_size, y - block_size)  # Bottom-left
                else:  # Horizontal line is to the right
                    insert_point = (x, y - block_size)  # Bottom-right
            else:  # Vertical line is above the intersection
                if horizontal_line.dxf.start.x < x:  # Horizontal line is to the left
                    insert_point = (x - block_size, y)  # Top-left
                else:  # Horizontal line is to the right
                    insert_point = (x, y)  # Top-right

            # Check if the septic tank fits within this angle
            if (
                insert_point[0] >= min(horizontal_line.dxf.start.x, horizontal_line.dxf.end.x)
                and insert_point[1] >= min(vertical_line.dxf.start.y, vertical_line.dxf.end.y)
            ):
                break
        else:
            return False

        # Create or use the existing septic tank block
        block_name = "SepticTank"
        try:
            septic_block = doc.blocks.new(name=block_name)

            # Outer box
            outer_box = [
                (0, 0),
                (block_size, 0),
                (block_size, block_size),
                (0, block_size),
                (0, 0),
            ]
            septic_block.add_lwpolyline(
                outer_box,
                close=True,
                dxfattribs={"color": 5},
            )

            # Inner offset box
            offset_box = [
                (offset, offset),
                (block_size - offset, offset),
                (block_size - offset, block_size - offset),
                (offset, block_size - offset),
                (offset, offset),
            ]
            septic_block.add_lwpolyline(
                offset_box,
                close=True,
                dxfattribs={"color": 5},
            )

            # Adding "Septic Tank" text in the center of the block
            text = septic_block.add_text(
                "Septic Tank",
                dxfattribs={
                    "height": 3,
                    "color": 1,
                    "style": "STANDARD",
                },
            )
            text.dxf.insert = (block_size / 2, block_size / 2)  # Center position
            text.dxf.halign = 1  # Center alignment horizontally
            text.dxf.valign = 1  # Center alignment vertically

        except ValueError:
            pass

        # Insert the block into the layer
        msp.add_blockref(block_name, insert_point, dxfattribs={"layer": layer_to_use})

        return True

    except Exception:
        return False

def main7(input_dxf7, output_dxf, user_input1):
    if user_input1 == "yes":
        layers = ["Parking", "Garden", "Setback"]  # Preference order

        try:
            doc = ezdxf.readfile(input_dxf7)
            success = place_septic_tank_block(doc, layers)

            if success:
                doc.saveas(output_dxf)
        except Exception:
            pass
    elif user_input == "no":
        pass

# %%

def read_dxf_file(input_dxf8: str):
    """Read the DXF file and return the document and modelspace."""
    try:
        doc = ezdxf.readfile(input_dxf8)
        return doc, doc.modelspace()
    except Exception as e:
        raise Exception(f"Error reading DXF file: {e}")

def create_rain_water_layer(doc):
    """Create or retrieve the 'Rain_Water' layer."""
    if 'Rain_Water' not in doc.layers:
        doc.layers.new(name='Rain_Water', dxfattribs={'color': 5})  # Color 5 (blue)

def find_box_position(msp, layers_to_check):
    """Find the first available position from the specified layers."""
    for layer_name in layers_to_check:
        if layer_name in msp.doc.layers:
            for entity in msp.query(f'*[layer=="{layer_name}"]'):
                if entity.dxftype() in {'LINE', 'LWPOLYLINE', 'POLYLINE'}:
                    return entity.dxf.start  # Use start point as box position
    return None

def create_box(msp, box_position, box_size, line_thickness, layer_name='Rain_Water'):
    """Create an outer and inner box at the specified position."""
    x, y = box_position.x, box_position.y
    offset_size = 2  # 2-inch offset for the outer box
    outer_box_size = box_size + offset_size * 2

    # Outer box (double-line box)
    msp.add_lwpolyline(
        [(x - offset_size, y - offset_size),
         (x + box_size + offset_size, y - offset_size),
         (x + box_size + offset_size, y + box_size + offset_size),
         (x - offset_size, y + box_size + offset_size)],
        close=True,
        dxfattribs={'layer': layer_name, 'lineweight': line_thickness}
    )

    # Inner box
    msp.add_lwpolyline(
        [(x, y),
         (x + box_size, y),
         (x + box_size, y + box_size),
         (x, y + box_size)],
        close=True,
        dxfattribs={'layer': layer_name, 'lineweight': line_thickness}
    )

def add_text(msp, box_position, box_size, text_height, layer_name='Rain_Water'):
    """Add text in the center of the box."""
    x, y = box_position.x, box_position.y
    text_x = x + box_size / 2
    text_y = y + box_size / 2
    msp.add_text(
        "Rain Water Tank",
        dxfattribs={'layer': layer_name, 'height': text_height, 'rotation': 0}
    ).dxf.insert = (text_x, text_y)
    

def main8(input_dxf8, output_dxf, box_size_input, line_thickness_inch_input, text_height_input, user_input2):
    """Main function to manage the creation of a Rain Water Tank."""
    # Get user input for placing the Rain Water tank
    

    if user_input2 == 'yes':
        add_rain_water = True
    elif user_input2 == 'no':
        add_rain_water = False
    else:
        print("Invalid input, skipping Rain Water Tank placement.")
        add_rain_water = False

    if not add_rain_water:
        print("Rain Water Tank placement skipped.")
        return

    try:
        # Step 1: Read the DXF file
        doc, msp = read_dxf_file(input_dxf8)

        # Step 2: Create or get the 'Rain_Water' layer
        create_rain_water_layer(doc)

        # Step 3: Find a position for the box
        layers_to_check = ['Garden', 'Parking', 'WashArea']
        box_position = find_box_position(msp, layers_to_check)

        if box_position:
            # Step 4: Create the outer and inner boxes
            create_box(msp, box_position, box_size_input, line_thickness_inch_input)

            # Step 5: Add text in the center of the box
            add_text(msp, box_position, box_size_input, text_height_input)

            # Step 6: Save the DXF file
            doc.saveas(output_dxf)
        else:
            print("No suitable position found on any layer (Garden, Parking, WashArea).")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to check if a layer Name matches target categories (with typo tolerance)
def is_relevant_layer(layer_name, keywords, threshold=80):
    for keyword in keywords:
        if fuzz.partial_ratio(layer_name.lower(), keyword.lower()) >= threshold:
            return True
    return False

# Function to process the CSV and calculate rainwater harvesting
def calculate_rainwater_from_csv(csv_file, rainfall_mm, runoff_coefficient=0.85):
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return f"Error loading CSV: {e}"

    # Clean up column names (remove extra spaces or special characters)
    df.columns = df.columns.str.strip()

    # Ensure the required columns exist
    if 'Layer name' not in df.columns or 'Area in sqft' not in df.columns:
        return "The CSV must contain 'Layer name' and 'Area in sqft' columns."

    # Keywords to match relevant layers
    relevant_keywords = ["Terrace","Momty","Balcony","Balcony1","Balcony2"]

    # Filter relevant layers (handling typos and case sensitivity)
    df['Is Relevant'] = df['Layer name'].apply(lambda x: is_relevant_layer(str(x), relevant_keywords))

    # Sum the areas of relevant layers
    relevant_df = df[df['Is Relevant']]
    if relevant_df.empty:
        return "No relevant layers (terrace, mumty, balcony) found in the data."

    total_area_sqft = relevant_df['Area in sqft'].sum()

    # Convert area to square meters (1 sqft = 0.092903 sqm)
    total_area_sqm = total_area_sqft * 0.092903

    # Convert rainfall from mm to meters
    rainfall_m = rainfall_mm / 1000

    # Calculate rainwater harvested (liters)
    rainwater_harvested_liters = total_area_sqm * rainfall_m * runoff_coefficient * 1000

    # Prepare results
    results = {
        "Total Area (sqft)": total_area_sqft,
        "Total Area (sqm)": round(total_area_sqm, 2),
        "Rainfall (mm)": rainfall_mm,
        "Rainwater Harvested (liters)": round(rainwater_harvested_liters, 2)
    }

    return results

# Function to copy a block to specific layers in a DXF file
def copy_block_to_specific_layers(input_dxf9, block_dxf_path, target_layer_names, output_dxf):
    # Load the target DXF file
    doc = ezdxf.readfile(input_dxf9)
    msp = doc.modelspace()

    # Load the block from the block DXF file
    block_doc = ezdxf.readfile(block_dxf_path)
    block_name = 'RPDT'

    # Ensure the block is in the target document
    if block_name not in doc.blocks:
        source_block = block_doc.blocks.get(block_name)
        if source_block is None:
            raise ValueError(f"Block '{block_name}' not found in the block DXF file.")
        # Create a new block in the target document
        new_block = doc.blocks.new(name=block_name)
        # Copy entities from the source block
        for entity in source_block:
            new_block.add_entity(entity.copy())

    # Place block on one line per specified layer
    for layer_name in target_layer_names:
        # Query all lines from the specified layer
        lines_in_layer = msp.query(f'LINE[layer=="{layer_name}"]')
        if not lines_in_layer:
            continue

        # Select the first line from the layer
        chosen_line = lines_in_layer[0]
        start_point = chosen_line.dxf.start
        end_point = chosen_line.dxf.end

        # Calculate the center point of the line (midpoint)
        center_point = (
            (start_point.x + end_point.x) / 2,
            (start_point.y + end_point.y) / 2,
            (start_point.z + end_point.z) / 2,
        )

        # Insert the block at the center of the chosen line without scaling
        inserted_block = msp.add_blockref(block_name, center_point)

        # Set the scale to (1, 1, 1) explicitly to avoid resizing
        inserted_block.dxf.xscale = 0.01
        inserted_block.dxf.yscale = 0.01
        inserted_block.dxf.zscale = 0.01

        # Copy the layer and color attributes from the line to the block
        inserted_block.dxf.layer = chosen_line.dxf.layer
        inserted_block.dxf.color = chosen_line.dxf.color
        doc.saveas(output_dxf)

# Unified main function
def main9(input_dxf9,block_dxf_path,target_layer_names,output_dxf,csv_file_path,annual_rainfall_mm):
    # Rainwater harvesting calculation
    
    rainwater_output = calculate_rainwater_from_csv(csv_file_path, annual_rainfall_mm)

    if isinstance(rainwater_output, dict):
        print("Rainwater Harvesting Calculation:")
        for key, value in rainwater_output.items():
            print(f"{key}: {value}")
    else:
        print(rainwater_output)
    try:
        copy_block_to_specific_layers(input_dxf9, block_dxf_path, target_layer_names, output_dxf)
        #print(f"Block copied and saved to {output_dxf}")
    except Exception as e:
        print(f"Error copying block: {e}")

def process_dxf4(doc, output_dxf):
    try:
        msp = doc.modelspace()

        # Define layers with their respective colors
        layers = {
            "Soil Drain Pipe": 30,  # Orange for SPDT to Toilet
            "Waste Drain Pipe": 3,  # Green for WPDT to Fixtures
            "IC GT Connection": 1,  # Red for IC and GT connections
            "FT Connection": 2,  # Blue for FT to WPDT connections
        }

        # Create layers if they do not exist
        for layer_name, color in layers.items():
            if layer_name not in doc.layers:
                doc.layers.new(name=layer_name, dxfattribs={"color": color})

        # Helper function to draw arrows for flow
        def draw_arrow(start, end, layer):
            arrow_size = 2  # Smaller arrow size
            direction = (end[0] - start[0], end[1] - start[1])

            # Create arrow points
            if direction[0] != 0:  # Horizontal arrow
                mid_x = (start[0] + end[0]) / 2
                arrow_head = [
                    (mid_x, end[1]), 
                    (mid_x - arrow_size, end[1] - arrow_size), 
                    (mid_x + arrow_size, end[1] - arrow_size)
                ]
            else:  # Vertical arrow
                mid_y = (start[1] + end[1]) / 2
                arrow_head = [
                    (end[0], mid_y), 
                    (end[0] - arrow_size, mid_y - arrow_size), 
                    (end[0] + arrow_size, mid_y - arrow_size)
                ]

            # Draw the arrow line
            msp.add_line(start=start, end=end, dxfattribs={"layer": layer})

            # Draw the arrowhead
            msp.add_lwpolyline(arrow_head, close=True, dxfattribs={"layer": layer})

        # Helper function to find the nearest block
        def find_nearest_block(source, targets):
            nearest_block = None
            min_distance = float("inf")
            for target in targets:
                dx = abs(source.dxf.insert[0] - target.dxf.insert[0])
                dy = abs(source.dxf.insert[1] - target.dxf.insert[1])
                distance = dx + dy  # Manhattan distance for horizontal/vertical alignment
                if distance < min_distance:
                    min_distance = distance
                    nearest_block = target
            return nearest_block

        # Query blocks by name
        def get_blocks(name):
            return list(msp.query(f'INSERT[name=="{name}"]'))

        # Process SPDT connections (Orange lines)
        for spdt in get_blocks("SPDT"):
            nearest_toilet = find_nearest_block(spdt, get_blocks("Toilet sheet"))
            if nearest_toilet:
                # Draw horizontal and vertical lines with arrows
                horizontal_end = (spdt.dxf.insert[0], nearest_toilet.dxf.insert[1])
                draw_arrow(spdt.dxf.insert, horizontal_end, "Soil Drain Pipe")
                draw_arrow(horizontal_end, nearest_toilet.dxf.insert, "Soil Drain Pipe")
            else:
                print(f"SPDT at {spdt.dxf.insert} has no nearby Toilet block.")

        # Process WPDT connections (Green lines)
        for wpdt in get_blocks("WPDT"):
            fixtures = get_blocks("Shower") + get_blocks("WashBasin") + get_blocks("Sink") + get_blocks("FT")
            nearest_fixture = find_nearest_block(wpdt, fixtures)
            if nearest_fixture:
                # Draw horizontal and vertical lines with arrows
                horizontal_end = (wpdt.dxf.insert[0], nearest_fixture.dxf.insert[1])
                draw_arrow(wpdt.dxf.insert, horizontal_end, "Waste Drain Pipe")
                draw_arrow(horizontal_end, nearest_fixture.dxf.insert, "Waste Drain Pipe")
            else:
                print(f"WPDT at {wpdt.dxf.insert} has no nearby fixtures.")

        # Connect Shower and WashBasin to nearest FT block
        for block in get_blocks("Shower") + get_blocks("WashBasin"):
            nearest_ft = find_nearest_block(block, get_blocks("FT"))
            if nearest_ft:
                horizontal_end = (block.dxf.insert[0], nearest_ft.dxf.insert[1])
                draw_arrow(block.dxf.insert, horizontal_end, "Waste Drain Pipe")
                draw_arrow(horizontal_end, nearest_ft.dxf.insert, "Waste Drain Pipe")
            else:
                print(f"Block {block.dxf.name} has no nearby FT block.")

        # Connect Sink to nearest WPDT
        for sink in get_blocks("Sink"):
            nearest_wpdt = find_nearest_block(sink, get_blocks("WPDT"))
            if nearest_wpdt:
                horizontal_end = (sink.dxf.insert[0], nearest_wpdt.dxf.insert[1])
                draw_arrow(sink.dxf.insert, horizontal_end, "Waste Drain Pipe")
                draw_arrow(horizontal_end, nearest_wpdt.dxf.insert, "Waste Drain Pipe")
            else:
                print(f"Sink at {sink.dxf.insert} has no nearby WPDT block.")

        # Connect IC and GT blocks
        ic_blocks = get_blocks("IC")
        gt_blocks = get_blocks("GT")
        if ic_blocks and gt_blocks:
            for i, ic in enumerate(ic_blocks[:-1]):
                next_ic = ic_blocks[i + 1]
                horizontal_end = (ic.dxf.insert[0], next_ic.dxf.insert[1])
                draw_arrow(ic.dxf.insert, horizontal_end, "IC GT Connection")
                draw_arrow(horizontal_end, next_ic.dxf.insert, "IC GT Connection")

            for i, gt in enumerate(gt_blocks[:-1]):
                next_gt = gt_blocks[i + 1]
                horizontal_end = (gt.dxf.insert[0], next_gt.dxf.insert[1])
                draw_arrow(gt.dxf.insert, horizontal_end, "IC GT Connection")
                draw_arrow(horizontal_end, next_gt.dxf.insert, "IC GT Connection")

        # Process FT connections on Terrace, Mumty, and Balcony
        terrace_ft_blocks = get_blocks("FT")
        for ft_block in terrace_ft_blocks:
            if "Terrace" in ft_block.dxf.name or "Mumty" in ft_block.dxf.name or "Balcony" in ft_block.dxf.name:
                nearest_wpdt = find_nearest_block(ft_block, get_blocks("WPDT"))
                if nearest_wpdt:
                    draw_arrow(ft_block.dxf.insert, nearest_wpdt.dxf.insert, "FT Connection")
                else:
                    print(f"FT block at {ft_block.dxf.insert} has no nearby WPDT block.")

        # Save the modified DXF file
        #print(f"DXF processing complete. Saved as: {output_dxf}")

    except Exception as e:
        print(f"Error: {e}")

def main10(input_dxf10, output_dxf):
    doc = ezdxf.readfile(input_dxf10)
    process_dxf4(doc, output_dxf)  # Pass doc here
    doc.saveas(output_dxf)  # Save the modified document
    #print(f"DXF file processed and saved as '{output_dxf9}'.")

import ezdxf
from ezdxf.math import BoundingBox
from typing import Optional, Tuple
import logging
import warnings

# Filter out specific ezdxf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='ezdxf')

# Set up logging to only show errors
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def compute_block_extents(block) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute the extents of a block manually with improved error handling.
    """
    try:
        bbox = BoundingBox()
        for entity in block:
            if hasattr(entity, 'bbox'):
                entity_bbox = entity.bbox()
                if entity_bbox:
                    bbox.extend(entity_bbox)

        if bbox.has_data:
            return (bbox.extmin.x, bbox.extmin.y, bbox.extmax.x, bbox.extmax.y)
        return None
    except Exception as e:
        logger.error(f"Error computing block extents: {e}")
        return None

def safe_add_block_reference(msp, block_name: str, insert_point: Tuple[float, float], scale_factor: float) -> bool:
    """
    Safely add a block reference with error handling.
    """
    try:
        if block_name not in msp.doc.blocks:
            return False

        block_ref = msp.add_blockref(block_name, insert_point)
        block_ref.dxf.xscale = scale_factor
        block_ref.dxf.yscale = scale_factor
        return True
    except Exception:
        return False

def process_dxf5(input_file11: str, output_dxf: str, block_mapping: dict) -> bool:
    """
    Process the DXF file and create a legend table without warnings or info messages.
    """
    try:
        # Read the input file
        doc = ezdxf.readfile(input_file11)
        msp = doc.modelspace()

        # Find the (max x, min y) and (max x, max y) points from entities on the "Boundary" layer
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")

        for entity in msp.query('LINE[layer=="Boundary"]'):
            x1, y1 = entity.dxf.start.x, entity.dxf.start.y
            x2, y2 = entity.dxf.end.x, entity.dxf.end.y
            max_x = max(max_x, x1, x2)
            min_y = min(min_y, y1, y2)
            max_y = max(max_y, y1, y2)

        if max_x == float("-inf") or min_y == float("inf") or max_y == float("-inf"):
            logger.error("No valid entities found on the 'Boundary' layer.")
            return False

        # Calculate the offset for placing the table
        table_start_x = max_x + 5 * 12  # 5 feet offset to the right
        table_center_y = (min_y + max_y) / 2  # Center between min_y and max_y
        table_height = 40 * (len(block_mapping) + 2)  # Table height based on number of rows (+2 for header and legends)
        table_start_y = table_center_y + table_height / 2
        table_end_y = table_center_y - table_height / 2

        # Table parameters
        square_size = 40
        col_width = 60
        text_height = 3
        bold_text_height = 5
        legends_height = 15

        # Draw Legends box
        total_table_width = square_size + col_width
        legends_y = table_start_y

        msp.add_lwpolyline(
            [
                (table_start_x, legends_y),
                (table_start_x + total_table_width, legends_y),
                (table_start_x + total_table_width, legends_y + legends_height),
                (table_start_x, legends_y + legends_height),
                (table_start_x, legends_y),
            ],
            close=True,
        )

        # Add centered legends text
        center_x = table_start_x + total_table_width / 2
        center_y = legends_y + legends_height / 2
        legends_text = msp.add_text(
            "Legends",
            dxfattribs={
                'height': bold_text_height,
                'halign': ezdxf.const.CENTER,
                'valign': ezdxf.const.MIDDLE,
            },
        )
        legends_text.set_placement((center_x, center_y))

        # Draw table headers
        header_lines = [
            [(table_start_x, table_start_y),
             (table_start_x + square_size, table_start_y),
             (table_start_x + square_size, table_start_y - square_size),
             (table_start_x, table_start_y - square_size)],
            [(table_start_x + square_size, table_start_y),
             (table_start_x + square_size + col_width, table_start_y),
             (table_start_x + square_size + col_width, table_start_y - square_size),
             (table_start_x + square_size, table_start_y - square_size)]
        ]
        
        for points in header_lines:
            msp.add_lwpolyline(points, close=True)

        # Add header texts
        headers = [
            ("Block", table_start_x + square_size / 2),
            ("Block Names", table_start_x + square_size + col_width / 2),
        ]
        
        for text, x_pos in headers:
            header_text = msp.add_text(
                text,
                dxfattribs={
                    'height': bold_text_height,
                    'halign': ezdxf.const.CENTER,
                    'valign': ezdxf.const.MIDDLE,
                },
            )
            header_text.set_placement((x_pos, table_start_y - square_size / 2))

        # Process blocks
        current_y = table_start_y - square_size
        for block_name, (description, scale) in block_mapping.items():
            # Draw row cells
            for x_start in [table_start_x, table_start_x + square_size]:
                width = square_size if x_start == table_start_x else col_width
                msp.add_lwpolyline(
                    [
                        (x_start, current_y - square_size),
                        (x_start + width, current_y - square_size),
                        (x_start + width, current_y),
                        (x_start, current_y),
                        (x_start, current_y - square_size),
                    ],
                    close=True,
                )

            # Add block reference
            block_center = (
                table_start_x + square_size / 2,
                current_y - square_size / 2,
            )
            if not safe_add_block_reference(msp, block_name, block_center, scale):
                continue

            # Add description text using mtext for multiline support
            if '\n' in description:
                desc_text = msp.add_mtext(
                    description,
                    dxfattribs={
                        'char_height': text_height,
                        'attachment_point': 5,  # Middle center
                    }
                )
                desc_text.set_location((
                    table_start_x + square_size + col_width / 2,
                    current_y - square_size / 2,
                ))
            else:
                desc_text = msp.add_text(
                    description,
                    dxfattribs={
                        'height': text_height,
                        'halign': ezdxf.const.CENTER,
                        'valign': ezdxf.const.MIDDLE,
                    },
                )
                desc_text.set_placement((
                    table_start_x + square_size + col_width / 2,
                    current_y - square_size / 2,
                ))

            current_y -= square_size

        # Validate and save the DXF file
        try:
            doc.validate()  # Validate silently
        except ezdxf.lldxf.const.DXFError:
            pass  # Ignore validation errors
            
        doc.saveas(output_dxf)
        return True

    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False

def main11(input_file11, output_dxf,block_mapping):
    # Block mappings with scale factors
    block_mapping=block_mapping
   

    success = process_dxf5(input_file11, output_dxf, block_mapping)
    if success:
        print(f"Processed DXF file successfully. Output saved to {output_dxf}")
    else:
        print("Failed to process the DXF file.")


def main_final_plumbing_complete(
    user_input1,
    user_input2,
    input_dxf1,
    block_file_IC_and_GT,
    block_file_FT,
    block_file_WPDT,
    block_dxf_path,
    csv_file_path,
    annual_rainfall_mm,
    output_dxf,
    block_mapping
):
    """
    Main function to handle sequential DXF processing based on user input
    and clean up intermediate files.
    """
    # Take user inputs
    user_input1 = user_input1
    user_input2 = user_input2
    # Define intermediate files
    intermediate_files = [f"{i}.dxf" for i in range(1, 11)]
    current_file = input_dxf1

    # Run main functions based on user inputs
    main1(input_dxf1=current_file, block_file_IC_and_GT=block_file_IC_and_GT, output_dxf=intermediate_files[0])
    current_file = intermediate_files[0]

    main2(input_dxf2=current_file, block_file_FT=block_file_FT, output_dxf=intermediate_files[1])
    current_file = intermediate_files[1]

    main3(input_dxf3=current_file, block_file_FT=block_file_FT,
          output_dxf=intermediate_files[2], layers=["Mumty", "Terrace", "Balcony", "Balcony1", "Balcony2", "Balcony3"])
    current_file = intermediate_files[2]

    main4(input_dxf4=current_file, output_dxf=intermediate_files[3])
    current_file = intermediate_files[3]

    main5(input_dxf5=current_file, block_file_WPDT=block_file_WPDT, output_dxf=intermediate_files[4])
    current_file = intermediate_files[4]

    main6(input_dxf6=current_file, output_dxf=intermediate_files[5], wpdt_block_name="WPDT", spdt_block_name="SPDT")
    current_file = intermediate_files[5]

    # Handle septic tank placement
    if user_input1 == "yes":
        main7(input_dxf7=current_file, output_dxf=intermediate_files[6], user_input1=user_input1)
        current_file = intermediate_files[6]

    # Handle rainwater tank placement
    if user_input2 == "yes":
        main8(input_dxf8=current_file, output_dxf=intermediate_files[7],
              box_size_input=30.0, line_thickness_inch_input=0.2, text_height_input=3.0, user_input2=user_input2)
        current_file = intermediate_files[7]

        main9(input_dxf9=current_file, block_dxf_path=block_dxf_path,
              target_layer_names=["Terrace", "Momty", "Balcony", "Balcony1", "Balcony2"],
              output_dxf=intermediate_files[8], csv_file_path=csv_file_path, annual_rainfall_mm=annual_rainfall_mm)
        current_file = intermediate_files[8]

    # Final processing
    main10(input_dxf10=current_file, output_dxf=intermediate_files[9])
    current_file=intermediate_files[9]
    
    main11(input_file11=current_file, output_dxf=output_dxf,block_mapping=block_mapping)
    
    # Organize and delete intermediate files
    print("Processing completed. Files structure:")
    print({"Intermediate Files": intermediate_files})

    for file in intermediate_files[:-1]:  # Skip final output file
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"File not found: {file}")



# main_final_plumbing_complete(
#     input_dxf1="Plumbing_Drawing.dxf",
#     user_input1="yes",
#     user_input2="yes",
#     block_file_IC_and_GT="IC & GT_Blocks.dxf",
#     block_file_FT="Floor_Trap.dxf",
#     block_file_WPDT="Waste_Pipe.dxf",
#     block_dxf_path="Rain_Water_Pipe.dxf",
#     csv_file_path="indian_residential_layers.csv",
#     annual_rainfall_mm=800,
#     output_dxf="Plumbing_Complete.dxf",
#     block_mapping = {
#         "Inlet_Pipe": ("Inlet Pipe", 10),
#         "Outlet_Pipe": ("Outlet Pipe",2),
#         "MWSP": ("Main Water\nSupply Pipe", 0.1),
#         "WPDT": ("Waste Pipe\nDown Take", 0.1),
#         "SPDT": ("Soil Pipe\nDown Take", 7),
#         "RPDT": ("Rain Water\nDown Take", 0.1),
#         "IC": ("Inspection Chamber", 0.8),
#         "GT": ("Gully Trap", 0.8),
#         "FT": ("Floor Trap", 0.05)
#     }
# )





