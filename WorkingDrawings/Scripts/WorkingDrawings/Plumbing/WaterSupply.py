
import ezdxf
import math
from ezdxf.math import Vec2
import os

def load_dxf_file(dxf_file):
    """
    Loads a DXF file and returns the document and modelspace.
    """
    try:
        doc = ezdxf.readfile(dxf_file)
        return doc, doc.modelspace()
    except Exception:
        return None, None

def ensure_linetype_exists(doc, linetype_name, pattern):
    """
    Ensures a specific linetype exists in the document, creates it if necessary.
    """
    if linetype_name not in doc.linetypes:
        doc.linetypes.new(name=linetype_name, dxfattribs={"pattern": pattern})

def find_valid_point_within_layer(msp, layer_name, size, offset):
    """
    Finds a valid point within the specified layer boundary for placing the tank.
    Ensures the shape is fully inside with the specified offset.
    """
    for entity in msp.query(f"LINE[layer=='{layer_name}']"):
        start = entity.dxf.start
        end = entity.dxf.end

        if start.y == end.y:  # Horizontal line
            mid_x = (start.x + end.x) / 2
            mid_y = start.y - size / 2 - offset
            return mid_x, mid_y

        elif start.x == end.x:  # Vertical line
            mid_x = start.x - size / 2 - offset
            mid_y = (start.y + end.y) / 2
            return mid_x, mid_y

    return None

def create_layer(doc, name, linetype="CONTINUOUS"):
    """
    Creates a new layer if it does not exist.
    """
    if name not in doc.layers:
        doc.layers.new(name=name, dxfattribs={"linetype": linetype})

def add_square_with_offset(msp, center, size, layer_name, line_type, thickness, label, offset=10):
    """
    Adds a square and an offset square to the DXF modelspace with specified attributes.
    """
    half_size = size / 2
    offset_half_size = (size - offset) / 2

    # Main square
    msp.add_lwpolyline(
        [
            (center[0] - half_size, center[1] - half_size),
            (center[0] + half_size, center[1] - half_size),
            (center[0] + half_size, center[1] + half_size),
            (center[0] - half_size, center[1] + half_size),
            (center[0] - half_size, center[1] - half_size),
        ],
        close=True,
        dxfattribs={
            'layer': layer_name,
            'linetype': line_type,
            'lineweight': thickness,
        }
    )

    # Offset square
    msp.add_lwpolyline(
        [
            (center[0] - offset_half_size, center[1] - offset_half_size),
            (center[0] + offset_half_size, center[1] - offset_half_size),
            (center[0] + offset_half_size, center[1] + offset_half_size),
            (center[0] - offset_half_size, center[1] + offset_half_size),
            (center[0] - offset_half_size, center[1] - offset_half_size),
        ],
        close=True,
        dxfattribs={
            'layer': layer_name,
            'linetype': line_type,
            'lineweight': thickness,
        }
    )

    # Add label
    msp.add_text(
        label,
        dxfattribs={
            'layer': layer_name,
            'height': 2,
            'insert': center
        }
    )

def add_circle_with_offset(msp, center, radius, layer_name, line_type, thickness, label, offset=2):
    """
    Adds a circle and an offset circle to the DXF modelspace with specified attributes.
    """
    # Main circle
    msp.add_circle(
        center=center,
        radius=radius,
        dxfattribs={
            'layer': layer_name,
            'linetype': line_type,
            'lineweight': thickness
        }
    )
    # Offset circle
    msp.add_circle(
        center=center,
        radius=radius - offset,
        dxfattribs={
            'layer': layer_name,
            'linetype': line_type,
            'lineweight': thickness
        }
    )
    # Add label
    msp.add_text(
        label,
        dxfattribs={
            'layer': layer_name,
            'height': 2,
            'insert': center
        }
    )

def find_underground_layer(doc):
    """
    Finds a valid layer for the Underground tank from the predefined list.
    """
    layers_to_check = ["Garden", "Garden1", "Garden2", "WashArea", "WashArea1", "WashArea2",
                       "Setback", "Setback1", "Setback2"]
    for layer_name in layers_to_check:
        if layer_name in [layer.dxf.name for layer in doc.layers]:
            return layer_name
    return None

def find_overhead_layer(doc):
    """
    Finds a valid layer for the Overhead tank in the priority order Mumty > Terrace > Mumty_Roof.
    """
    layers_to_check = ["Mumty", "Terrace", "Mumty_Roof"]
    for layer_name in layers_to_check:
        if layer_name in [layer.dxf.name for layer in doc.layers]:
            return layer_name
    return None

def place_underground_tank(msp, doc, size=8, offset=10):
    """
    Places the Underground tank on a suitable layer and creates a new 'Underground' layer.
    """
    underground_layer = find_underground_layer(doc)
    if not underground_layer:
        return

    placement_point = find_valid_point_within_layer(msp, underground_layer, size, offset)
    if placement_point:
        create_layer(doc, "Underground", linetype="DASHED")

        add_square_with_offset(
            msp, center=placement_point, size=size,
            layer_name="Underground", line_type="DASHED",
            thickness=76.2, label="Underground", offset=offset
        )

def place_overhead_tank(msp, doc, radius=25.5, offset=2):
    """
    Places the Overhead tank on a suitable layer and creates a new 'Overhead' layer.
    """
    overhead_layer = find_overhead_layer(doc)
    if not overhead_layer:
        return

    placement_point = find_valid_point_within_layer(msp, overhead_layer, radius, offset)
    if placement_point:
        create_layer(doc, "Overhead", linetype="CONTINUOUS")

        add_circle_with_offset(
            msp, center=placement_point, radius=radius,
            layer_name="Overhead", line_type="CONTINUOUS",
            thickness=60, label="Overhead", offset=offset
        )

def main1(input_dxf1, output_dxf):
    """
    Main function to place tanks and save the DXF file.
    """
    # Load the DXF file
    doc, msp = load_dxf_file(input_dxf1)
    if doc is None or msp is None:
        raise FileNotFoundError(f"Unable to load the DXF file: {input_dxf1}")
    
    # Ensure linetypes are available
    ensure_linetype_exists(doc, "DASHED", [10, 10])  # Dashed pattern for Underground
    ensure_linetype_exists(doc, "CONTINUOUS", [])    # Continuous pattern for Overhead

    # Create and place both tanks
    place_underground_tank(msp, doc, size=50, offset=5)
    place_overhead_tank(msp, doc, radius=25.5, offset=2)

    # Save the updated DXF file
    doc.saveas(output_dxf)
    return output_dxf


def copy_block_to_dxf(doc, block_doc, block_name):
    """Copy the block from the block file to the input DXF file."""
    block = block_doc.blocks.get(block_name)
    if block and block_name not in doc.blocks:
        new_block = doc.blocks.new(name=block_name)
        for entity in block:
            new_block.add_entity(entity.copy())

def find_lines_on_layer(layer, msp):
    """Find all lines in the specified layer."""
    return [line for line in msp.query('LINE') if line.dxf.layer == layer]

def calculate_line_center(line):
    """Calculate the center point of a line."""
    center_x = (line.dxf.start.x + line.dxf.end.x) / 2
    center_y = (line.dxf.start.y + line.dxf.end.y) / 2
    return center_x, center_y

def place_block1(msp, block_name, insert_point, layer_name):
    """Place a block in the model space at a specific point, ensuring it remains within the specified layer."""
    msp.add_blockref(
        block_name,
        insert_point,
        dxfattribs={
            'xscale': 0.04,  # Adjusted block size
            'yscale': 0.04,
            'rotation': 0.0,
            'layer': layer_name  # Ensure block placement on the correct layer
        },
    )

def process_dxf1(input_dxf2, block_file):
    """Process the input DXF file, copying blocks and placing them in specified layers."""
    # Load the DXF file
    doc = ezdxf.readfile(input_dxf2)
    msp = doc.modelspace()

    # Load the block file
    block_doc = ezdxf.readfile(block_file)

    try:
        # Copy MWSP block if not already present
        copy_block_to_dxf(doc, block_doc, "MWSP")
    except ValueError as e:
        print(f"Error loading MWSP block: {e}")
        return None

    # List of BathRoom layers
    bathroom_layers = [
        "BathRoom1", "BathRoom1a", "BathRoom2", "BathRoom2a", "BathRoom2b", "BathRoom2c",
        "BathRoom1", "BathRoom2", "BathRoom3", "BathRoom4", "Common_BathRoom",
        "Common_BathRoom1", "Common_BathRoom2", "Master_BathRoom", "Master_BathRoom1",
        "Master_BathRoom2", "Powder_Room", "Powder_Room1", "Powder_Room2"
    ]

    # Process each BathRoom layer
    for layer in bathroom_layers:
        if layer in [lyr.dxf.name for lyr in doc.layers]:
            lines = find_lines_on_layer(layer, msp)
            if lines:
                # Select the first line in the layer
                selected_line = lines[0]

                # Calculate the center of the selected line
                center_point = calculate_line_center(selected_line)

                # Place the MWSP block at the calculated center point
                place_block1(msp, "MWSP", center_point, layer)

    # Return the modified doc
    return doc

def main2(input_dxf2, block_file, output_dxf):
    """Main function to process the DXF file and save the changes."""
    # Process the DXF file
    doc = process_dxf1(input_dxf2, block_file)

    if doc is not None:
        # Save the modified DXF file in the main function
        doc.saveas(output_dxf)
        #print(f"Modified DXF saved as {output_dxf}.")
        return output_dxf
    else:
        print("Processing failed. DXF file not saved.")
        return None

def connect_mwsp_to_fixtures(input_dxf3):
    # Open the input DXF file
    doc = ezdxf.readfile(input_dxf3)
    msp = doc.modelspace()

    # Specify the block names for MWSP and fixtures
    mwsp_block_name = "MWSP"
    fixture_block_names = ["Toilet sheet", "Shower", "WashBasin", "WC"]

    # Function to calculate the Euclidean distance between two points
    def calculate_distance(point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    # Find all MWSP blocks
    mwsp_blocks = [
        entity for entity in msp.query("INSERT")
        if entity.dxf.name == mwsp_block_name
    ]

    # Find all fixture blocks (Toilet sheet, Shower, WashBasin)
    fixture_blocks = {
        name: [
            entity for entity in msp.query("INSERT")
            if entity.dxf.name == name
        ]
        for name in fixture_block_names
    }

    # Check if MWSP block exists
    if not mwsp_blocks:
        print("No MWSP blocks found.")
        return None

    # Check if fixture blocks exist
    if not any(fixture_blocks.values()):
        print("No fixture blocks found.")
        return None

    # Function to draw an arrow at the end of the line
    def draw_arrow(start, end, color):
        arrow_size = 5  # Adjust the size of the arrowhead
        # Calculate the direction of the arrow
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx**2 + dy**2)
        unit_dx = dx / length
        unit_dy = dy / length

        # Arrowhead points
        arrow_end1 = (end[0] - arrow_size * unit_dy, end[1] + arrow_size * unit_dx)
        arrow_end2 = (end[0] + arrow_size * unit_dy, end[1] - arrow_size * unit_dx)

        # Draw the main line
        msp.add_line(start=start, end=end, dxfattribs={"color": color, "lineweight": 60})

        # Draw the two arrowhead lines
        msp.add_line(end, arrow_end1, dxfattribs={"color": color, "lineweight": 60})
        msp.add_line(end, arrow_end2, dxfattribs={"color": color, "lineweight": 60})

    # Process each MWSP block
    for mwsp in mwsp_blocks:
        mwsp_location = mwsp.dxf.insert

        # Process each fixture and connect to the nearest one
        for fixture_name, fixtures in fixture_blocks.items():
            for fixture in fixtures:
                fixture_location = fixture.dxf.insert

                # Calculate the distance between MWSP and fixture
                distance = calculate_distance(mwsp_location, fixture_location)

                # Check if the fixture is within a reasonable distance
                if distance < 100:  # Adjust threshold distance if needed
                    # Draw a blue horizontal line from MWSP to the fixture's X-coordinate
                    draw_arrow(
                        start=(mwsp_location[0], mwsp_location[1]),
                        end=(fixture_location[0], mwsp_location[1]),
                        color=5  # Blue color for the line
                    )

                    # Draw a blue vertical line from the fixture's X-coordinate to the fixture's Y-coordinate
                    draw_arrow(
                        start=(fixture_location[0], mwsp_location[1]),
                        end=(fixture_location[0], fixture_location[1]),
                        color=5  # Blue color for the line
                    )

                    # Add red lines for Shower and WashBasin ensuring both lines are visible
                    if fixture_name in ["Shower", "WashBasin"]:
                        offset_x = 5  # Offset red line horizontally to separate from blue lines

                        # Draw a red horizontal line from MWSP to the fixture's X-coordinate
                        draw_arrow(
                            start=(mwsp_location[0] + offset_x, mwsp_location[1]),
                            end=(fixture_location[0] + offset_x, mwsp_location[1]),
                            color=1  # Red color for the line
                        )

                        # Draw a red vertical line from the fixture's X-coordinate to the fixture's Y-coordinate
                        draw_arrow(
                            start=(fixture_location[0] + offset_x, mwsp_location[1]),
                            end=(fixture_location[0] + offset_x, fixture_location[1]),
                            color=1  # Red color for the line
                        )
    # Return the modified doc
    return doc

# Define the main function
def main3(input_dxf3, output_dxf):
    # Process the DXF file
    doc = connect_mwsp_to_fixtures(input_dxf3)

    if doc is not None:
        # Save the modified DXF file
        doc.saveas(output_dxf)
        #print(f"Updated DXF saved as {output_dxf}.")
        return output_dxf
    else:
        print("Processing failed. DXF file not saved.")
        return None

def copy_block_from_source_to_target(source_file, block_name, target_file):
    """
    Copies a block from the source DXF file to the target DXF file.
    """
    source_doc = ezdxf.readfile(source_file)
    target_doc = ezdxf.readfile(target_file)

    block = source_doc.blocks.get(block_name)
    if block_name in target_doc.blocks:
        pass
    else:
        target_block = target_doc.blocks.new(name=block_name)
        for entity in block:
            target_block.add_entity(entity.copy())

    target_doc.saveas(target_file)
    return target_doc

def find_midpoint_of_line(line):
    """
    Finds the midpoint of a given line entity.
    """
    start = Vec2(line.dxf.start)
    end = Vec2(line.dxf.end)
    return (start + end) / 2

def find_nearest_line_outside(msp, layer_name):
    """
    Finds the nearest line outside the specified layer for placing a block.
    Returns the midpoint of the line.
    """
    layer_lines = msp.query(f'LINE[layer=="{layer_name}"]')

    if not layer_lines:
        return None

    all_lines = msp.query("LINE")
    nearest_line = None
    min_distance = float("inf")
    layer_midpoints = [find_midpoint_of_line(line) for line in layer_lines]

    for line in all_lines:
        if line.dxf.layer != layer_name:  # Only consider lines outside the layer
            line_midpoint = find_midpoint_of_line(line)
            for layer_midpoint in layer_midpoints:
                distance = (line_midpoint - layer_midpoint).magnitude
                if distance < min_distance:
                    min_distance = distance
                    nearest_line = line

    if nearest_line:
        return find_midpoint_of_line(nearest_line)

    return None

def find_layer_present(msp, layer_name):
    """
    Checks if a layer exists in the DXF file.
    """
    return bool(msp.query(f'LINE[layer=="{layer_name}"]'))

def place_block2(msp, block_name, location):
    """
    Places a block at a specific location in the model space.
    """
    if location:
        msp.add_blockref(block_name, location)

def process_dxf(input_dxf4, block_file_inlet, block_file_outlet):
    """
    Processes the input DXF file to copy and place blocks as per the logic.
    """
    # Copy blocks from source files
    input_doc = copy_block_from_source_to_target(block_file_inlet, "Inlet_Pipe", input_dxf4)
    input_doc = copy_block_from_source_to_target(block_file_outlet, "Outlet_Pipe", input_dxf4)

    # Reload input DXF after copying blocks
    input_doc = ezdxf.readfile(input_dxf4)
    msp = input_doc.modelspace()

    # Always place Inlet_Pipe on Terrace, regardless of Overhead layer
    terrace_location = find_nearest_line_outside(msp, "Terrace")
    if terrace_location:
        place_block2(msp, "Inlet_Pipe", terrace_location)

    # Placement logic for Inlet_Pipe near Kitchen or BathRoom layers
    kitchen_bathroom_layers = ["Kitchen", "BathRoom"]
    nearby_layers = ["Garden", "Garden1", "Garden2", "Pathway1", "WashArea", "Setback"]

    for layer in kitchen_bathroom_layers:
        location = find_nearest_layer_line(msp, layer, nearby_layers)
        if location:
            place_block2(msp, "Inlet_Pipe", location)
            break

    # Check if the Underground layer exists
    if find_layer_present(msp, "Underground"):
        underground_location = find_nearest_line_outside(msp, "Underground")
        if underground_location:
            place_block2(msp, "Outlet_Pipe", underground_location)

    # Placement logic for Outlet_Pipe
    bathroom_layers = ["BathRoom1", "BathRoom2"]
    kitchen_layers = ["Kitchen"]

    for layer in bathroom_layers + kitchen_layers:
        location_outside_nearby = find_nearest_layer_line(msp, layer, nearby_layers)
        if location_outside_nearby:
            place_block2(msp, "Outlet_Pipe", location_outside_nearby)

    return input_doc

def find_nearest_layer_line(msp, target_layer, nearby_layers):
    """
    Finds the nearest line in a nearby layer to the target layer.
    Returns the midpoint of the nearest line.
    """
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

def main4(input_dxf4, block_file_inlet, block_file_outlet, output_dxf):
    input_doc = process_dxf(input_dxf4, block_file_inlet, block_file_outlet)
    input_doc.saveas(output_dxf)
    return output_dxf

# Helper function to calculate the distance between two points
def distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Function to draw a line with an arrowhead and fixed lineweight
def draw_line_with_arrow(msp, start, end, color=5, lineweight=24.5):
    """Draws a line with an arrowhead at the end and a fixed lineweight."""
    # Draw the main line
    line = msp.add_line(start, end, dxfattribs={'color': color, 'lineweight': 20})

    # Calculate direction vector
    direction = (end[0] - start[0], end[1] - start[1])
    length = math.sqrt(direction[0]**2 + direction[1]**2)
    unit_dir = (direction[0] / length, direction[1] / length)

    # Arrowhead size
    arrow_size = 2

    # Perpendicular vector for arrowhead
    perp_dir = (-unit_dir[1], unit_dir[0])
    arrow_point1 = (
        end[0] - arrow_size * unit_dir[0] + arrow_size * perp_dir[0],
        end[1] - arrow_size * unit_dir[1] + arrow_size * perp_dir[1]
    )
    arrow_point2 = (
        end[0] - arrow_size * unit_dir[0] - arrow_size * perp_dir[0],
        end[1] - arrow_size * unit_dir[1] - arrow_size * perp_dir[1]
    )

    # Draw the arrowhead
    msp.add_lwpolyline([end, arrow_point1, arrow_point2, end], dxfattribs={'color': color, 'lineweight': lineweight})

def process_dxf5(doc):
    msp = doc.modelspace()

    # Find all relevant blocks and layers
    sink_blocks = [entity for entity in msp.query('INSERT') if entity.dxf.name == "Sink"]
    outlet_pipes = [entity for entity in msp.query('INSERT') if entity.dxf.name == "Outlet_Pipe"]
    mwsp_blocks = [entity for entity in msp.query('INSERT') if entity.dxf.name == "MWSP"]
    inlet_pipes = [entity for entity in msp.query('INSERT') if entity.dxf.name == "Inlet_Pipe"]
    overhead_circles = [
        entity for entity in msp.query('CIRCLE') 
        if entity.dxf.layer == "Overhead" and entity.dxf.layer != "Terrace"
    ]

    underground_layer = [
        entity for entity in msp.query('CIRCLE')
        if entity.dxf.layer == "Underground"
    ]

    # Connect Sink to the nearest MWSP or Outlet_Pipe
    if sink_blocks:
        for sink in sink_blocks:
            sink_position = (sink.dxf.insert.x, sink.dxf.insert.y)

            # Find nearest MWSP or Outlet_Pipe
            nearest_entity = None
            nearest_distance = float('inf')

            # Check MWSP blocks
            for mwsp in mwsp_blocks:
                mwsp_position = (mwsp.dxf.insert.x, mwsp.dxf.insert.y)
                dist = distance(sink_position, mwsp_position)
                if dist < nearest_distance:
                    nearest_entity = mwsp
                    nearest_distance = dist

            # Check Outlet_Pipes
            for outlet_pipe in outlet_pipes:
                outlet_position = (outlet_pipe.dxf.insert.x, outlet_pipe.dxf.insert.y)
                dist = distance(sink_position, outlet_position)
                if dist < nearest_distance:
                    nearest_entity = outlet_pipe
                    nearest_distance = dist

            # Draw horizontal and vertical lines to Sink
            if nearest_entity:
                nearest_position = (nearest_entity.dxf.insert.x, nearest_entity.dxf.insert.y)
                horizontal_end = (sink_position[0], nearest_position[1])
                draw_line_with_arrow(msp, nearest_position, horizontal_end, color=5)
                vertical_end = (sink_position[0], sink_position[1])
                draw_line_with_arrow(msp, horizontal_end, vertical_end, color=5)

    # Process Outlet_Pipe connections to MWSP
    if mwsp_blocks and outlet_pipes:
        for outlet_pipe in outlet_pipes:
            if outlet_pipe.dxf.layer == "Terrace":
                continue

            outlet_position = (outlet_pipe.dxf.insert.x, outlet_pipe.dxf.insert.y)

            nearest_mwsp = min(
                mwsp_blocks,
                key=lambda mwsp: distance(outlet_position, (mwsp.dxf.insert.x, mwsp.dxf.insert.y))
            )
            mwsp_position = (nearest_mwsp.dxf.insert.x, nearest_mwsp.dxf.insert.y)

            horizontal_end = (mwsp_position[0], outlet_position[1])
            draw_line_with_arrow(msp, outlet_position, horizontal_end, color=5)
            vertical_end = (mwsp_position[0], mwsp_position[1])
            draw_line_with_arrow(msp, horizontal_end, vertical_end, color=5)

    # Process connections to Overhead circles
    if overhead_circles:
        for circle in overhead_circles:
            circle_position = (circle.dxf.center.x, circle.dxf.center.y)

            if outlet_pipes:
                nearest_outlet_pipe = min(
                    outlet_pipes,
                    key=lambda outlet_pipe: distance(
                        (outlet_pipe.dxf.insert.x, outlet_pipe.dxf.insert.y),
                        circle_position
                    )
                )
                outlet_position = (nearest_outlet_pipe.dxf.insert.x, nearest_outlet_pipe.dxf.insert.y)
                draw_line_with_arrow(msp, (circle_position[0], circle_position[1]), (outlet_position[0], circle_position[1]), color=5)
                draw_line_with_arrow(msp, (outlet_position[0], circle_position[1]), outlet_position, color=5)

            if inlet_pipes:
                nearest_inlet_pipe = min(
                    inlet_pipes,
                    key=lambda inlet_pipe: distance(
                        (inlet_pipe.dxf.insert.x, inlet_pipe.dxf.insert.y),
                        circle_position
                    )
                )
                inlet_position = (nearest_inlet_pipe.dxf.insert.x, nearest_inlet_pipe.dxf.insert.y)
                draw_line_with_arrow(msp, (inlet_position[0], inlet_position[1]), (circle_position[0], inlet_position[1]), color=6)
                draw_line_with_arrow(msp, (circle_position[0], inlet_position[1]), circle_position, color=6)

            horizontal_end = (circle_position[0] + 10, circle_position[1])
            draw_line_with_arrow(msp, circle_position, horizontal_end, color=5)

            vertical_end = (circle_position[0], circle_position[1] + 10)
            draw_line_with_arrow(msp, circle_position, vertical_end, color=5)

    # Process connections to Underground layer
    if underground_layer and inlet_pipes:
        for circle in underground_layer:
            circle_position = (circle.dxf.center.x, circle.dxf.center.y)

            nearest_inlet_pipe = min(
                inlet_pipes,
                key=lambda inlet_pipe: distance(
                    (inlet_pipe.dxf.insert.x, inlet_pipe.dxf.insert.y),
                    circle_position
                )
            )
            inlet_position = (nearest_inlet_pipe.dxf.insert.x, nearest_inlet_pipe.dxf.insert.y)
            draw_line_with_arrow(msp, (inlet_position[0], inlet_position[1]), (circle_position[0], inlet_position[1]), color=6)
            draw_line_with_arrow(msp, (circle_position[0], inlet_position[1]), circle_position, color=6)

def main5(input_dxf5, output_dxf):
    doc = ezdxf.readfile(input_dxf5)
    process_dxf5(doc)
    doc.saveas(output_dxf)
    return f"DXF file saved as {output_dxf}"

def main_final_water(input_dxf1, mwsp_block, inlet_block, outlet_block, final_output):
    """
    Main function to handle sequential DXF processing and clean up intermediate files.

    Args:
        input_dxf1 (str): The initial input DXF file.
        mwsp_block (str): DXF file containing MWSP pipe block definitions.
        inlet_block (str): DXF file containing inlet pipe block definitions.
        outlet_block (str): DXF file containing outlet pipe block definitions.
        final_output (str): Final output DXF file name.
    """
    intermediate_files = ["1.dxf", "2.dxf", "3.dxf", "4.dxf"]

    # Sequential processing using provided main functions
    main1(input_dxf1=input_dxf1, output_dxf=intermediate_files[0])
    main2(input_dxf2=intermediate_files[0], block_file=mwsp_block, output_dxf=intermediate_files[1])
    main3(input_dxf3=intermediate_files[1], output_dxf=intermediate_files[2])
    main4(input_dxf4=intermediate_files[2], block_file_inlet=inlet_block, block_file_outlet=outlet_block, output_dxf=intermediate_files[3])
    main5(input_dxf5=intermediate_files[3], output_dxf=final_output)

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
# main_final(
#     input_dxf1="Betatest1 (2).dxf",
#     mwsp_block="MWSP Pipe block.dxf",
#     inlet_block="Inlet_pipe.dxf",
#     outlet_block="Outlet_pipe.dxf",
#     final_output="Plumbing_Drawing.dxf"
# )

