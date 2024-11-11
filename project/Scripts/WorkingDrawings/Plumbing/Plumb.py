# %%
import ezdxf
import pandas as pd
from ezdxf.math import Vec3# %%
import sys
import os
import ezdxf
import matplotlib.pyplot as plt
from ezdxf import recover
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Step 1: Function to insert a block and set color to all entities inside the block
def insert_block_with_color(block_dxf, floor_plan_dxf, block_name, layer_insert_point, modelspace, placed_blocks, color, size_in_inches):
    if block_name in placed_blocks or layer_insert_point is None:
        return

    if block_name in block_dxf.blocks:
        # Create a new block definition in the floor plan if it does not exist
        if block_name not in floor_plan_dxf.blocks:
            floor_plan_dxf.blocks.new(name=block_name)
            for entity in block_dxf.blocks.get(block_name):
                new_entity = floor_plan_dxf.blocks.get(block_name).add_entity(entity.copy())
                if new_entity is not None:  # Only proceed if the entity was successfully added
                    new_entity.dxf.color = color

        # Scale and insert block reference in the main DXF modelspace
        scale_factor = size_in_inches
        modelspace.add_blockref(
            block_name,
            layer_insert_point,
            dxfattribs={
                'xscale': scale_factor,
                'yscale': scale_factor,
                'zscale': scale_factor
            }
        )
        placed_blocks.add(block_name)

# Function to process DXF with block placement and Excel data
def pipeline_main(input_file, output_filename, block_file, excel_file_path):
    # Load floor plan and block DXF files
    floor_plan_dxf = ezdxf.readfile(input_file)
    block_dxf = ezdxf.readfile(block_file)
    modelspace = floor_plan_dxf.modelspace()
    
    # Define block insertion points for layers
    layers_to_check = ["Garden", "Washarea", "Parking", "Setback"]
    mtext_insert_points = {
        layer: next((mtext.dxf.insert for mtext in modelspace.query('MTEXT') if mtext.dxf.layer == layer), None)
        for layer in layers_to_check
    }
    
    # Place blocks for Gully_Trap and Inspection_Chamber
    placed_blocks = set()
    for layer, insert_point in mtext_insert_points.items():
        if layer in ["Garden", "Washarea", "Setback"]:
            insert_block_with_color(block_dxf, floor_plan_dxf, 'Gully_Trap', insert_point, modelspace, placed_blocks, color=5, size_in_inches=3)
        elif layer in ["Parking", "Setback", "Garden"]:
            insert_block_with_color(block_dxf, floor_plan_dxf, 'Inspection_Chamber', insert_point, modelspace, placed_blocks, color=6, size_in_inches=4)
    
    # Load Excel data for plumbing materials
    df = pd.read_excel(excel_file_path)
    df.columns = df.columns.str.strip()
    df['Material_Name_lower'] = df['Material Name'].str.lower()
    
    # Count existing blocks
    block_counts = {name: len(list(modelspace.query(f'INSERT[name=="{name}"]'))) for name in ['Sink', 'WashBasin', 'Shower', 'Toilet sheet']}
    
    # Match block counts with material IDs from Excel data
    result = {}
    for block_name, count in block_counts.items():
        matched_rows = df[df['Material_Name_lower'].str.contains(block_name.lower())]
        if not matched_rows.empty:
            result[matched_rows.iloc[0]['Material ID']] = count

    # Save the modified floor plan DXF
    floor_plan_dxf.saveas(output_filename)
    print("Processed DXF file saved as:", output_filename)
    print("Material ID and Counts:", result)

# Step 2: Functions for drainage pipeline

# Function to find block insertion point by block name
def find_block_insert_point(block_name, modelspace):
    for entity in modelspace.query('INSERT'):
        if entity.dxf.name == block_name:
            return entity.dxf.insert
    return None

# Function to detect blocks by layer (e.g., Sink, Shower, WashBasin, Toilet_Sheet)
def find_fixture_blocks(layer_name, modelspace):
    insert_points = []
    for entity in modelspace.query('INSERT'):
        if entity.dxf.layer == layer_name:
            insert_points.append(entity.dxf.insert)
    return insert_points

# Function to check if a line segment intersects any entities in restricted layers
def is_path_blocked(start, end, modelspace, restricted_layers):
    for entity in modelspace.query('LINE ARC CIRCLE LWPOLYLINE POLYLINE'):
        if entity.dxf.layer in restricted_layers:
            if entity.dxf.start and entity.dxf.end:
                if do_lines_intersect(start, end, entity.dxf.start, entity.dxf.end):
                    return True
    return False

# Helper function to determine if two line segments intersect
def do_lines_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

# Function to draw a strictly horizontal and vertical line between two points
def draw_line_between_points(start_point, end_point, modelspace, color, restricted_layers):
    x1, y1 = start_point.x, start_point.y
    x2, y2 = end_point.x, end_point.y
    
    if x1 != x2 and y1 != y2:
        # Check if horizontal path from start to x2 is blocked
        horizontal_point = Vec3(x2, y1, 0)
        vertical_point = Vec3(x1, y2, 0)
        
        if not is_path_blocked(start_point, horizontal_point, modelspace, restricted_layers) and \
           not is_path_blocked(horizontal_point, end_point, modelspace, restricted_layers):
            # Draw horizontal and then vertical line
            modelspace.add_line(start=start_point, end=horizontal_point, dxfattribs={'color': color})
            modelspace.add_line(start=horizontal_point, end=end_point, dxfattribs={'color': color})
        elif not is_path_blocked(start_point, vertical_point, modelspace, restricted_layers) and \
             not is_path_blocked(vertical_point, end_point, modelspace, restricted_layers):
            # Draw vertical and then horizontal line
            modelspace.add_line(start=start_point, end=vertical_point, dxfattribs={'color': color})
            modelspace.add_line(start=vertical_point, end=end_point, dxfattribs={'color': color})
    elif x1 == x2 or y1 == y2:
        # Draw a straight line if start and end are already aligned horizontally or vertically
        if not is_path_blocked(start_point, end_point, modelspace, restricted_layers):
            modelspace.add_line(start=start_point, end=end_point, dxfattribs={'color': color})

# Main function to execute both steps

import io
import sys
import re
import ezdxf
# from your_drawing_module import pipeline_main, find_block_insert_point, find_fixture_blocks, draw_line_between_points

def complete_pipeline(input_dxf, output_filename, block_file, excel_file_path):
    # Create a string buffer to capture the print output
    captured_output = io.StringIO()
    sys.stdout = captured_output  # Redirect stdout to the buffer

    try:
        # Run Step 1 - Block Placement and Material Counts
        pipeline_main(input_dxf, output_filename, block_file, excel_file_path)
    finally:
        sys.stdout = sys.__stdout__  # Restore stdout to the default stream

    # Get the captured output
    output = captured_output.getvalue()
    print("Captured Output:", output)  # Optionally print captured output for debugging

    # Extract material counts from the captured output
    material_counts = extract_material_counts(output)

    # Load the modified floor plan with block placements for Step 2
    floor_plan_dxf = ezdxf.readfile(output_filename)
    modelspace = floor_plan_dxf.modelspace()

    # Detect insertion points for Gully_Trap and Inspection_Chamber
    gully_trap_point = find_block_insert_point('Gully_Trap', modelspace)
    inspection_chamber_point = find_block_insert_point('Inspection_Chamber', modelspace)

    # Detect blocks in Bathroom, Kitchen, and WashArea layers
    sink_points = find_fixture_blocks('Sink', modelspace)
    shower_points = find_fixture_blocks('Shower', modelspace)
    washbasin_points = find_fixture_blocks('WashBasin', modelspace)
    wc_points = find_fixture_blocks('WC', modelspace)

    # Define restricted layers
    restricted_layers = ['Staircase', 'Staircase_innerwall', 'Staircase_outerwall']

    # Draw waste water pipes (green) from Gully_Trap to Sink, Shower, WashBasin
    if gully_trap_point:
        for sink in sink_points:
            draw_line_between_points(gully_trap_point, sink, modelspace, color=3, restricted_layers=restricted_layers)
        for shower in shower_points:
            draw_line_between_points(gully_trap_point, shower, modelspace, color=3, restricted_layers=restricted_layers)
        for washbasin in washbasin_points:
            draw_line_between_points(gully_trap_point, washbasin, modelspace, color=3, restricted_layers=restricted_layers)
    else:
        print("Gully_Trap not found in the floor plan.")

    # Draw soil drain pipes (orange) from Inspection_Chamber to WC/Toilet_Sheet
    if inspection_chamber_point:
        for wc in wc_points:
            draw_line_between_points(inspection_chamber_point, wc, modelspace, color=1, restricted_layers=restricted_layers)
    else:
        print("Inspection_Chamber not found in the floor plan.")

    # Draw line from Gully_Trap to Inspection_Chamber (orange)
    if gully_trap_point and inspection_chamber_point:
        draw_line_between_points(gully_trap_point, inspection_chamber_point, modelspace, color=1, restricted_layers=restricted_layers)

    # Save the final DXF with drainage pipes
    final_output_file = 'Final_Floor_Plan_with_Drainage_Pipes.dxf'
    floor_plan_dxf.saveas(final_output_file)
    print(f"Drainage pipes drawn. File saved as {final_output_file}")
    
    return material_counts  # Return the captured material counts

def extract_material_counts(output):
    """
    Extract material counts from the captured output of the pipeline.
    This assumes the output follows a specific pattern.
    """
    material_counts = {}

    # Assuming the material counts are in the format: Material ID and Counts: {'P-026': 2, 'P-025': 2, ...}
    match = re.search(r"Material ID and Counts: ({.*})", output)
    if match:
        material_counts_str = match.group(1)
        # Convert the material counts string to a dictionary
        try:
            material_counts = eval(material_counts_str)
        except Exception as e:
            print(f"Error parsing material counts: {str(e)}")

    return material_counts


# if not auditor.has_errors:
#     fig = plt.figure()
#     ax = fig.add_axes([0, 0, 1, 1])
#     ctx = RenderContext(doc)
#     out = MatplotlibBackend(ax)
#     Frontend(ctx, out).draw_layout(doc.modelspace(), finalize=True)
    
   
#     png_file = os.path.splitext(dxf_file)[0] + '.png'
    
    
#     ezdxf.addons.drawing.matplotlib.qsave(layout=doc.modelspace(), 
#                                           filename=png_file, 
#                                           bg='#FFFFFF',  # White background color
#                                           dpi=720)


# img = plt.imread(png_file)


# plt.figure(figsize=(15, 15))  


# plt.imshow(img, alpha=1.0) 
# plt.axis('off')  
# plt.show()
