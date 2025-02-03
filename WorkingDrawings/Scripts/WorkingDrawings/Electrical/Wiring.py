
import math  
from math import atan2, degrees  
import logging 
import re  
import ezdxf
from ezdxf.math import BoundingBox
from typing import Optional, Tuple
import logging
import warnings
import numpy as np  
import pandas as pd  
from shapely.geometry import (
    Polygon, 
    LineString, 
    MultiPolygon, 
    Point, 
    JOIN_STYLE
)  
from shapely.ops import unary_union, nearest_points 
import ezdxf 
from ezdxf.math import (
    area as calculate_polyline_area, 
    Vec3 )
from ezdxf.addons.drawing import (
    RenderContext, 
    Frontend
) 
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend  
import matplotlib.pyplot as plt 

# Parallel Execution
from concurrent.futures import ThreadPoolExecutor  # Provides a high-level interface for parallel execution using threads.

# Additional geometric operations
from itertools import product  # Generates the Cartesian product of input iterables, useful for creating combinations of values.
import sys
from contextlib import redirect_stdout
from io import StringIO
import os

# %%
"""
Defines layers to avoid and maps them to their potential typos using a dictionary.

Data Domain: CAD Layer Management
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

# List of layers to avoid crossing, defined with their standard names
avoid_layers = [
    "OTS", "Boundary", "Garden", "Setback", "Entrance",
    "Staircase", "Washarea", "Doors"
]

"""
Defines a dictionary mapping each layer name to its potential typos.

Key:
    - Each key is a standard layer name.
Value:
    - A list of potential typos or alternative spellings for that layer name.
"""
typos = {
    "OTS": ["otS", "Ost", "OtS", "ots", "OsT", "Ot5", "QT5", "OT", "O T S", "OTS_"],
    "Boundary": ["boundary", "Boundry", "Bounary", "Buondary", "BOUNDARY", "Boundray", 
                 "Bondary", "Boundayr", "Bounday", "B0undary"],
    "Garden": ["garden", "Gardne", "Gaden", "GARDEN", "Gardrn", "Gardan", "Gard3n", 
               "G@rden", "Gardenn", "Graden"],
    "Setback": ["setback", "Setbak", "Steback", "SETBACK", "Setbck", "S3tback", 
                "Setbackk", "S3tbak", "Set Back", "Seetback"],
    "Entrance": ["entrance", "Entrnace", "Entrence", "ENTRANCE", "Entrans", "Entr@ance", 
                 "Entrnace", "Entrence", "Entrrnce", "Etnrance"],
    "Staircase": ["staircase", "Stiarcase", "Staiircase", "STAIRCASE", "Staicase", 
                  "Staircsae", "St@ircase", "Stai1case", "Starcase", "Stari-case"],
    "Washarea": ["washarea", "Washrea", "Waasharea", "WASHAREA", "Washar3a", 
                 "WashARea", "Wasarea", "Washarea_", "Washaerea", "W@sharea"],
    "Doors": ["doors", "Dorrs", "Doros", "DOORS", "Dorrs", "Do0rs", "D0ors", 
              "Do-ors", "Door", "Dorrrs"],
    "BOUNDARY": ["boundary", "Boundry", "Boundayr", "BoUNDARY", "BOUDNARY", "B0UNDARY", 
                 "BOUNDAR", "BOUND@RY", "BOOUNDARY", "BOUND-ARY"],
    "GARDEN": ["garden", "Gardrn", "GARDN", "GARDEEN", "G@RDEN", "GARD3N", 
               "GRDEN", "GRADEN", "GARDNE", "GAR DEN"]
}
"""
Expands a list of layer names to include their potential typos based on a given dictionary.

Parameters:
    avoid_layers (list): A list of layer names to avoid.
    typo_dict (dict): A dictionary mapping layer names to lists of potential typos.

Returns:
    list: A list of layer names including the original layers and their potential typos.

Function Domain: CAD Layer Management
Function Name: get_layers_with_typos
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

# Function to map layers to their typos
def get_layers_with_typos(avoid_layers, typo_dict):
    # Start with the original layers as a set
    expanded_layers = set(avoid_layers)  
    
    # Iterate through each layer in the list of avoid layers
    for layer in avoid_layers:
        # If the layer exists in the typo dictionary, add its typos to the set
        if layer in typo_dict:
            expanded_layers.update(typo_dict[layer])
    
    # Return the expanded set of layers as a list
    return list(expanded_layers)

# Replace avoid_layers dynamically to include their typos
# avoid_layers: List of original layers to avoid
# typos: Dictionary of potential typos for the layers
avoid_layers_with_typos = get_layers_with_typos(avoid_layers, typos)
"""
Detects blocks from the specified block names in the given DXF document.

Parameters:
    doc (ezdxf.document): The DXF document to search for blocks.
    block_names (list): A list of block names to detect in the document.

Returns:
    list: A list of dictionaries where each dictionary represents a detected block and contains:
        - 'name' (str): The name of the block.
        - 'position' (tuple): The (x, y) coordinates of the block's insertion point.
        - 'entity' (ezdxf.entities.Insert): The block entity object.

Function Domain: CAD Geometry, Block Detection
Function Name: detect_blocks
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

def detect_blocks(doc, block_names):
    """Detect all blocks from the specified block names."""
    blocks = []  # Initialize an empty list to store detected blocks
    
    # Iterate through all INSERT entities in the modelspace
    for entity in doc.modelspace().query('INSERT'):
        # Check if the block name matches any of the specified block names
        if entity.dxf.name in block_names:
            blocks.append({
                'name': entity.dxf.name,  # Block name
                'position': (entity.dxf.insert.x, entity.dxf.insert.y),  # Insertion point coordinates
                'entity': entity  # The block entity object
            })
    
    # Return the list of detected blocks
    return blocks
"""
Detects all closed polylines in the DXF document and returns them as Shapely Polygon objects.

Parameters:
    doc (ezdxf.document): The DXF document to search for closed polylines.

Returns:
    list: A list of Shapely Polygon objects representing the closed polylines.

Function Domain: CAD Geometry, Polyline Detection
Function Name: detect_polylines
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

def detect_polylines(doc):
    """Detect all closed polylines and return them as Shapely Polygon objects."""
    polylines = []  # Initialize an empty list to store valid polygons
    
    # Iterate through all LWPOLYLINE entities in the modelspace
    for entity in doc.modelspace().query('LWPOLYLINE'):
        # Check if the polyline is closed
        if entity.closed:
            # Get the coordinates of the polyline's points
            polyline_coords = [(point[0], point[1]) for point in entity.get_points()]
            
            # Create a Shapely Polygon object from the coordinates
            polyline = Polygon(polyline_coords)
            
            # Ensure the polygon is valid (no self-intersections or invalid geometry)
            if polyline.is_valid:
                polylines.append(polyline)  # Append valid polygons to the list
    
    # Return the list of valid closed polylines as Shapely Polygon objects
    return polylines
"""
Groups blocks to their respective 'M_SW' (Switch) block or the nearest 'M_SW' block based on polyline containment.

Parameters:
    blocks (list): A list of blocks, each represented by a dictionary containing block name, position, and entity.
    polylines (list): A list of Shapely Polygon objects representing closed polylines.

Returns:
    dict: A dictionary where keys are the 'M_SW' block entities, and values are dictionaries containing:
        - 'M_SW' (dict): The 'M_SW' block information.
        - 'blocks' (list): A list of blocks grouped to this 'M_SW' block.

Function Domain: CAD Geometry, Block Grouping
Function Name: group_blocks_to_M_SW
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

def group_blocks_to_M_SW(blocks, polylines):
    """Group blocks to their respective M_SW or nearest M_SW."""
    grouped = {}  # Initialize an empty dictionary to store grouped blocks
    
    # Separate M_SW blocks from other blocks (e.g., SM_L, Fan_C, CHND_L)
    M_SW_blocks = [b for b in blocks if b['name'] == 'M_SW']
    other_blocks = [b for b in blocks if b['name'] in {'SM_L', 'Fan_C', 'CHND_L'}]

    # Iterate over the other blocks and assign them to an M_SW
    for block in other_blocks:
        block_point = Point(block['position'])  # Convert the block position to a Shapely Point

        # Try to find the containing polyline for the block, using a buffer to handle precision issues
        containing_polyline = None
        for poly in polylines:
            if poly.buffer(0.01).contains(block_point):
                containing_polyline = poly  # Set the containing polyline if block is inside
                break

        # If the block is inside a polyline, attempt to find an M_SW inside the same polyline
        if containing_polyline:
            # Identify all M_SW blocks inside the buffered polyline
            M_SW_in_polyline = [
                M_SW for M_SW in M_SW_blocks if containing_polyline.buffer(0.01).contains(Point(M_SW['position']))
            ]

            # If M_SW blocks are found inside, choose the first one
            if M_SW_in_polyline:
                nearest_M_SW = M_SW_in_polyline[0]
            else:
                # No M_SW inside, fallback to nearest M_SW globally
                nearest_M_SW = min(M_SW_blocks, key=lambda M_SW: block_point.distance(Point(M_SW['position'])))
        else:
            # If no polyline contains the block, fallback to nearest M_SW globally
            nearest_M_SW = min(M_SW_blocks, key=lambda M_SW: block_point.distance(Point(M_SW['position'])))

        # Group the block under the identified nearest M_SW block
        if nearest_M_SW['entity'] not in grouped:
            grouped[nearest_M_SW['entity']] = {'M_SW': nearest_M_SW, 'blocks': []}
        grouped[nearest_M_SW['entity']]['blocks'].append(block)

    return grouped
"""
Generates points along a cubic Bézier curve based on four control points.

Parameters:
    p0 (array-like): The first control point (start of the curve).
    p1 (array-like): The second control point.
    p2 (array-like): The third control point.
    p3 (array-like): The fourth control point (end of the curve).
    n_points (int, optional): The number of points to generate along the curve. Default is 100.

Returns:
    np.ndarray: A 2D array of points representing the cubic Bézier curve, with shape (n_points, 2).

Function Domain: Computational Geometry, Curve Generation
Function Name: cubic_bezier_curve
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""

def cubic_bezier_curve(p0, p1, p2, p3, n_points=100):
    """Generate points for a cubic Bézier curve."""
    # Generate 'n_points' evenly spaced values of t in the range [0, 1]
    t = np.linspace(0, 1, n_points).reshape(-1, 1)  # Reshape t for broadcasting in the equation
    
    # Ensure that control points are NumPy arrays for vectorized operations
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))  
    
    # Compute the cubic Bézier curve using the Bernstein polynomial formula
    curve = (1 - t) ** 3 * p0 + \
            3 * (1 - t) ** 2 * t * p1 + \
            3 * (1 - t) * t ** 2 * p2 + \
            t ** 3 * p3

    # Return the generated points as a 2D array
    return curve
"""
Connects blocks to their respective 'M_SW' block using curved S-lines (Bézier curves), avoiding specified layers.

Parameters:
    doc (ezdxf.document): The DXF document containing blocks, lines, and other entities.
    grouped (dict): A dictionary where keys are 'M_SW' block entities, and values are dictionaries containing:
        - 'M_SW' (dict): The 'M_SW' block information.
        - 'blocks' (list): A list of blocks to connect to this 'M_SW' block.
    avoid_layers (list): A list of layer names to avoid when drawing the connection lines.

Returns:
    list: A list of Shapely LineString objects representing the successfully generated Bézier curves (S-lines).

Function Domain: CAD Geometry, Block Connectivity, Pathfinding
Function Name: connect_blocks_to_M_SW
Creator Name: Subhrajyoti Mazumder
Date of Creation: 2025-01-07
"""
def connect_blocks_to_M_SW(doc, grouped, avoid_layers):
    """Connect blocks to their respective M_SW block with only curved S-Lines, avoiding specified layers."""
    # Collect lines from avoid layers (lines that should not be intersected by the connection)
    avoid_lines = []
    for layer in avoid_layers:
        for line in doc.modelspace().query(f'LINE[layer=="{layer}"]'):
            avoid_lines.append(
                LineString([(line.dxf.start.x, line.dxf.start.y), (line.dxf.end.x, line.dxf.end.y)])
            )

    s_lines = []  # List to store successfully generated Bézier curves

    # Iterate through each group of blocks and their respective 'SW' block
    for group in grouped.values():
        M_SW = group['M_SW']
        M_SW_point = Point(M_SW['position'])  # 'M_SW' block position as a Shapely Point

        # Connect each block to the 'M_SW' block
        for block in group['blocks']:
            block_point = Point(block['position'])  # Block position as a Shapely Point

            # Attempt to create a connection using Bézier curves
            successful_connection = False
            attempts = 0
            max_attempts = 20  # Maximum retries to avoid infinite loops

            while not successful_connection and attempts < max_attempts:
                # Adjust control points for the Bézier curve
                mid_x = (M_SW_point.x + block_point.x) / 2
                mid_y = (M_SW_point.y + block_point.y) / 2
                offset = 10 + attempts * 5  # Increase the offset for longer paths with each attempt
                control1 = (M_SW_point.x, mid_y + offset)
                control2 = (block_point.x, mid_y - offset)

                # Generate the Bézier curve between 'M_SW' and the block
                curve_points = cubic_bezier_curve(
                    p0=M_SW_point.coords[0],
                    p1=control1,
                    p2=control2,
                    p3=block_point.coords[0]
                )
                        # Generate a Bézier curve connecting the block and 'M_SW'
                bezier_line = LineString(curve_points)

                # Check if the curve intersects any lines from the avoid layers
                intersects_any = any(bezier_line.intersects(line.buffer(0.01)) for line in avoid_lines)

                if not intersects_any:
                    # If the curve does not intersect any avoid lines, add it to the document
                    doc.modelspace().add_lwpolyline(
                        curve_points,
                        dxfattribs={'color': 3}  # Set the polyline color to green
                    )
                    s_lines.append(bezier_line)  # Add the successfully generated curve to the list
                    successful_connection = True
                else:
                    attempts += 1  # Retry with adjusted control points if intersection occurs

    return s_lines  # Return the list of successfully generated S-lines (curved connections)

def main1(
    input_file,
    output_path_final,
    
):
    try:
        # Step 1: Load the target and source DXF files
        target_dxf = ezdxf.readfile(input_file)  # Load architectural drawing

        # Step 16: Save the modified DXF file
        target_dxf.saveas(output_path_final)  # Save the modified DXF file

        # Additional Steps for Processing Blocks and Lines
        doc = ezdxf.readfile(output_path_final)  # Reload the saved DXF file
        block_names = ['SM_L', 'M_SW', 'Fan_C', 'CHND_L']

        # Step A: Detect blocks
        blocks = detect_blocks(doc, block_names)

        # Step B: Detect polylines
        polylines = detect_polylines(doc)

        # Step C: Group blocks
        grouped = group_blocks_to_M_SW(blocks, polylines)

        # Step D: Connect blocks to M_SW
        connect_blocks_to_M_SW(doc, grouped, avoid_layers_with_typos)

        # Save the modified DXF again
        doc.saveas(output_path_final)

        return output_path_final  # Return the path to the final saved DXF file
    except Exception as e:
        print(f"Error in main_final: {e}")  # Catch any errors and print the error message
        return None  # Return None if an error occurred




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

def main2(input_file: str, output_path_final: str) -> bool:
    """
    Process the DXF file and create a legend table without warnings or info messages.
    """
    try:
        # Read the input file
        doc = ezdxf.readfile(input_file)
        msp = doc.modelspace()

        # Find the farthest vertical boundary line (X-coordinate)
        farthest_line_x = 0
        for entity in msp.query('LINE[layer=="Boundary"]'):
            x1, x2 = entity.dxf.start.x, entity.dxf.end.x
            if abs(x1 - x2) < 1e-6:  # Check if vertical line
                farthest_line_x = max(farthest_line_x, abs(x1))

        # Table parameters
        table_start_x = farthest_line_x + 50
        table_start_y = 400
        square_size = 40
        col_width = 60
        text_height = 3
        bold_text_height = 5
        legends_height = 15

        # Block mappings with scale factors
        block_mapping = {
            "SM_L": ("Ceiling Light", 2),
            "M_SW": ("Modular Switch", 3),
            "Fan_C": ("Ceiling Fan", 0.4),
            "MBD": ("Main Boards Distribution\nWith MCB and Meter Box", 1.5),
            "EvSwitch": ("EV Switch", 4),
            "AC": ("Air Conditioner", 1.0),
        }

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
            
        doc.saveas(output_path_final)
        return True

    except Exception as e:
        logger.error(f"Critical error: {e}")
        return False

def main_process_wiring(input_file, output_file_final):
    """
    Handles sequential processing of main1 and main2 functions.

    Args:
        input_file (str): The initial input DXF file.
        output_file_final (str): The final output DXF file.
    """
    intermediate_file = "Intermediate1.dxf"

    try:
        # Step 1: Process using main1
        main1(input_file=input_file, output_path_final=intermediate_file)
        print(f"main1 processing completed. Output: {intermediate_file}")

        # Step 2: Process using main2
        main2(input_file=intermediate_file, output_path_final=output_file_final)
        print(f"main2 processing completed. Final Output: {output_file_final}")
    except Exception as e:
        print(f"Error during processing: {e}")

    # Cleanup intermediate file
    cleanup_files([intermediate_file])


def cleanup_files(files):
    """
    Deletes a list of files if they exist.

    Args:
        files (list): List of file paths to delete.
    """
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
        else:
            print(f"File not found: {file}")

# Run the main processing function
# main_process_wiring(
#     input_file="Electrical_Drawing.dxf",
#     output_file_final="Electrical_Complete.dxf"
# )


