import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, LineString, JOIN_STYLE
from shapely.plotting import plot_polygon
from shapely.ops import unary_union
import pandas as pd
import ezdxf

Smh19 = [
    {"name": "Parking", "target_coord": (5, 5), "num_blocks": 100, "aspect_ratio": 1/1},
    {"name": "Garden", "target_coord": (15, 2.5), "num_blocks": 50, "aspect_ratio": 1/3},
    {"name": "Bathroom1", "target_coord": (18, 22), "num_blocks": 30, "aspect_ratio": 1/1},
    {"name": "Bathroom2", "target_coord": (18, 27), "num_blocks": 30, "aspect_ratio": 1/1},
    {"name": "Bedroom", "target_coord": (15, 35), "num_blocks": 80, "aspect_ratio": 4/3},
    {"name": "Staircase", "target_coord": (3, 17), "num_blocks": 70, "aspect_ratio": 1/2},
    {"name": "Foyer", "target_coord": (8, 11), "num_blocks": 30, "aspect_ratio": 1/1},
    {"name": "Livingroom", "target_coord": (15, 8), "num_blocks": 150, "aspect_ratio": 4/3},
    {"name": "Washarea", "target_coord": (4, 35), "num_blocks": 40, "aspect_ratio": 1/1},
    {"name": "Dining", "target_coord": (12, 22), "num_blocks": 50, "aspect_ratio": 1/1},
    {"name": "Kitchen", "target_coord": (7, 35), "num_blocks": 100, "aspect_ratio": 4/3},
]


# Define plot dimensions and block size
plot_width = 20
plot_height = 40
block_size = 1

# Colors for each room
colors = ["lightblue", "lightgreen", "lightcoral", "lightpink", "lightsalmon",
          "lightyellow", "lavender", "lightgray", "powderblue", "mistyrose"]

def home_grid_maker(Smh19, plot_width, plot_height, block_size):
    
    # Initialize arrays and grid
    blocks = [] 
    centers = []
    grid = np.full((plot_height, plot_width), '0', dtype=object)
    
    # Generate blocks and calculate their centers
    for i in range(0, plot_height, block_size): # along y-axis
        for j in range(0, plot_width, block_size): # along x-axis
            block = Polygon([(j, i), (j + block_size, i), 
                            (j + block_size, i + block_size), (j, i + block_size)])
            blocks.append(block)
            center_x = j + block_size / 2
            center_y = i + block_size / 2
            centers.append((center_x, center_y))
            
    # Initial room placement
    for i, spec in enumerate(Smh19):
        target_coord = spec["target_coord"]
        num_blocks = spec["num_blocks"]
        aspect_ratio = spec["aspect_ratio"]
        name = spec["name"]

        rectangle_width = int(np.sqrt(num_blocks * aspect_ratio))
        rectangle_height = int(num_blocks / rectangle_width)

        x_min = max(0, int(target_coord[0] - rectangle_width // 2))
        x_max = min(plot_width, int(target_coord[0] + rectangle_width // 2))
        y_min = max(0, int(target_coord[1] - rectangle_height // 2))
        y_max = min(plot_height, int(target_coord[1] + rectangle_height // 2))

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                grid[y, x] = name
                
    def find_nearest_room(y, x, grid):
        """Find the nearest room to a given coordinate using Manhattan distance"""
        min_distance = float('inf')
        nearest_room = None

        for room_name in set(grid.flatten()) - {'0'}: # all possible names of room inside the grid
            room_coords = np.where(grid == room_name) # for specfic blocks index inside the whole grid
            for ry, rx in zip(room_coords[0], room_coords[1]):
                distance = abs(y - ry) + abs(x - rx)
                if distance < min_distance:
                    min_distance = distance
                    nearest_room = room_name

        return nearest_room


    def find_nearest_room_op(y, x, grid):
        """Find the nearest room to a given coordinate using optimized Manhattan distance calculation"""
        # Exclude '0' (no room) and get unique room identifiers
        room_labels = np.unique(grid[grid != '0'])

        # Initialize minimum distance and nearest room name
        min_distance = float('inf')
        nearest_room = None

        for room in room_labels:
            # Find coordinates of all cells belonging to the current room
            room_coords = np.argwhere(grid == room)

            # Calculate Manhattan distances for all cells in the room in one step
            distances = np.abs(room_coords[:, 0] - y) + np.abs(room_coords[:, 1] - x)

            # Find the minimum distance for this room and update if it's the closest so far
            min_room_distance = np.min(distances)
            if min_room_distance < min_distance:
                min_distance = min_room_distance
                nearest_room = room

        return nearest_room

    def can_expand_rectangle(grid, room_name):
        """Check if the room can be expanded while maintaining rectangular shape"""
        room_coords = np.where(grid == room_name)
        y_min, y_max = room_coords[0].min(), room_coords[0].max()
        x_min, x_max = room_coords[1].min(), room_coords[1].max()

        # Try expanding in all four directions
        expansions = [
            (y_min-1, y_max, x_min, x_max, 'up'),
            (y_min, y_max+1, x_min, x_max, 'down'),
            (y_min, y_max, x_min-1, x_max, 'left'),
            (y_min, y_max, x_min, x_max+1, 'right')
        ]

        valid_expansions = []
        for y1, y2, x1, x2, direction in expansions:
            if (y1 >= 0 and y2 < plot_height and 
                x1 >= 0 and x2 < plot_width):
                if direction in ['up', 'down']:
                    line = grid[y1 if direction == 'up' else y2, x1:x2+1]
                else:
                    line = grid[y1:y2+1, x1 if direction == 'left' else x2]
                if all(cell == '0' for cell in line):
                    valid_expansions.append(direction)

        return valid_expansions


    def can_expand_rectangle_op(grid, room_name):
        """Check if the room can be expanded while maintaining rectangular shape."""
        # Get grid dimensions
        plot_height, plot_width = grid.shape

        # Identify the bounding box of the room
        room_coords = np.where(grid == room_name)
        y_min, y_max = room_coords[0].min(), room_coords[0].max()
        x_min, x_max = room_coords[1].min(), room_coords[1].max()

        # Define expansion directions with their respective boundaries
        directions = {
            'up': (y_min - 1, y_min - 1, x_min, x_max),
            'down': (y_max + 1, y_max + 1, x_min, x_max),
            'left': (y_min, y_max, x_min - 1, x_min - 1),
            'right': (y_min, y_max, x_max + 1, x_max + 1)
        }

        valid_expansions = []
        for direction, (y1, y2, x1, x2) in directions.items():
            # Check if the expansion is within grid bounds
            if 0 <= y1 < plot_height and 0 <= y2 < plot_height and 0 <= x1 < plot_width and 0 <= x2 < plot_width:
                # Extract the line of cells in the direction of expansion
                line = grid[y1:y2+1, x1:x2+1].flatten()
                # Check if all cells in the expansion area are empty ('0')
                if np.all(line == '0'):
                    valid_expansions.append(direction)

        return valid_expansions
    
    def smart_fill_unshaded_regions():
        """Fill unshaded regions while maintaining rectangular shapes"""
        changes_made = True
        while changes_made:
            changes_made = False

            # Find all unshaded blocks
            unshaded_coords = np.where(grid == '0')

            # Try to expand existing rooms
            for room_name in set(grid.flatten()) - {'0'}:
                valid_expansions = can_expand_rectangle_op(grid, room_name)

                if valid_expansions:
                    room_coords = np.where(grid == room_name)
                    y_min, y_max = room_coords[0].min(), room_coords[0].max()
                    x_min, x_max = room_coords[1].min(), room_coords[1].max()

                    for direction in valid_expansions:
                        if direction == 'up':
                            grid[y_min-1, x_min:x_max+1] = room_name
                        elif direction == 'down':
                            grid[y_max+1, x_min:x_max+1] = room_name
                        elif direction == 'left':
                            grid[y_min:y_max+1, x_min-1] = room_name
                        elif direction == 'right':
                            grid[y_min:y_max+1, x_max+1] = room_name
                        changes_made = True
    # [Previous code remains the same until smart_fill_unshaded_regions()]

    def get_neighbors(y, x, grid):
        """Get the neighboring cells and their room names"""
        neighbors = {}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # right, left, down, up

        for dy, dx in directions:
            new_y, new_x = y + dy, x + dx
            if (0 <= new_y < grid.shape[0] and 
                0 <= new_x < grid.shape[1] and 
                grid[new_y, new_x] != '0'):
                room = grid[new_y, new_x]
                neighbors[room] = neighbors.get(room, 0) + 1

        return neighbors


    def fill_remaining_blocks():
        """Fill any remaining unshaded blocks based on neighboring rooms"""
        while True:
            unshaded = np.where(grid == '0')
            if len(unshaded[0]) == 0:
                break

            changes_made = False
            for y, x in zip(unshaded[0], unshaded[1]):
                # Get neighboring rooms and their frequency
                neighbors = get_neighbors(y, x, grid)

                if neighbors:
                    # Find room with most shared boundaries
                    max_shared = max(neighbors.values())
                    candidate_rooms = [room for room, count in neighbors.items() 
                                     if count == max_shared]

                    if len(candidate_rooms) == 1:
                        # If one room has the most boundaries, assign to that room
                        grid[y, x] = candidate_rooms[0]
                        changes_made = True
                    else:
                        # If tie, find the nearest room center among candidates
                        min_dist = float('inf')
                        best_room = None

                        for room in candidate_rooms:
                            room_coords = np.where(grid == room)
                            center_y = np.mean(room_coords[0])
                            center_x = np.mean(room_coords[1])
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)

                            if dist < min_dist:
                                min_dist = dist
                                best_room = room

                        grid[y, x] = best_room
                        changes_made = True

            if not changes_made:
                # If no changes made, assign remaining blocks to nearest room
                for y, x in zip(unshaded[0], unshaded[1]):
                    nearest_room = find_nearest_room_op(y, x, grid)
                    if nearest_room:
                        grid[y, x] = nearest_room

    # Apply the original smart filling algorithm
    smart_fill_unshaded_regions()

    # Apply the final filling for any remaining blocks
    fill_remaining_blocks()
    
    # [Keep all previous imports and initial setup code]

    # Modified Room Placement with Aspect Ratio Enforcement
    for i, spec in enumerate(Smh19):
        target_coord = spec["target_coord"]
        num_blocks = spec["num_blocks"]
        aspect_ratio = spec["aspect_ratio"]
        name = spec["name"]

        # Find optimal integer dimensions that respect aspect ratio
        best_diff = float('inf')
        best_w, best_h = 1, 1
        for w in range(1, int(num_blocks**0.5) + 2):
            h = int(num_blocks / w)
            if abs((w/h) - aspect_ratio) < best_diff and w*h >= num_blocks:
                best_diff = abs((w/h) - aspect_ratio)
                best_w, best_h = w, h

        # Center the rectangle around target coordinates
        start_x = max(0, int(target_coord[0] - best_w/2))
        start_y = max(0, int(target_coord[1] - best_h/2))
        end_x = min(plot_width, start_x + best_w)
        end_y = min(plot_height, start_y + best_h)

        # Ensure minimum area requirement
        placed_blocks = 0
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if placed_blocks >= num_blocks:
                    break
                if grid[y, x] == '0':
                    grid[y, x] = name
                    placed_blocks += 1

    # Enhanced Expansion Logic
    def smart_fill_unshaded_regions():
        """Expand rooms while maintaining rectangular shapes and avoiding overlaps"""
        changes_made = True
        priority_order = sorted(Smh19, key=lambda x: -x['num_blocks'])  # Bigger rooms first

        while changes_made:
            changes_made = False
            for spec in priority_order:
                room_name = spec["name"]
                target = spec["target_coord"]

                room_coords = np.where(grid == room_name)
                if len(room_coords[0]) == 0:
                    continue

                y_min, y_max = room_coords[0].min(), room_coords[0].max()
                x_min, x_max = room_coords[1].min(), room_coords[1].max()

                # Determine preferred expansion direction based on target position
                vertical_preference = 'down' if target[1] > y_max else 'up'
                horizontal_preference = 'right' if target[0] > x_max else 'left'

                # Try expansion in preferred directions first
                for direction in [vertical_preference, horizontal_preference]:
                    valid = can_expand_rectangle_op(grid, room_name)
                    if direction in valid:
                        if direction == 'up':
                            new_y = y_min - 1
                            if np.all(grid[new_y, x_min:x_max+1] == '0'):
                                grid[new_y, x_min:x_max+1] = room_name
                                changes_made = True
                        elif direction == 'down':
                            new_y = y_max + 1
                            if np.all(grid[new_y, x_min:x_max+1] == '0'):
                                grid[new_y, x_min:x_max+1] = room_name
                                changes_made = True
                        elif direction == 'left':
                            new_x = x_min - 1
                            if np.all(grid[y_min:y_max+1, new_x] == '0'):
                                grid[y_min:y_max+1, new_x] = room_name
                                changes_made = True
                        elif direction == 'right':
                            new_x = x_max + 1
                            if np.all(grid[y_min:y_max+1, new_x] == '0'):
                                grid[y_min:y_max+1, new_x] = room_name
                                changes_made = True

    # Post-Processing Cleanup
    def rectangularize_rooms():
        """Ensure all rooms are perfect rectangles by filling their bounding boxes"""
        for spec in Smh19:
            room_name = spec["name"]
            room_coords = np.where(grid == room_name)
            if len(room_coords[0]) == 0:
                continue

            y_min, y_max = room_coords[0].min(), room_coords[0].max()
            x_min, x_max = room_coords[1].min(), room_coords[1].max()

            # Fill entire rectangle
            grid[y_min:y_max+1, x_min:x_max+1] = room_name

    # Execute enhanced algorithms
    smart_fill_unshaded_regions()
    rectangularize_rooms()
    
    return grid

output = home_grid_maker(Smh19, plot_width, plot_height, block_size)

def create_hollow_polygon(plot_width, plot_height, shrink_amount=0.3):
    """Create a hollow polygon for a plot of given dimensions (plot_width, plot_height)
       and apply a shrink effect to form a hollow with sharp corners.
    """
    # Define the outer boundary as the rectangle defined by plot_width and plot_height
    outer_polygon = Polygon([
        (0, 0), 
        (plot_width, 0),
        (plot_width, plot_height),
        (0, plot_height)
    ])
    
    # Create the inner polygon by shrinking the outer polygon
    shrunk_polygon = outer_polygon.buffer(-shrink_amount, join_style=JOIN_STYLE.mitre)

    # Ensure the shrunk polygon is valid and non-empty
    if not shrunk_polygon.is_empty and shrunk_polygon.is_valid:
        # Return a hollow polygon (outer boundary + inner hole)
        hollow_polygon = Polygon(
            shell=outer_polygon.exterior.coords,  # Outer boundary
            holes=[shrunk_polygon.exterior.coords]  # Inner hole
        )
        return hollow_polygon
    else:
        # If shrinking fails, return the original polygon
        return outer_polygon
    
# Create a hollow polygon for the given plot dimensions
hollow_polygon = create_hollow_polygon(plot_width, plot_height, shrink_amount = 1)

def convert_to_hollow_polygons_no_curves(grid, shrink_amount=0.1):
    """Convert each room in the grid to a hollow polygon structure without rounded corners."""
    hollow_polygons = {}

    # Unique room labels (excluding '0')
    room_labels = np.unique(grid[grid != '0'])

    for room_name in room_labels:
        # Get coordinates of all blocks belonging to the current room
        room_coords = np.argwhere(grid == room_name)

        # Generate the polygon by creating a convex hull of the room coordinates
        block_polygons = [
            Polygon([
                (x, y), (x + 1, y), 
                (x + 1, y + 1), (x, y + 1)
            ])
            for y, x in room_coords
        ]

        # Merge all block polygons into one
        room_polygon = block_polygons[0]
        for block_polygon in block_polygons[1:]:
            room_polygon = room_polygon.union(block_polygon)

        # Create a shrunk version of the polygon with sharp corners
        shrunk_polygon = room_polygon.buffer(
            -shrink_amount, join_style=JOIN_STYLE.mitre
        )

        # Ensure the shrunk polygon is valid and non-empty
        if not shrunk_polygon.is_empty and shrunk_polygon.is_valid:
            # Create a hollow polygon by combining the outer and inner (hole) boundaries
            hollow_polygon = Polygon(
                shell=room_polygon.exterior.coords,  # Outer boundary
                holes=[shrunk_polygon.exterior.coords]  # Inner hole
            )
            hollow_polygons[room_name] = hollow_polygon
        else:
            # If shrinking fails, use the original polygon as is
            hollow_polygons[room_name] = room_polygon

    return hollow_polygons


# Convert grid to hollow polygons without rounded corners
hollow_polygons = convert_to_hollow_polygons_no_curves(output, shrink_amount=0.3)
hollow_polygons['Boundary'] = hollow_polygon

def extract_boundary_linestrings_with_modified_exterior(hollow_polygons):
    """
    Extract boundary LineStrings and modify the exterior boundary using bounding box coordinates.
    
    Parameters:
    -----------
    hollow_polygons : dict
        Dictionary of room names as keys and Polygon objects as values
    
    Returns:
    --------
    pd.DataFrame with modified exterior LineString
    """
    # Perform unary_union on all room polygons
    all_polygons = list(hollow_polygons.values())
    unary_union_polygon = unary_union(all_polygons)
    
    # Calculate bounding box perimeter for exterior boundary
    minx, miny, maxx, maxy = unary_union_polygon.bounds
    bounding_box_line = LineString([
        (minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)
    ])
    
    # Extract boundaries
    boundary_data = []

    # Handle different boundary types
    if unary_union_polygon.boundary.geom_type == 'MultiLineString':
        boundary_lines = list(unary_union_polygon.boundary.geoms)
    else:
        boundary_lines = [unary_union_polygon.boundary]
    
    # Process each boundary line
    for idx, line in enumerate(boundary_lines):
        # Modify the exterior boundary to use the bounding box coordinates
        if line.equals(unary_union_polygon.exterior):
            line = bounding_box_line

        # Determine line type and associated rooms
        assigned_rooms = []
        for room_name, polygon in hollow_polygons.items():
            if polygon.intersects(line):
                assigned_rooms.append(room_name)
        
        boundary_data.append({
            'boundary_id': idx,
            'boundary_type': 'Exterior' if line.equals(bounding_box_line) else 'Interior',
            'room_names': ', '.join(assigned_rooms),
            'linestring': line,
            'length': line.length,
            'coords': list(line.coords)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(boundary_data)
    
    return df

# Extract boundary LineStrings with modified exterior
boundary_df = extract_boundary_linestrings_with_modified_exterior(hollow_polygons)

boundary_df.loc[boundary_df['boundary_type'] == 'Exterior', 'room_names'] = boundary_df.loc[
    boundary_df['boundary_type'] == 'Exterior', 'room_names'].apply(lambda x: 'Boundary' if 'Boundary' in x else x)

# Modify 'room_names' for interior boundaries
boundary_df.loc[boundary_df['boundary_type'] == 'Interior', 'room_names'] = boundary_df.loc[
    boundary_df['boundary_type'] == 'Interior', 'room_names'].str.replace('Boundary', '').str.strip()


# Helper function to check collinearity
def are_collinear(p1, p2, p3):
    """Check if points p1, p2, p3 are collinear using the cross-product method."""
    return np.isclose((p2[0] - p1[0]) * (p3[1] - p1[1]), (p2[1] - p1[1]) * (p3[0] - p1[0]))

# Function to break down LINESTRING into individual segments and handle collinear merging
def breakdown_linestring(row):
    original_linestring = row['linestring']
    room_name = row['room_names']
    
    # Extract coordinates
    coords = list(original_linestring.coords)
    
    # Create individual line segments
    segments = [LineString([coords[i], coords[i + 1]]) for i in range(len(coords) - 1)]
    
    # Group collinear segments (if any)
    merged_segments = []
    current_coords = [coords[0], coords[1]]
    for i in range(1, len(coords) - 1):
        if are_collinear(current_coords[-2], current_coords[-1], coords[i + 1]):
            current_coords.append(coords[i + 1])
        else:
            merged_segments.append(LineString(current_coords))
            current_coords = [coords[i], coords[i + 1]]
    merged_segments.append(LineString(current_coords))  # Append the last segment
    
    # Create a list of dictionaries for each segment
    segment_rows = [{'boundary_id': row['boundary_id'],
                     'boundary_type': row['boundary_type'],
                     'room_names': room_name,
                     'linestring': segment} for segment in merged_segments]
    return segment_rows

# Apply the function and expand the DataFrame
new_rows = boundary_df.apply(breakdown_linestring, axis=1)
expanded_df = pd.DataFrame([item for sublist in new_rows for item in sublist])

expanded_df['length'] = expanded_df['linestring'].apply(lambda x: x.length)
expanded_df['xstart'] = expanded_df['linestring'].apply(lambda line: line.coords[0][0])
expanded_df['ystart'] = expanded_df['linestring'].apply(lambda line: line.coords[0][1])
expanded_df['xend'] = expanded_df['linestring'].apply(lambda line: line.coords[-1][0])
expanded_df['yend'] = expanded_df['linestring'].apply(lambda line: line.coords[-1][-1])

def dataframe_to_dxf(df, filename="output.dxf"):
    doc = ezdxf.new()
    msp = doc.modelspace()

    for _, row in df.iterrows():
        x1, y1, x2, y2 = row["xstart"], row["ystart"], row["xend"], row["yend"]
        msp.add_line(start=(x1, y1), end=(x2, y2))

    doc.saveas(filename)
    print(f"DXF file saved as {filename}")


dataframe_to_dxf(expanded_df, "house_layout.dxf")


def main_function(home_dict: list[dict()], plot_width : int, plot_height : int, block_size = 1) -> None:
    
    output = home_grid_maker(home_dict, plot_width, plot_height, block_size)
    hollow_polygon = create_hollow_polygon(plot_width, plot_height, shrink_amount = 1)
    hollow_polygons = convert_to_hollow_polygons_no_curves(output, shrink_amount=0.3)
    hollow_polygons['Boundary'] = hollow_polygon
    boundary_df = extract_boundary_linestrings_with_modified_exterior(hollow_polygons)
    
    boundary_df.loc[boundary_df['boundary_type'] == 'Exterior', 'room_names'] = boundary_df.loc[
        boundary_df['boundary_type'] == 'Exterior', 'room_names'].apply(lambda x: 'Boundary' if 'Boundary' in x else x)

    boundary_df.loc[boundary_df['boundary_type'] == 'Interior', 'room_names'] = boundary_df.loc[
        boundary_df['boundary_type'] == 'Interior', 'room_names'].str.replace('Boundary', '').str.strip()
    
    new_rows = boundary_df.apply(breakdown_linestring, axis=1)
    expanded_df = pd.DataFrame([item for sublist in new_rows for item in sublist])
    
    expanded_df['length'] = expanded_df['linestring'].apply(lambda x: x.length)
    expanded_df['xstart'] = expanded_df['linestring'].apply(lambda line: line.coords[0][0])
    expanded_df['ystart'] = expanded_df['linestring'].apply(lambda line: line.coords[0][1])
    expanded_df['xend'] = expanded_df['linestring'].apply(lambda line: line.coords[-1][0])
    expanded_df['yend'] = expanded_df['linestring'].apply(lambda line: line.coords[-1][-1])
    
    dataframe_to_dxf(expanded_df, "house_layout.dxf")

main_function(Smh19, plot_width, plot_height)





