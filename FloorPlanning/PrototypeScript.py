import pandas as pd
import numpy as np,sys
import ezdxf,math,os, sys,json, django
from pathlib import Path
import matplotlib.pyplot as plt
from shapely import MultiLineString, MultiPoint
from shapely.affinity import scale
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import linemerge, unary_union
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from django.conf import settings
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option('display.float_format', '{:.5f}'.format)
pd.options.mode.copy_on_write = True

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SKM.settings')
data_folder = settings.BASE_DIR / 'FloorPlanning' / 'Datapoints/'
django.setup()

filename = [
    os.path.join(data_folder, 'SMH_DXF1.dxf'),
    os.path.join(data_folder, 'SMH_DXF2.dxf'),
    os.path.join(data_folder, 'SMH_DXF3.dxf'),
    os.path.join(data_folder, 'SMH_DXF4.dxf'),
    os.path.join(data_folder, 'SMH_DXF5.dxf'),
    os.path.join(data_folder, 'SMH_DXF6.dxf'),
    os.path.join(data_folder, 'SMH_DXF7.dxf'),
    os.path.join(data_folder, 'SMH_DXF8.dxf'),
    os.path.join(data_folder, 'SMH_DXF9.dxf'),
    os.path.join(data_folder, 'SMH_DXF10.dxf'),
    os.path.join(data_folder, 'SMH_DXF11.dxf'),
    os.path.join(data_folder, 'SMH_DXF12.dxf'),
    os.path.join(data_folder, 'SMH_DXF13.dxf'),
    os.path.join(data_folder, 'SMH_DXF14.dxf'),
    os.path.join(data_folder, 'SMH_DXF15.dxf'),
    os.path.join(data_folder, 'SMH_DXF16.dxf'),
    os.path.join(data_folder, 'SMH_DXF21.dxf'),
    os.path.join(data_folder, 'SMH_DXF20.dxf'),
    os.path.join(data_folder, 'SMH_DXF19.dxf'),
    os.path.join(data_folder, 'SMH_DXF18.dxf'),
    os.path.join(data_folder, 'SMH_DXF17.dxf'),
]
#Working
# print(data_folder)
# print(filename)


# Parse the JSON data passed as a command-line argument
if len(sys.argv) > 1:
    input_data = json.loads(sys.argv[1])
    
    # Now you can access the data like this:
    project_name = input_data.get('project_name', '')
    u_W = input_data.get('width', 0)
    u_L = input_data.get('length', 0)
    u_Bedroom = input_data.get('bedroom', 0)
    u_Bathroom = input_data.get('bathroom', 0)
    u_Car = input_data.get('car', 0)
    u_Temple = input_data.get('temple', 0)
    u_Garden = input_data.get('garden', 0)
    u_Living_Room = input_data.get('living_room', 0)
    u_Store_Room = input_data.get('store_room', 0)
    u_Plot_Size = u_W*u_L
    u_Aspect_Ratio = u_W/u_W

new_point = pd.DataFrame({'Total Area':[u_Plot_Size], 'Total Area width':[u_W],'Total Area length':[u_L],'No_of_Bedrooms':[u_Bedroom],'No_of_Bathrooms':[u_Bathroom],
                   'No_of_Parking':[u_Car],'No_of_Poojarooms':[u_Temple] , 'No_of_Garden':[u_Garden] ,'No_of_Livingrooms':[u_Living_Room],'No_of_Storerooms':[u_Store_Room] })
User_LB = [u_W,u_L]



def adjust_dxf_coordinates_to00(dxf_file):
    # Read the DXF file
    print(f"Checking file existence: {dxf_file}")  # Debug: Log file path
    if not os.path.exists(dxf_file):
        print(f"File does not exist: {dxf_file}")  # Debug: Log if file does not exist
        raise FileNotFoundError(f"File not found: {dxf_file}")

    print(f"File found: {dxf_file}")  # Debug: Log if file exists
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    # Find the minimum X and Y coordinates
    min_x_entities = [entity.dxf.center.x if entity.dxftype() == 'CIRCLE' else
                      entity.dxf.center.x - entity.dxf.radius if entity.dxftype() == 'ARC' else
                      entity.dxf.insert.x if entity.dxftype() in ['TEXT', 'MTEXT'] else
                      min(entity.dxf.start.x, entity.dxf.end.x) if entity.dxftype() == 'LINE' else
                      None
                      for entity in msp]
    min_y_entities = [entity.dxf.center.y if entity.dxftype() == 'CIRCLE' else
                      entity.dxf.center.y - entity.dxf.radius if entity.dxftype() == 'ARC' else
                      entity.dxf.insert.y if entity.dxftype() in ['TEXT', 'MTEXT'] else
                      min(entity.dxf.start.y, entity.dxf.end.y) if entity.dxftype() == 'LINE' else
                      None
                      for entity in msp]
    min_x = min(x for x in min_x_entities if x is not None)
    min_y = min(y for y in min_y_entities if y is not None)

    # Calculate the offset
    offset_x = -min_x
    offset_y = -min_y

    # Update all coordinates in the DXF file
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            start = (start.x + offset_x, start.y + offset_y, start.z)
            end = (end.x + offset_x, end.y + offset_y, end.z)
            entity.dxf.start = start
            entity.dxf.end = end
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            center = (center.x + offset_x, center.y + offset_y, center.z)
            entity.dxf.center = center
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            center = (center.x + offset_x, center.y + offset_y, center.z)
            entity.dxf.center = center
        elif entity.dxftype() in ['TEXT', 'MTEXT']:
            insert = entity.dxf.insert
            insert = (insert.x + offset_x, insert.y + offset_y, insert.z)
            entity.dxf.insert = insert

    # Save the modified DXF file
    # output_filename = dxf_file+'new'
    output_filename = os.path.splitext(os.path.basename(dxf_file))[0] + project_name+"_new.dxf"
    output_path = os.path.join(settings.BASE_DIR, 'Temp', 'dxfCache', output_filename)
    doc.saveas(output_path)

# Example usage
def calculate_length(start, end):
    return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)

#DXF to PANDAS DATAFRAME
def Dxf_to_DF(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    
    entities_data = []
    for entity in msp:
        entity_data = {'Type': entity.dxftype(), 'Layer': entity.dxf.layer}
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            length = calculate_length(start, end)
            entity_data.update({
                'X_start': start.x, 'Y_start': start.y, 'Z_start': start.z,
                'X_end': end.x, 'Y_end': end.y, 'Z_end': end.z,
                'Length': length})
            horizontal = abs(end.x - start.x) > abs(end.y - start.y)
            vertical = not horizontal
            entity_data.update({'Horizontal': horizontal, 'Vertical': vertical})
            
            
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Radius': radius})
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Radius': radius,
                'Start Angle': start_angle,
                'End Angle': end_angle})
            
        elif entity.dxftype() == 'TEXT':
            insert = entity.dxf.insert
            text = entity.dxf.text
            entity_data.update({
                'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                'Text': text})
        elif entity.dxftype() == 'MTEXT':
            text = entity.plain_text()
            insertion_point = entity.dxf.insert
            entity_data.update({
                'Text': text,
                'X_insert': insertion_point.x,
                'Y_insert': insertion_point.y,
                'Z_insert': insertion_point.z
            })
            
            
        entities_data.append(entity_data)
    
    return pd.DataFrame(entities_data) 

def adjust_Xstart_ystart(df):
    df_copy = df.copy()  # Create a copy of the original DataFrame
    # Swap X_start and X_end if X_start is greater than X_end
    df_copy.loc[df_copy['X_start'] > df_copy['X_end'], ['X_start', 'X_end']] = df_copy.loc[df_copy['X_start'] > df_copy['X_end'], ['X_end', 'X_start']].values
    # Swap Y_start and Y_end if Y_start is greater than Y_end
    df_copy.loc[df_copy['Y_start'] > df_copy['Y_end'], ['Y_start', 'Y_end']] = df_copy.loc[df_copy['Y_start'] > df_copy['Y_end'], ['Y_end', 'Y_start']].values
    return df_copy


def Similarity_fuc_main(new_point, filename):
    print(f"Debugging information for filename:")
    print(f"Received filename: {filename}")
    print(f"Absolute path: {os.path.abspath(filename)}")
    print(f"File exists: {os.path.exists(filename)}")
    
    if os.path.exists(filename):
        print(f"Is file: {os.path.isfile(filename)}")
        print(f"File size: {os.path.getsize(filename)} bytes")
        print(f"File permissions: {oct(os.stat(filename).st_mode)[-3:]}")
    cols = new_point.columns
    MetaData = pd.read_csv(filename)
    MetaData = MetaData.set_index('Unnamed: 0')
    MetaData = MetaData[cols]
    # Separate categorical and continuous data
    data_cat = MetaData.select_dtypes(include='object')
    data_cont = MetaData.select_dtypes(exclude='object')

    new_data_cat = new_point.select_dtypes(include='object')
    new_data_cont = new_point.select_dtypes(exclude='object')
    
    # Standardize continuous data
    scaler = StandardScaler()
    data_cont_S = scaler.fit_transform(data_cont)
    new_data_cont_S = scaler.transform(new_data_cont)
    
    data_cont_S_DF = pd.DataFrame(data_cont_S, 
                                  columns = data_cont.columns + '_S', 
                                  index = data_cont.index)
    new_data_cont_S_DF = pd.DataFrame(new_data_cont_S, 
                                      columns = new_data_cont.columns + '_S', 
                                      index = new_data_cont.index)
    
    
    
    # Combine continuous and categorical data
    # data_comb = pd.concat([data_cont_S_DF, data_cat], axis=1)
    # new_data_comb = pd.concat([new_data_cont_S_DF, new_data_cat], axis=1)
    
    # Ensure the weights match the number of features in the combined data
    num_features = data_cont_S_DF.shape[1]
    # weights = np.ones(num_features)
    weights = np.array([1, 100,20, 1, 1, 1, 1, 1, 1, 1],dtype = 'float')

    # weights /= weights.sum()

    # Manually compute the weighted distances
    new_data_point = new_data_cont_S_DF.values.flatten()
    weighted_distances = np.sqrt(np.sum(weights * ((data_cont_S_DF.values - new_data_point) ** 2), axis=1))

    # Find the indices of the k-nearest neighbors (k=3)
    k = 3
    nearest_neighbors_indices = np.argsort(weighted_distances)[:k]

    # Get the nearest neighbors
    nearest_neighbors = MetaData.iloc[nearest_neighbors_indices]
    sorted_points = nearest_neighbors.index
    
    # Calculate differences
    differences = []
    for idx in nearest_neighbors_indices:
        neighbor = MetaData.iloc[idx]
        diff = {}
        for col in MetaData.columns:
            if MetaData[col].dtype == 'object':
                diff[col] = neighbor[col] == new_point[col].values[0]
            else:
                diff[col] = neighbor[col] - new_point[col].values[0]
        differences.append(diff)
    
    Differences = pd.DataFrame(differences, index=nearest_neighbors.index)
    
    return nearest_neighbors, sorted_points


def add_horizontal_lines_for_X(df):
    # Find the minimum and maximum Y coordinates of the start points
    min_y = df['Y_start'].min()
    max_y = df['Y_start'].max()
    min_x = df['X_start'].min()
    max_x = df['X_start'].max()

    # Calculate the number of lines to create
    num_lines = int((max_x - min_x) / 0.5)  # Assuming 5 inch distance between lines

    # Create a list to store the new lines
    new_lines = []

    # Generate the horizontal lines
    for i in range(num_lines + 1):
        x = min_x + i * 0.5  # Assuming 5 inch distance between lines
        
        # Check if there are existing horizontal lines at this Y coordinate
        existing_lines_at_x = df[(df['X_start'] == x) & (df['X_end'] == x)]
        if len(existing_lines_at_x) > 1:
            continue  # Skip adding new line if there are already more than one horizontal lines at this Y coordinate

        # Check if there are existing horizontal lines within the range of +10 and -10 of this Y coordinate
        nearby_lines = df[(df['X_start'] >= x - 12) & (df['X_start'] <= x + 12) & (df['X_end'] >= x - 12) & (df['X_end'] <= x + 12)]
        if len(nearby_lines) > 0:
            continue  # Skip adding new line if there are already existing lines within +10 and -10 of this Y coordinate

        new_line = {
            'Type': 'LINE',
            'Layer': 'HorizontalLines',
            'X_start': x,
            'Y_start': min_y,
            'Z_start': 0,
            'X_end': x,
            'Y_end': max_y,
            'Z_end': 0,
            'Length': max_y - min_y
        }
        new_lines.append(new_line)

    # Concatenate the new lines with the existing DataFrame
    df_with_horizontal_lines = pd.DataFrame(new_lines)
    

    return df_with_horizontal_lines

#----------------------------------------------------------------------------------------------------------------------


def add_horizontal_lines_for_X_updated(df):
    # Find the minimum and maximum Y coordinates of the start points
    min_y = df['Y_start'].min()
    max_y = df['Y_start'].max()
    min_x = df['X_start'].min()
    max_x = df['X_start'].max()

    # Calculate the number of lines to create
    num_lines = int((max_x - min_x) / 0.5)  # Assuming 5 inch distance between lines

    # Create a list to store the new lines
    new_lines = []

    # Get the bounding box for the 'Staircase' layer
    if 'Staircase' in df['Layer'].values:
        staircase_layer = df[df['Layer'] == 'Staircase']
        staircase_min_x = staircase_layer['X_start'].min()
        staircase_max_x = staircase_layer['X_end'].max()
        staircase_min_y = staircase_layer['Y_start'].min()
        staircase_max_y = staircase_layer['Y_end'].max()
    else:
        staircase_min_x = staircase_max_x = staircase_min_y = staircase_max_y = None

    # Generate the horizontal lines
    for i in range(num_lines + 1):
        x = min_x + i * 0.5  # Assuming 5 inch distance between lines
        
        # Check if there are existing horizontal lines at this Y coordinate
        existing_lines_at_x = df[(df['X_start'] == x) & (df['X_end'] == x)]
        if len(existing_lines_at_x) > 1:
            continue  # Skip adding new line if there are already more than one horizontal lines at this Y coordinate

        # Check if there are existing horizontal lines within the range of +10 and -10 of this Y coordinate
        nearby_lines = df[(df['X_start'] >= x - 12) & (df['X_start'] <= x + 12) & (df['X_end'] >= x - 12) & (df['X_end'] <= x + 12)]
        if len(nearby_lines) > 0:
            continue  # Skip adding new line if there are already existing lines within +10 and -10 of this Y coordinate

        # Check if the line intersects with the 'Staircase' layer
        if staircase_min_x is not None and staircase_max_x is not None and staircase_min_y is not None and staircase_max_y is not None:
            if staircase_min_x <= x <= staircase_max_x and staircase_min_y <= max_y and staircase_max_y >= min_y:
                continue  # Skip adding new line if it intersects with the 'Staircase' layer

        new_line = {
            'Type': 'LINE',
            'Layer': 'HorizontalLines',
            'X_start': x,
            'Y_start': min_y,
            'Z_start': 0,
            'X_end': x,
            'Y_end': max_y,
            'Z_end': 0,
            'Length': max_y - min_y
        }
        new_lines.append(new_line)

    # Concatenate the new lines with the existing DataFrame
    df_with_horizontal_lines = pd.concat([df, pd.DataFrame(new_lines)], ignore_index=True)
    
    return df_with_horizontal_lines

#----------------------------------------------------------------------------------------------------------------------

def find_line_groups_for_X(df):
    # Filter the DataFrame to include only the horizontal lines
    horizontal_lines = df[df['Layer'] == 'HorizontalLines']

    # Sort the horizontal lines by Y coordinate
    horizontal_lines_sorted = horizontal_lines.sort_values(by=['X_start']).reset_index(drop=True)

    # Initialize variables
    current_group = 1
    current_y = horizontal_lines_sorted.loc[0, 'X_start']
    horizontal_lines_sorted['Group'] = current_group

    # Group lines into bundles
    for index in range(1, len(horizontal_lines_sorted)):
        if horizontal_lines_sorted.loc[index, 'X_start'] - current_y > 0.5:
            current_group += 1
        current_y = horizontal_lines_sorted.loc[index, 'X_start']
        horizontal_lines_sorted.at[index, 'Group'] = current_group

    # Calculate the number of lines in each group
    group_counts = horizontal_lines_sorted['Group'].value_counts()

    # Calculate the proportion column
    horizontal_lines_sorted['Proportion'] = horizontal_lines_sorted['Group'].apply(lambda x: group_counts[x] / len(horizontal_lines_sorted) * 100)

    # Calculate the middle line of each group
    group_middle_lines_for_X = horizontal_lines_sorted.groupby('Group').agg({
        'Y_start': 'first',
        'Y_end': 'first',
        'X_start': 'mean',
        'X_end': 'mean',
        'Z_start': 'first',
        'Z_end': 'first',
        'Length': 'first',
        'Layer': 'first',
        'Type': 'first',
        'Proportion': 'first'
    }).reset_index()

    return group_middle_lines_for_X

def distribute_units_proportionally_for_X(df, total_units_for_X):
    # Calculate the total proportion
    total_proportion = df['Proportion'].sum()

    # Calculate the units per proportion
    units_per_proportion = total_units_for_X / total_proportion

    # Distribute units proportionally
    df['Units'] = df['Proportion'] * units_per_proportion

    return df

def distribute_units_between_lines_for_X(df):
    point_pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Distribute half of the spacing above the line and half below the line
        above_line = row['X_start'] + row['Units'] / 2
        below_line = row['X_start'] - row['Units']/ 2

        # Add the points to the list
        point_pairs.append((above_line, below_line))
        point_pairs = sorted(point_pairs, key=lambda x: x[0])
    return point_pairs

#----------------------------------------------------------------------------------------------------------------------

def trim_for_X(df, cut_x1, cut_x2,diff, filename1, filename2, filename3):

    doc1 = ezdxf.new()
    msp1 = doc1.modelspace()

    doc2 = ezdxf.new()
    msp2 = doc2.modelspace()

    doc3 = ezdxf.new()
    msp3 = doc3.modelspace()

    for _, row in df.iterrows():
        if row['Type'] == 'LINE':
            layer = row['Layer']
            dxfattribs = {'layer': layer}
            if cut_x1 <= row['X_start'] <= cut_x2:
            # Horizontal line
                if row['X_start'] == row['X_end']:
                # Line is between cut_x1 and cut_x2
                    cal = row['X_start'] - cut_x1
                    msp1.add_line((row['X_start']-cal, row['Y_start']), (row['X_end']-cal, row['Y_end']), dxfattribs)
            elif row['X_start'] < cut_x1 and row['X_end'] > cut_x1:
                # Split the line at cut_x1 
                #split_x1 = row['X_start'] + (cut_x1 - row['Y_start']) / (row['Y_end'] - row['Y_start']) * (row['X_end'] - row['X_start'])
                #if row['Y_start'] < row['Y_end']:
                msp1.add_line((row['X_start'], row['Y_start']), (cut_x1,row['Y_end']),dxfattribs)
                #msp2.add_line((split_x1, cut_x1), (row['X_end'], row['Y_end']),dxfattribs)
#                         else:
#                             msp1.add_line((split_x1, cut_x1), (row['X_start'], row['Y_start']),dxfattribs)
#                             msp2.add_line((row['X_end'], row['Y_end']), (split_x1, cut_x1),dxfattribs)
            elif row['X_start'] <= cut_x1 and row['X_end'] <= cut_x1:
                msp1.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)
#                     elif row['Y_start'] >= cut_x1 and row['Y_end'] >= cut_x1:
#                         msp2.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)

            if row['X_start'] < cut_x2 and row['X_end'] > cut_x2:
                # Split the line at cut_x2
#                         split_x2 = row['X_start'] + (cut_x2 - row['Y_start']) / (row['Y_end'] - row['Y_start']) * (row['X_end'] - row['X_start'])
#                         if row['Y_start'] < row['Y_end']:
                #msp2.add_line((row['X_start'], row['Y_start']), (split_x2, cut_x2),dxfattribs)
                msp3.add_line(( cut_x2-diff,row['Y_start']), (row['X_end']-diff, row['Y_end']),dxfattribs)
#                         else:
#                             msp2.add_line((split_x2, cut_x2), (row['X_start'], row['Y_start']),dxfattribs)
#                             msp3.add_line((row['X_end'], row['Y_end']-diff), (split_x2, cut_x2-dif),dxfattribs)
            elif row['X_start'] <= cut_x2 and row['X_end'] <= cut_x2:
                msp2.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)
            elif row['X_start'] >= cut_x2 and row['X_end'] >= cut_x2:
                msp3.add_line((row['X_start']-diff, row['Y_start']), (row['X_end']-diff, row['Y_end']),dxfattribs)
            elif row['X_start'] <= cut_x1 and row['X_end'] >= cut_x2:
                msp1.add_line((row['X_start'], row['Y_start']), (cut_x1, row['Y_end']),dxfattribs)
                msp3.add_line((cut_x2-diff,row['Y_start']), (row['X_end']-diff, row['Y_end']),dxfattribs)
            elif row['X_end'] <= cut_x1 and row['X_start'] >= cut_x2:
                msp1.add_line((cut_x1, row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)
                msp3.add_line((row['X_start']-diff, row['Y_start']), (cut_x2-diff,row['Y_end']),dxfattribs)
            elif row['X_start'] <= cut_x2 and row['X_start'] >= cut_x1:
                msp3.add_line((cut_x2-diff, row['Y_start']), (row['X_end']-diff, row['Y_end']),dxfattribs)
            elif row['X_end'] <= cut_x2 and row['X_end'] >= cut_x1:
                msp3.add_line((row['X_start']-diff, row['Y_start']), (cut_x2-diff, row['Y_end']),dxfattribs)


        elif row['Type'] == 'CIRCLE':
            layer = row['Layer']
            dxfattribs = {'layer': layer}
            if row['X_center'] - row['Radius'] < cut_x1 and row['X_center'] + row['Radius'] > cut_x1:
                # Circle intersects the cut line at cut_x1, add to msp1
                msp1.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['X_center'] - row['Radius'] <= cut_x1:
                msp1.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['X_center'] + row['Radius'] >= cut_x1:
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)

            if row['X_center'] - row['Radius'] < cut_x2 and row['X_center'] + row['Radius'] > cut_x2:
                # Circle intersects the cut line at cut_x2, add to msp2
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['X_center'] - row['Radius'] <= cut_x2:
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['X_center'] + row['Radius'] >= cut_x2:
                msp3.add_circle((row['X_center']-diff, row['Y_center']), row['Radius'],dxfattribs)

        elif row['Type'] == 'TEXT':
            layer = row['Layer'] 
            if row['X_insert'] <= cut_x1:
            # TEXT entity
                msp1.add_mtext(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert']),'layer': layer})
            elif row['X_insert'] >= cut_x2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_mtext(row['Text'],dxfattribs={'insert': (row['X_insert']-diff, row['Y_insert']),'layer': layer})
            elif  cut_x1 <=row ['X_insert'] <= cut_x2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_mtext(row['Text'],dxfattribs={'insert': (cut_x2-diff, row['Y_insert']),'layer': layer})



        elif row['Type'] == 'MTEXT':
            layer = row['Layer'] 
            if row['X_insert'] <= cut_x1:
            # TEXT entity
                msp1.add_mtext(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert']),'layer': layer})
            elif row['X_insert'] >= cut_x2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_mtext(row['Text'],dxfattribs={'insert': (row['X_insert']-(diff), row['Y_insert']),'layer': layer})
            elif  cut_x1 <=row ['X_insert'] <= cut_x2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp1.add_mtext(row['Text'],dxfattribs={'insert': (cut_x1, row['Y_insert']),'layer': layer})
                
            
        
        epsilon = 1e-6  # Small value for floating-point comparisons

        for line1 in msp1.query('LINE'):
            start1 = line1.dxf.start
            end1 = line1.dxf.end
            slope1 = (end1[1] - start1[1]) / (end1[0] - start1[0]) if (end1[0] - start1[0]) != 0 else math.inf

            for line2 in msp3.query('LINE'):
                start2 = line2.dxf.start
                end2 = line2.dxf.end
                slope2 = (end2[1] - start2[1]) / (end2[0] - start2[0]) if (end2[0] - start2[0]) != 0 else math.inf

                # Check if the lines have similar slopes and end/start points within epsilon
                if abs(slope1 - slope2) < epsilon and abs(end1 - start2) < epsilon:
                    if line1.dxf.layer == line2.dxf.layer:
                        msp1.add_line(start1, end2, dxfattribs={'layer': line1.dxf.layer})
                        msp1.delete_entity(line1)
                        msp3.delete_entity(line2)
                        break
                     
    filepath1=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename1)   
    filepath2=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename2)  
    filepath3=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename3)                
    doc1.saveas(filepath1)
    doc2.saveas(filepath2)
    doc3.saveas(filepath3)

    
def add_horizontal_lines_for_Y(df):
    # Find the minimum and maximum Y coordinates of the start points
    min_y = df['Y_start'].min()
    max_y = df['Y_start'].max()
    min_x = df['X_start'].min()
    max_x = df['X_start'].max()

    # Calculate the number of lines to create
    num_lines = int((max_y - min_y) / 0.5)  # Assuming 5 inch distance between lines

    # Create a list to store the new lines
    new_lines = []

    # Generate the horizontal lines
    for i in range(num_lines + 1):
        y = min_y + i * 0.5  # Assuming 5 inch distance between lines
        
        # Check if there are existing horizontal lines at this Y coordinate
        existing_lines_at_y = df[(df['Y_start'] == y) & (df['Y_end'] == y)]
        if len(existing_lines_at_y) > 1:
            continue  # Skip adding new line if there are already more than one horizontal lines at this Y coordinate

        # Check if there are existing horizontal lines within the range of +10 and -10 of this Y coordinate
        nearby_lines = df[(df['Y_start'] >= y - 12) & (df['Y_start'] <= y + 12) & (df['Y_end'] >= y - 12) & (df['Y_end'] <= y + 12)]
        if len(nearby_lines) > 0:
            continue  # Skip adding new line if there are already existing lines within +10 and -10 of this Y coordinate

        new_line = {
            'Type': 'LINE',
            'Layer': 'HorizontalLines',
            'X_start': min_x,
            'Y_start': y,
            'Z_start': 0,
            'X_end': max_x,
            'Y_end': y,
            'Z_end': 0,
            'Length': max_x - min_x
        }
        new_lines.append(new_line)

    # Concatenate the new lines with the existing DataFrame
    df_with_horizontal_lines = pd.DataFrame(new_lines)
    

    return df_with_horizontal_lines



#----------------------------------------------------------------------------------------------------------------------


def add_horizontal_lines_for_Y_updated(df):
    # Find the minimum and maximum Y coordinates of the start points
    min_y = df['Y_start'].min()
    max_y = df['Y_start'].max()
    min_x = df['X_start'].min()
    max_x = df['X_start'].max()

    # Calculate the number of lines to create
    num_lines = int((max_y - min_y) / 0.5)  # Assuming 5 inch distance between lines

    # Create a list to store the new lines
    new_lines = []
    
    # Get the bounding box for the 'Staircase' layer
    if 'Staircase' in df['Layer'].values:
        staircase_layer = df[df['Layer'] == 'Staircase']
        staircase_min_x = staircase_layer['X_start'].min()
        staircase_max_x = staircase_layer['X_end'].max()
        staircase_min_y = staircase_layer['Y_start'].min()
        staircase_max_y = staircase_layer['Y_end'].max()
    else:
        staircase_min_x = staircase_max_x = staircase_min_y = staircase_max_y = None

    # Generate the horizontal lines
    for i in range(num_lines + 1):
        y = min_y + i * 0.5  # Assuming 5 inch distance between lines
        
        # Check if there are existing horizontal lines at this Y coordinate
        existing_lines_at_y = df[(df['Y_start'] == y) & (df['Y_end'] == y)]
        if len(existing_lines_at_y) > 1:
            continue  # Skip adding new line if there are already more than one horizontal lines at this Y coordinate

        # Check if there are existing horizontal lines within the range of +10 and -10 of this Y coordinate
        nearby_lines = df[(df['Y_start'] >= y - 12) & (df['Y_start'] <= y + 12) & (df['Y_end'] >= y - 12) & (df['Y_end'] <= y + 12)]
        if len(nearby_lines) > 0:
            continue  # Skip adding new line if there are already existing lines within +10 and -10 of this Y coordinate

        # Check if the line intersects with the 'Staircase' layer
        if staircase_min_x is not None and staircase_max_x is not None and staircase_min_y is not None and staircase_max_y is not None:
            if staircase_min_y <= y <= staircase_max_y and staircase_min_x <= max_x and staircase_max_x >= min_x:
                continue  # Skip adding new line if it intersects with the 'Staircase' layer

        new_line = {
            'Type': 'LINE',
            'Layer': 'HorizontalLines',
            'X_start': min_x,
            'Y_start': y,
            'Z_start': 0,
            'X_end': max_x,
            'Y_end': y,
            'Z_end': 0,
            'Length': max_x - min_x
        }
        new_lines.append(new_line)

    # Concatenate the new lines with the existing DataFrame
    df_with_horizontal_lines = pd.concat([df, pd.DataFrame(new_lines)], ignore_index=True)
    
    return df_with_horizontal_lines

#----------------------------------------------------------------------------------------------------------------------

def find_line_groups_for_Y(df):
    # Filter the DataFrame to include only the horizontal lines
    horizontal_lines = df[df['Layer'] == 'HorizontalLines']

    # Sort the horizontal lines by Y coordinate
    horizontal_lines_sorted = horizontal_lines.sort_values(by=['Y_start']).reset_index(drop=True)

    # Initialize variables
    current_group = 1
    current_y = horizontal_lines_sorted.loc[0, 'Y_start']
    horizontal_lines_sorted['Group'] = current_group

    # Group lines into bundles
    for index in range(1, len(horizontal_lines_sorted)):
        if horizontal_lines_sorted.loc[index, 'Y_start'] - current_y > 0.5:
            current_group += 1
        current_y = horizontal_lines_sorted.loc[index, 'Y_start']
        horizontal_lines_sorted.at[index, 'Group'] = current_group

    # Calculate the number of lines in each group
    group_counts = horizontal_lines_sorted['Group'].value_counts()

    # Calculate the proportion column
    horizontal_lines_sorted['Proportion'] = horizontal_lines_sorted['Group'].apply(lambda x: group_counts[x] / len(horizontal_lines_sorted) * 100)

    # Calculate the middle line of each group
    group_middle_lines_for_Y = horizontal_lines_sorted.groupby('Group').agg({
        'Y_start': 'mean',
        'Y_end': 'mean',
        'X_start': 'first',
        'X_end': 'first',
        'Z_start': 'first',
        'Z_end': 'first',
        'Length': 'first',
        'Layer': 'first',
        'Type': 'first',
        'Proportion': 'first'
    }).reset_index()

    return group_middle_lines_for_Y


#----------------------------------------------------------------------------------------------------------------------

def distribute_units_proportionally_for_Y(df, total_units_for_Y):
    # Calculate the total proportion
    total_proportion = df['Proportion'].sum()

    # Calculate the units per proportion
    units_per_proportion = total_units_for_Y / total_proportion

    # Distribute units proportionally
    df['Units'] = df['Proportion'] * units_per_proportion

    return df

# Assuming df is the DataFrame with the proportional column
# and total_units is the total number of units to distribute

#----------------------------------------------------------------------------------------------------------------------

def distribute_units_between_lines_for_Y(df):
    point_pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Distribute half of the spacing above the line and half below the line
        above_line = row['Y_start'] + row['Units'] / 2
        below_line = row['Y_start'] - row['Units']/ 2

        # Add the points to the list
        point_pairs.append((above_line, below_line))
        point_pairs = sorted(point_pairs, key=lambda x: x[0])
    return point_pairs

#----------------------------------------------------------------------------------------------------------------------

def trim_dxf_for_Y(df, cut_y1, cut_y2,diff, filename1, filename2, filename3):
    doc1 = ezdxf.new()
    msp1 = doc1.modelspace()

    doc2 = ezdxf.new()
    msp2 = doc2.modelspace()

    doc3 = ezdxf.new()
    msp3 = doc3.modelspace()

    for _, row in df.iterrows():
        if row['Type'] == 'LINE':
            layer = row['Layer']
            dxfattribs = {'layer': layer}
            if cut_y1 <= row['Y_start'] <= cut_y2:
            # Horizontal line
                if row['Y_start'] == row['Y_end']:
                # Line is between cut_y1 and cut_y2
                    cal = row['Y_start'] - cut_y1
                    msp1.add_line((row['X_start'], row['Y_start']-cal), (row['X_end'], row['Y_end']-cal), dxfattribs)
            elif row['Y_start'] < cut_y1 and row['Y_end'] > cut_y1:
                # Split the line at cut_y1 
                #split_x1 = row['X_start'] + (cut_y1 - row['Y_start']) / (row['Y_end'] - row['Y_start']) * (row['X_end'] - row['X_start'])
                #if row['Y_start'] < row['Y_end']:
                msp1.add_line((row['X_start'], row['Y_start']), (row['X_end'], cut_y1),dxfattribs)
                #msp2.add_line((split_x1, cut_y1), (row['X_end'], row['Y_end']),dxfattribs)
#                         else:
#                             msp1.add_line((split_x1, cut_y1), (row['X_start'], row['Y_start']),dxfattribs)
#                             msp2.add_line((row['X_end'], row['Y_end']), (split_x1, cut_y1),dxfattribs)
            elif row['Y_start'] <= cut_y1 and row['Y_end'] <= cut_y1:
                msp1.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)
#                     elif row['Y_start'] >= cut_y1 and row['Y_end'] >= cut_y1:
#                         msp2.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)

            if row['Y_start'] < cut_y2 and row['Y_end'] > cut_y2:
                # Split the line at cut_y2
#                         split_x2 = row['X_start'] + (cut_y2 - row['Y_start']) / (row['Y_end'] - row['Y_start']) * (row['X_end'] - row['X_start'])
#                         if row['Y_start'] < row['Y_end']:
                #msp2.add_line((row['X_start'], row['Y_start']), (split_x2, cut_y2),dxfattribs)
                msp3.add_line((row['X_start'], cut_y2-diff), (row['X_end'], row['Y_end']-diff),dxfattribs)
#                         else:
#                             msp2.add_line((split_x2, cut_y2), (row['X_start'], row['Y_start']),dxfattribs)
#                             msp3.add_line((row['X_end'], row['Y_end']-diff), (split_x2, cut_y2-dif),dxfattribs)
            elif row['Y_start'] <= cut_y2 and row['Y_end'] <= cut_y2:
                msp2.add_line((row['X_start'], row['Y_start']), (row['X_end'], row['Y_end']),dxfattribs)
            elif row['Y_start'] >= cut_y2 and row['Y_end'] >= cut_y2:
                msp3.add_line((row['X_start'], row['Y_start']-diff), (row['X_end'], row['Y_end']-diff),dxfattribs)
            elif row['Y_start'] <= cut_y1 and row['Y_end'] >= cut_y2:
                msp1.add_line((row['X_start'], row['Y_start']), (row['X_end'], cut_y1),dxfattribs)
                msp3.add_line((row['X_start'], cut_y2-diff), (row['X_end'], row['Y_end']-diff),dxfattribs)
            elif row['Y_end'] <= cut_y1 and row['Y_start'] >= cut_y2:
                msp1.add_line((row['X_start'], cut_y1), (row['X_end'], row['Y_end']),dxfattribs)
                msp3.add_line((row['X_start'], row['Y_start']-diff), (row['X_end'], cut_y2-diff),dxfattribs)
            elif row['Y_start'] <= cut_y2 and row['Y_start'] >= cut_y1:
                msp3.add_line((row['X_start'], cut_y2-diff), (row['X_end'], row['Y_end']-diff),dxfattribs)
            elif row['Y_end'] <= cut_y2 and row['Y_end'] >= cut_y1:
                msp3.add_line((row['X_start'], row['Y_start']-diff), (row['X_end'], cut_y2-diff),dxfattribs)


        elif row['Type'] == 'CIRCLE':
            layer = row['Layer']
            dxfattribs = {'layer': layer}
            if row['Y_center'] - row['Radius'] < cut_y1 and row['Y_center'] + row['Radius'] > cut_y1:
                # Circle intersects the cut line at cut_y1, add to msp1
                msp1.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['Y_center'] - row['Radius'] <= cut_y1:
                msp1.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['Y_center'] + row['Radius'] >= cut_y1:
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)

            if row['Y_center'] - row['Radius'] < cut_y2 and row['Y_center'] + row['Radius'] > cut_y2:
                # Circle intersects the cut line at cut_y2, add to msp2
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['Y_center'] - row['Radius'] <= cut_y2:
                msp2.add_circle((row['X_center'], row['Y_center']), row['Radius'],dxfattribs)
            elif row['Y_center'] + row['Radius'] >= cut_y2:
                msp3.add_circle((row['X_center'], row['Y_center']-diff), row['Radius'],dxfattribs)

        elif row['Type'] == 'TEXT':
            layer = row['Layer'] 
            if row['Y_insert'] <= cut_y1:
            # TEXT entity
                msp1.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
            elif row['Y_insert'] >= cut_y2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert']-diff)})



        elif row['Type'] == 'MTEXT':
            layer = row['Layer']
            if row['Y_insert'] <= cut_y1:
            # TEXT entity
                msp1.add_mtext(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert']),'layer': layer})
            elif row['Y_insert'] >= cut_y2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_mtext(row['Text'],dxfattribs={'insert': (row['X_insert'], row['Y_insert']-(1*diff)),'layer': layer})
            elif  cut_y1 <=row ['Y_insert'] <= cut_y2:
                #msp2.add_text(row['Text'], dxfattribs={'insert': (row['X_insert'], row['Y_insert'])})
                msp3.add_mtext(row['Text'],dxfattribs={'insert': (row['X_insert'], cut_y2-(1*diff)),'layer': layer})
        
        
        epsilon = 1e-6
        for line1 in msp1.query('LINE'):
            start1 = line1.dxf.start
            end1 = line1.dxf.end
            slope1 = (end1[0] - start1[0]) / (end1[1] - start1[1]) if (end1[1] - start1[1]) != 0 else math.inf

            for line2 in msp3.query('LINE'):
                start2 = line2.dxf.start
                end2 = line2.dxf.end
                slope2 = (end2[0] - start2[0]) / (end2[1] - start2[1]) if (end2[1] - start2[1]) != 0 else math.inf

                # Check if the lines have similar slopes and end/start points within epsilon
                if abs(slope1 - slope2) < epsilon and abs(end1 - start2) < epsilon:
                    if line1.dxf.layer == line2.dxf.layer:
                        msp1.add_line(start1, end2, dxfattribs={'layer': line1.dxf.layer})
                        msp1.delete_entity(line1)
                        msp3.delete_entity(line2)
                        break
    filepath1=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename1)   
    filepath2=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename2)  
    filepath3=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename3)               
    doc1.saveas(filepath1)
    doc2.saveas(filepath2)
    doc3.saveas(filepath3)

#----------------------------------------------------------------------------------------------------------------------


def Multiple_trim_for_X(df):
    #point_pairs = distribute_units_between_lines(add_horizontal_lines_new(testing), diff_Y)
    dis = 0
    for i in range(len(selected_pairs_for_X)): 
        selected_pairs_for_X[i] = (selected_pairs_for_X[i][0]-dis,selected_pairs_for_X[i][1]-dis)
        #print(selected_pairs_for_X)
        cut_x1 = selected_pairs_for_X[i][1]
        cut_x2 = selected_pairs_for_X[i][0]
        diff = cut_x2-cut_x1
        filename1 = 'lefttrim.dxf'
        filename2 = 'centertrim.dxf'
        filename3 = 'righttrim.dxf'
        trim_for_X(df, cut_x1, cut_x2,diff, filename1, filename2, filename3)
        filepath1=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename1)   
        filepath2=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename2)  
        filepath3=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename3)
        files = [filepath1,filepath3]
        lefttrimnew = Dxf_to_DF(files[0])
        righttrimnew = Dxf_to_DF(files[1])
        df = pd.concat((lefttrimnew,righttrimnew))
        dis += (selected_pairs_for_X[i][0]-selected_pairs_for_X[i][1]) 
    return df
#----------------------------------------------------------------------------------------------------------------------

def Multiple_trim_for_Y(df):
    #point_pairs = distribute_units_between_lines(add_horizontal_lines_new(testing), diff_Y)
    dis = 0
    for i in range(len(selected_pairs_for_Y)): 
        selected_pairs_for_Y[i] = (selected_pairs_for_Y[i][0]-dis,selected_pairs_for_Y[i][1]-dis)
        dis += (selected_pairs_for_Y[i][0]-selected_pairs_for_Y[i][1]) 
        #print(selected_pairs_for_Y)
        cut_y1 = selected_pairs_for_Y[i][1]
        cut_y2 = selected_pairs_for_Y[i][0]
        diff = cut_y2-cut_y1
        filename1 = 'belowcut.dxf'
        filename2 = 'betweencut.dxf'
        filename3 = 'abovecut.dxf'
        trim_dxf_for_Y(df, cut_y1, cut_y2,diff, filename1, filename2, filename3)
        filepath1=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename1)   
        filepath2=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename2)  
        filepath3=os.path.join(settings.BASE_DIR,'Temp','trimCache',project_name+filename3)
        files = [filepath1,filepath3]

        belowtrimnew = Dxf_to_DF(files[0])
        abovetrimnew = Dxf_to_DF(files[1])
        df = pd.concat((belowtrimnew,abovetrimnew))
    return df

def create_dxf_from_dataframe(df, output_filename):
    doc = ezdxf.new()
    
    # Create dictionary to store layers
    layers = {}
    
    for index, row in df.iterrows():
        layer_name = str(row['Layer'])  # Convert to string
        if not layer_name or layer_name == 'nan':  # Check for empty or NaN values
            continue
        
        if layer_name not in layers:
            if layer_name == '0':
                continue
            layers[layer_name] = doc.layers.new(name=layer_name)  # Create new layer if it doesn't exist
        
        msp = doc.modelspace()
        if row['Type'] == 'LINE':
            start = (row['X_start'], row['Y_start'])
            end = (row['X_end'], row['Y_end'])
            msp.add_line(start, end, dxfattribs={'layer': layer_name})
        elif row['Type'] == 'CIRCLE':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            msp.add_circle(center, radius, dxfattribs={'layer': layer_name})
        elif row['Type'] == 'ARC':
            center = (row['X_center'], row['Y_center'])
            radius = row['Radius']
            start_angle = row['Start Angle']
            end_angle = row['End Angle']
            msp.add_arc(center, radius, start_angle, end_angle, dxfattribs={'layer': layer_name})
        elif row['Type'] == 'TEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row['Text']
            msp.add_text(text, dxfattribs={'insert': insert, 'layer': layer_name})
        elif row['Type'] == 'MTEXT':
            insert = (row['X_insert'], row['Y_insert'])
            text = row['Text']
            msp.add_mtext(text, dxfattribs={'insert': insert, 'layer': layer_name})

    # Ensure the output directory exists and create if not
    output_directory = os.path.join(settings.MEDIA_ROOT, 'dxfs')
    # os.makedirs(output_directory, exist_ok=True)
    
    output_filepath = os.path.join(output_directory, os.path.basename(output_filename))
    
    doc.saveas(output_filepath)
    # print("Hello",output_filepath)
    return output_filepath


def floor_dist_line_test(df):
    max_along_x = np.round(np.max(np.abs(df['X_end'] - df['X_start'])), 2)
    max_along_y = np.round(np.max(np.abs(df['Y_end'] - df['Y_start'])), 2)

    max_x_df = df[df['Length'].round(4) == max_along_x]
    print(max_x_df)

    l1 = sorted(max_x_df['X_start'].unique())
    l2 = sorted(max_x_df['X_end'].unique())
    l1.extend(l2)
    sorted_list = sorted(list(set(l1)))
    
    ranges = [(sorted_list[i], sorted_list[i + 1]) for i in range(0, len(sorted_list) - 1, 2)]


    count = 0
    df_dict = {}

    for start,end in ranges:
        df_dict[f'floor_{count}'] = df[(df['X_start'] >= start) & (df['X_start'] <= end)]
        df_dict[f'floor_{count}']['floor'] = count
        count += 1

    floor_list= list(df_dict.values())
    floor_df = pd.concat(floor_list)

    return floor_df

def floor_dist_mtext_test(df):
    max_along_x = np.round(np.max(np.abs(df['X_end'] - df['X_start'])), 2)
    max_along_y = np.round(np.max(np.abs(df['Y_end'] - df['Y_start'])), 2)

    max_x_df = df[df['Length'].round(4) == max_along_x]
    
    l1 = sorted(max_x_df['X_start'].unique())
    l2 = sorted(max_x_df['X_end'].unique())
    l1.extend(l2)
    sorted_list = sorted(list(set(l1)))
    
    ranges = [(sorted_list[i], sorted_list[i + 1]) for i in range(0, len(sorted_list) - 1, 2)]
     

    count = 0
    df_dict = {}

    for start,end in ranges:
        df_dict[f'floor_{count}'] = df[(df['X_insert'] >= (start - 1)) & (df['X_insert'] <= (end + 1))]
        df_dict[f'floor_{count}']['floor'] = count
        count += 1

    floor_list= list(df_dict.values())
    floor_df = pd.concat(floor_list)

    return floor_df

def floor_main(df):
    line = floor_dist_line_test(df)
    mtext = floor_dist_mtext_test(df)
    
    floor = pd.concat([line, mtext])
    
    return floor



def calculate_length1(start, end):
    return math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2 + (end.z - start.z)**2) / 12  # Convert to feet

def Dxf_to_DF1(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()

    entities_data = []
    for entity in msp:
        entity_data = {'Type': entity.dxftype(), 'Layer': entity.dxf.layer}
        if entity.dxftype() == 'LINE':
            start = entity.dxf.start
            end = entity.dxf.end
            length = calculate_length1(start, end)
            entity_data.update({
                'X_start': start.x, 'Y_start': start.y, 'Z_start': start.z,
                'X_end': end.x, 'Y_end': end.y, 'Z_end': end.z,
                'Length': length})
            
            # Feature Engineering: Horizontal and Vertical
            horizontal = abs(end.x - start.x) > abs(end.y - start.y)
            vertical = not horizontal
            entity_data.update({'Horizontal': horizontal, 'Vertical': vertical})
            
        elif entity.dxftype() == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius / 12  # Convert to feet
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Radius': radius})
            
        elif entity.dxftype() == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius / 12  # Convert to feet
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            entity_data.update({
                'X_center': center.x, 'Y_center': center.y, 'Z_center': center.z,
                'Size': radius,
                'Start Angle': start_angle,
                'End Angle': end_angle})
            
        elif entity.dxftype() == 'TEXT':
            insert = entity.dxf.insert
            text = entity.dxf.text
            entity_data.update({
                'X_insert': insert.x, 'Y_insert': insert.y, 'Z_insert': insert.z,
                'Text': text})
            
        elif entity.dxftype() == 'MTEXT':
            text = entity.plain_text()
            insertion_point = entity.dxf.insert
            entity_data.update({
                'Text': text,
                'X_insert': insertion_point.x,
                'Y_insert': insertion_point.y,
                'Z_insert': insertion_point.z
    } )          
        entities_data.append(entity_data)
    
    return pd.DataFrame(entities_data)


def process_and_aggregate(df):
      
    grouped = df.groupby('Layer')

    # Initialize lists to store the aggregated data
    layer_names = []
    lengths = []
    widths = []

    # Iterate over each group
    for layer_name, group in grouped:
        # Store the layer name
        layer_names.append(layer_name)

        # Filter out only the LINE entities for this layer
        lines = group[group['Type'] == 'LINE']

        # Check if there are any lines in the group
        if not lines.empty:
            # Calculate maximum vertical and horizontal lengths
            vertical_length = lines[lines['Vertical']]['Length'].max()
            horizontal_length = lines[lines['Horizontal']]['Length'].max()
        else:
            # If no lines present, assign NaN values
            vertical_length = float('nan')
            horizontal_length = float('nan')

        # Append the maximum lengths to the lists
        lengths.append(vertical_length)
        widths.append(horizontal_length)

    # Create a DataFrame from the aggregated data
    sizedata = pd.DataFrame({
        'LayerName': layer_names,
        'Length': lengths,
        'Width': widths
    })
    sizedata['Area'] = sizedata['Length']*sizedata['Width']

    return sizedata

def merge_layer_constraints(layer_df: pd.DataFrame, constraints_file: str = 'General Constraints - Sheet1.csv') -> pd.DataFrame:

    # Read the constraints data from the specified CSV file
    csv_dir = settings.BASE_DIR / 'assets'
    csv_path = csv_dir / 'General Constraints - Sheet1 (1).csv'
    constraints_df = pd.read_csv(csv_path)

    # Define aggregation functions for the 'Min Dim' and 'Max Dim' columns
    aggregation_functions = {
        'Min Dim': 'first',
        'Max Dim': 'first'
    }
    
    # Group the constraints data by 'LayerName' and 'Dimension Type', applying the aggregation functions
    grouped_constraints = constraints_df.groupby(['LayerName', 'Dimension Type'], as_index=False).agg(aggregation_functions)
    
    # Merge the grouped constraints with the input DataFrame on 'LayerName'
    merged_df = pd.merge(grouped_constraints, layer_df, on='LayerName', how='inner')
    
    # Return the merged DataFrame
    return merged_df


def check_conditions(layer_df: pd.DataFrame) -> pd.DataFrame:

    results = []

    for layer_name in layer_df['LayerName'].unique():
        # Filter data for the current layer
        layer_data = layer_df[layer_df['LayerName'] == layer_name]

        # Initialize constraint violated list and error variables
        constraints_violated = []
        error_area = None
        error_length = None
        error_width = None

        # Check Area
        area_row = layer_data[layer_data['Dimension Type'] == 'Area']
        if not area_row.empty:
            area_value = area_row['Area'].values[0]
            min_dim_area = area_row['Min Dim'].values[0]
            max_dim_area = area_row['Max Dim'].values[0]
            condition_area = min_dim_area <= area_value <= max_dim_area
        else:
            condition_area = False
        if not condition_area:
            constraints_violated.append('Area')
            if area_value < min_dim_area:
                error_area = area_value - min_dim_area
            else:
                error_area = area_value - max_dim_area

        # Check Length
        length_value = layer_data.iloc[0]['Length']  # Get the 'Length' value for the current layer
        condition_length = False

        for index, row in layer_data.iterrows():
            if row['Dimension Type'] == 'Length':
                min_dim_length = row['Min Dim']
                max_dim_length = row['Max Dim']
                if min_dim_length <= length_value <= max_dim_length:
                    condition_length = True
                    break

        if not condition_length:
            for index, row in layer_data.iterrows():
                if row['Dimension Type'] == 'Width':
                    min_dim_length = row['Min Dim']
                    max_dim_length = row['Max Dim']
                    if min_dim_length <= length_value <= max_dim_length:
                        condition_length = True
                        break

        if not condition_length:
            constraints_violated.append('Length')
            for index, row in layer_data.iterrows():
                if row['Dimension Type'] == 'Length':
                    min_dim_length = row['Min Dim']
                    max_dim_length = row['Max Dim']
                    if length_value < min_dim_length:
                        error_length = length_value - min_dim_length
                    else:
                        error_length = length_value - max_dim_length

        # Check Width
        width_value = layer_data.iloc[0]['Width']  # Get the 'Width' value for the current layer
        condition_width = False

        for index, row in layer_data.iterrows():
            if row['Dimension Type'] == 'Width':
                min_dim_width = row['Min Dim']
                max_dim_width = row['Max Dim']
                if min_dim_width <= width_value <= max_dim_width:
                    condition_width = True
                    break
                
        if not condition_width:
            for index, row in layer_data.iterrows():
                if row['Dimension Type'] == 'Length':
                    min_dim_width = row['Min Dim']
                    max_dim_width = row['Max Dim']
                    if min_dim_width <= width_value <= max_dim_width:
                        condition_width = True
                        break

        if not condition_width:
            constraints_violated.append('Width')
            if width_value < min_dim_width:
                error_width = width_value - min_dim_width
            else:
                error_width = width_value - max_dim_width

        # Append results for the current layer
        results.append({
            'LayerName': layer_name,
            'Condition_Area': condition_area,
            'Condition_Length': condition_length,
            'Condition_Width': condition_width,
            'Constraints_Violated': constraints_violated,
            'Error_Area': error_area,
            'Error_Length': error_length,
            'Error_Width': error_width
        })

    return pd.DataFrame(results)


def calculate_percentage(df, column_name):
    true_count = df[column_name].sum()
    false_count = len(df) - true_count
    total_count = len(df)
    true_percentage = (true_count / total_count) * 100
    false_percentage = (false_count / total_count) * 100
    return true_percentage, false_percentage



def constrains(filename1):
    sizedata1 = process_and_aggregate(Dxf_to_DF1(filename1))
    grouped_df1 = merge_layer_constraints(sizedata1)
    result_df1 = check_conditions(grouped_df1)
    
    return result_df1


#Carpet Area addons


def line_exists(point1, point2, lines):
    new_line = LineString([point1, point2])
    for line in lines:
        if new_line.equals(line):
            return True
    return False

def find_pairs_with_shortest_distance(unclosed_points, existing_lines):
    pairs = []
    used_indices = set()
    
    for i, point1 in enumerate(unclosed_points):
        if i in used_indices:
            continue
        min_dist = float('inf')
        closest_point_index = None
        
        for j, point2 in enumerate(unclosed_points):
            if i != j and j not in used_indices:
                dist = distance.euclidean((point1.x, point1.y), (point2.x, point2.y))
                if dist < min_dist and not line_exists(point1, point2, existing_lines):
                    min_dist = dist
                    closest_point_index = j
                    
        if closest_point_index is not None:
            pairs.append((point1, unclosed_points[closest_point_index]))
            used_indices.add(i)
            used_indices.add(closest_point_index)
    
    return pairs

def area_extraction(dataframe):
    layers = dataframe['Layer'].unique()
    layer_dict = {}
    
    for idx, layer in enumerate(layers):
        if "Door" in layer or "Main_gate" in layer:
            continue 
        try:
            coord_df = np.round(dataframe[(dataframe['Type'] == 'LINE') & (dataframe['Layer'] == f'{layer}')][['X_start','Y_start','X_end','Y_end']], 4)
            pts = [[Point(rows['X_start'], rows['Y_start']), Point(rows['X_end'], rows['Y_end'])] for idx, rows in coord_df.iterrows()]
            ln = [LineString([i[0], i[-1]]) for i in pts]
            start = [(x, y) for x, y in zip(coord_df['X_start'], coord_df['Y_start'])]
            end = [(x, y) for x, y in zip(coord_df['X_end'], coord_df['Y_end'])]
            comb = start + end
            coord_counts = {}
            for coord in comb:
                if coord in coord_counts:
                    coord_counts[coord] += 1
                else:
                    coord_counts[coord] = 1
            unclosed = [val for val, idx in coord_counts.items() if idx == 1]
            upts = [Point(i[0], i[1]) for i in unclosed]
            imgpts = find_pairs_with_shortest_distance(upts, ln)
            imgln = [LineString([i[0], i[-1]]) for i in imgpts]
            linestrings = imgln + ln
            merged_lines = linemerge(linestrings)
            
            if isinstance(merged_lines, LineString):
                if merged_lines.is_closed and len(merged_lines.coords) >= 30:
                    polygon = Polygon(merged_lines)
                else:
                    polygon = MultiPoint(merged_lines.coords).convex_hull
            elif isinstance(merged_lines, MultiLineString):
                polygon = unary_union(merged_lines)
                if not isinstance(polygon, Polygon):
                    endpoints = []
                    for line in merged_lines.geoms:
                        endpoints.extend(line.coords)
                    polygon = MultiPoint(endpoints).convex_hull
            else:
                print(f"Unexpected geometry type after merging for layer {layer}")
                layer_dict[f'{layer}'] = 0
                continue
            
            layer_dict[f'{layer}'] = polygon.area / 144
        
        except Exception as e:
            print(f"An error occurred while processing layer {layer}: {str(e)}")
            layer_dict[f'{layer}'] = 0  # Assign 0 or any other default value for the layer that caused an error
    
    return layer_dict


def area_extraction_for_layer(dataframe, layer):
    try:
        coord_df = np.round(dataframe[(dataframe['Type'] == 'LINE') & (dataframe['Layer'] == f'{layer}')][['X_start','Y_start','X_end','Y_end']], 4)
        pts = [[Point(rows['X_start'], rows['Y_start']), Point(rows['X_end'], rows['Y_end'])] for idx, rows in coord_df.iterrows()]
        ln = [LineString([i[0], i[-1]]) for i in pts]
        start = [(x, y) for x, y in zip(coord_df['X_start'], coord_df['Y_start'])]
        end = [(x, y) for x, y in zip(coord_df['X_end'], coord_df['Y_end'])]
        comb = start + end
        coord_counts = {}
        for coord in comb:
            if coord in coord_counts:
                coord_counts[coord] += 1
            else:
                coord_counts[coord] = 1
        unclosed = [val for val, idx in coord_counts.items() if idx == 1]
        upts = [Point(i[0], i[1]) for i in unclosed]
        imgpts = find_pairs_with_shortest_distance(upts, ln)
        imgln = [LineString([i[0], i[-1]]) for i in imgpts]
        linestrings = imgln + ln
        merged_lines = linemerge(linestrings)
        
        if isinstance(merged_lines, LineString):
            if merged_lines.is_closed and len(merged_lines.coords) >= 30:
                polygon = Polygon(merged_lines)
            else:
                polygon = MultiPoint(merged_lines.coords).convex_hull
        elif isinstance(merged_lines, MultiLineString):
            polygon = unary_union(merged_lines)
            if not isinstance(polygon, Polygon):
                endpoints = []
                for line in merged_lines.geoms:
                    endpoints.extend(line.coords)
                polygon = MultiPoint(endpoints).convex_hull
        else:
            # Handle unexpected geometry type
            print(f"Unexpected geometry type after merging for layer {layer}")
            return 0
        
        return (polygon.area) / 144
    except Exception as e:
        print(f"An error occurred while processing layer {layer}: {str(e)}")
        return 0




def latest_carpet_area(dataframe):
    area = 0

    not_carpet_layer = ['Parking','Stairscase','FoyerStairs','Boundary','Entrance_Staircase','Setback','0',
                        'BoundaryWalls','Garden','Foyer','Stairs','FoyerStair','WashArea_Staircase']

    for i in list(dataframe.Layer.unique()):
        if i not in not_carpet_layer:
            area += area_extraction_for_layer(dataframe, f'{i}')
            
    return area




def latest_build_area(dataframe):
    area = 0

    not_build_layer = ['Boundary','0','Balcony1','Balcony2','Balcony3','Terrace','Parking','Garden','OTS','Entrance_Staircase','Setback']

    for i in list(dataframe.Layer.unique()):
        if i not in not_build_layer:
            area += area_extraction_for_layer(dataframe, f'{i}')
            
    return area



def area_main(dataframe):
    main_dict = area_extraction(dataframe)
    carpet_area = round(latest_carpet_area(dataframe),2)
    build_up_area = round((latest_carpet_area(dataframe)  * 1.3),2)
    
    major_area_dict = {
        'Carpet_Area': carpet_area,
        'Build_up_Area': build_up_area
    }
    
    main_dict = {key: round(value, 2) for key, value in main_dict.items()}
    main_dict['Major_Areas'] = major_area_dict
    keys_to_remove = ['Boundary', '0']
    for key in keys_to_remove:
        if key in main_dict:
            del main_dict[key]
    print(f"AREA:{json.dumps(main_dict)}")
    return main_dict

def plot_dxf(filename):
    """
    Plots entities from a DXF file using matplotlib.

    Parameters:
        filename(str): The filename of the DXF file to be plotted.

    Returns:
        None
    """
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()

    fig, ax = plt.subplots()
    for entity in msp:
        if entity.dxftype() == 'LINE':
            start = (entity.dxf.start.x / 12, entity.dxf.start.y / 12)  # Convert inches to feet
            end = (entity.dxf.end.x / 12, entity.dxf.end.y / 12)  # Convert inches to feet
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)
        elif entity.dxftype() == 'CIRCLE':
            center = (entity.dxf.center.x / 12, entity.dxf.center.y / 12)  # Convert inches to feet
            radius = entity.dxf.radius / 12  # Convert inches to feet
            circle = plt.Circle(center, radius, color='black', fill=False, linewidth=1)
            ax.add_artist(circle)
        elif entity.dxftype() == 'ARC':
            center = (entity.dxf.center.x / 12, entity.dxf.center.y / 12)  # Convert inches to feet
            radius = entity.dxf.radius / 12  # Convert inches to feet
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)
            arc = plt.Arc(center, 2 * radius, 2 * radius, 0, math.degrees(start_angle), math.degrees(end_angle), color='black', linewidth=1)
            ax.add_artist(arc)
        elif entity.dxftype() == 'TEXT':
            insert = (entity.dxf.insert.x / 12, entity.dxf.insert.y / 12)  # Convert inches to feet
            text = entity.dxf.text
            ax.text(insert[0], insert[1], text, fontsize=4, color='black')
        elif entity.dxftype() == 'MTEXT':
            insert = (entity.dxf.insert.x / 12, entity.dxf.insert.y / 12)  # Convert inches to feet
            text = entity.dxf.text
            ax.text(insert[0], insert[1], text, fontsize=2, color='black')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (feet)')
    ax.set_ylabel('Y (feet)')
    ax.set_title('Architectural Plan')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', direction='inout', which='both')
    
    # Set x and y axis ticks in feet
    x_ticks = [i * 10 for i in range(int(ax.get_xlim()[0] / 10), int(ax.get_xlim()[1] / 10) + 1)]
    y_ticks = [i * 10 for i in range(int(ax.get_ylim()[0] / 10), int(ax.get_ylim()[1] / 10) + 1)]
    #plt.xlim(-20 , 10)
    #plt.ylim(-70, -55)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    #API Helper
    png_folder = os.path.join(settings.MEDIA_ROOT, 'pngs')
    # if not os.path.exists(png_folder):
    #     os.makedirs(png_folder)
    new_filename = filename.replace('.dxf', '.png')
    print('new_filename:',new_filename)
    png_filepath = os.path.join(png_folder,os.path.basename(new_filename))
    print('png_filepath:',png_filepath)
    fig.savefig(png_filepath, bbox_inches='tight')
    plt.close(fig)
    # print('Hello',png_filepath)
    return png_filepath


def plot_dataframe(df,inmage_name):
    """
    Plots entities from a DataFrame using matplotlib.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the entities to be plotted.

    Returns:
        None
    """
    fig, ax = plt.subplots()

    for _, entity in df.iterrows():
        if entity['Type'] == 'LINE':
            start = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Convert inches to feet
            end = (entity['X_end'] / 12, entity['Y_end'] / 12)  # Convert inches to feet
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=1)
        elif entity['Type'] == 'CIRCLE':
            center = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Assuming center coordinates are in X_start, Y_start
            radius = entity['Length'] / 12  # Assuming Length column represents radius for circles
            circle = plt.Circle(center, radius, color='black', fill=False, linewidth=1)
            ax.add_artist(circle)
        elif entity['Type'] == 'ARC':
            center = (entity['X_start'] / 12, entity['Y_start'] / 12)  # Assuming center coordinates are in X_start, Y_start
            radius = entity['Length'] / 12  # Assuming Length column represents radius for arcs
            start_angle = math.radians(entity['X_end'])  # Assuming start angle is in X_end
            end_angle = math.radians(entity['Y_end'])  # Assuming end angle is in Y_end
            arc = plt.Arc(center, 2 * radius, 2 * radius, 0, math.degrees(start_angle), math.degrees(end_angle), color='black', linewidth=1)
            ax.add_artist(arc)
        elif entity['Type'] == 'TEXT':
            insert = (entity['X_insert'] / 12, entity['Y_insert'] / 12)  # Convert inches to feet
            text = entity['Text']
            ax.text(insert[0], insert[1], text, fontsize=4, color='black')
        elif entity['Type'] == 'MTEXT':
            insert = (entity['X_insert'] / 12, entity['Y_insert'] / 12)  # Convert inches to feet
            text = entity['Text']
            ax.text(insert[0], insert[1], text, fontsize=2, color='black')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X (feet)')
    ax.set_ylabel('Y (feet)')
    ax.set_title('Architectural Plan')
    ax.tick_params(axis='both', direction='inout', which='both')

    # Set x and y axis ticks in feet
    x_ticks = [i * 10 for i in range(int(ax.get_xlim()[0] / 10), int(ax.get_xlim()[1] / 10) + 1)]
    y_ticks = [i * 10 for i in range(int(ax.get_ylim()[0] / 10), int(ax.get_ylim()[1] / 10) + 1)]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    
    png_folder1 = os.path.join(settings.MEDIA_ROOT,'pngs')    
    # if not os.path.exists(png_folder1):
    #     os.makedirs(png_folder1)
    #new_filename = filename.replace('.dxf', '.png')
    print('new_filename:',inmage_name)
    png_filepath1 = os.path.join(png_folder1,inmage_name)
    print('png_filepath:',png_filepath1)
    #plt.savefig(png_filepath1, dpi=300)
    fig.savefig(png_filepath1, bbox_inches='tight')
    plt.close(fig)
    print('Hello',png_filepath1)
    return png_filepath1

    
# #########################################################################


def stairecase_manage(df):
    mtext_df = df[df['Type'] == 'MTEXT'].copy()
    line_df = df[(df['Type'] == 'LINE') & (df['Layer'] == 'Staircase')].copy()

    # Initialize a set to keep track of assigned mtexts
    assigned_mtexts = set()

    # Function to find the nearest available MTEXT that hasn't been assigned yet
    def find_nearest_mtext(x, y, assigned_mtexts):
        # Calculate the distance to each MTEXT
        distances = np.sqrt((mtext_df['X_insert'] - x)**2 + (mtext_df['Y_insert'] - y)**2)
        # Sort distances and find the nearest available MTEXT
        for idx in distances.sort_values().index:
            if mtext_df.loc[idx, 'Text'] not in assigned_mtexts:
                assigned_mtexts.add(mtext_df.loc[idx, 'Text'])
                return mtext_df.loc[idx, 'Text']
        return None

    # Assign MTEXT to each Staircase LINE, avoiding duplicates
    for index, row in line_df.iterrows():
        x_center = (row['X_start'] + row['X_end']) / 2
        y_center = (row['Y_start'] + row['Y_end']) / 2
        nearest_mtext = find_nearest_mtext(x_center, y_center, assigned_mtexts)
        df.at[index, 'Staircase_number'] = nearest_mtext

    # Fill NaN values with a placeholder (e.g., 0) and convert to integer
    df['Staircase_number'] = df['Staircase_number'].fillna('Stair 0').apply(lambda x: int(x.split()[-1]) if isinstance(x, str) else 0)

    # Sort the DataFrame by 'Staircase_number'
    df = df.sort_values(by='Staircase_number')
    staircase_layer = df[df['Staircase_number']>0]
    # Display the final DataFrame
    return staircase_layer

def create_iron_main_gate(width=3, height=7, thickness=2, color='black'):
    fig = go.Figure()

    # Gate frame
    frame_thickness = 2
    for x in [0, width/2, width]:
        fig.add_trace(go.Mesh3d(
            x=[x, x, x, x],
            y=[0, thickness, thickness, 0],
            z=[0, 0, height, height],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=1
        ))
    for z in [0, height]:
        fig.add_trace(go.Mesh3d(
            x=[0, width, width, 0],
            y=[0, 0, thickness, thickness],
            z=[z, z, z, z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=1
        ))

    # Vertical bars
    num_bars = 8  # 4 bars per half
    bar_width = 0.5
    for i in range(num_bars):
        x = width * (i + 0.5) / num_bars
        fig.add_trace(go.Mesh3d(
            x=[x-bar_width/2, x+bar_width/2, x+bar_width/2, x-bar_width/2],
            y=[thickness/2, thickness/2, thickness/2, thickness/2],
            z=[frame_thickness, frame_thickness, height-frame_thickness, height-frame_thickness],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='#964B00', opacity=1
        ))

    # Decorative elements (circles at the top)
    circle_radius = 10
    num_circles = num_bars - 1
    for i in range(num_circles):
        x = width * (i + 1) / num_bars
        z = height - frame_thickness - circle_radius
        theta = np.linspace(0, 2*np.pi, 20)
        fig.add_trace(go.Mesh3d(
            x=x + circle_radius*np.cos(theta),
            y=[thickness/2]*20,
            z=z + circle_radius*np.sin(theta),
            i=[0]*18, j=list(range(1, 19)), k=list(range(2, 20)) + [1],
            color='gold', opacity=1
        ))

    # Center handles
    handle_radius = 10
    handle_height = height / 2
    handle_depth = thickness + 0.2
    for x in [width/2 - 0.1, width/2 + 0.1]:  # Two handles, slightly offset from center
        fig.add_trace(go.Mesh3d(
            x=[x]*20 + [x-handle_radius]*20,
            y=[handle_depth*np.cos(t) for t in np.linspace(0, 2*np.pi, 20)] + 
              [handle_depth*np.cos(t) for t in np.linspace(0, 2*np.pi, 20)],
            z=[handle_height + handle_radius*np.sin(t) for t in np.linspace(0, 2*np.pi, 20)] + 
              [handle_height + handle_radius*np.sin(t) for t in np.linspace(0, 2*np.pi, 20)],
            i=[0, 0] + list(range(20, 39)), j=[1, 20] + list(range(21, 40)),
            k=[20, 21] + list(range(22, 41)),
            color='gold', opacity=1
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Thickness',
            zaxis_title='Height',
            aspectmode='data'
        ),
        title='3D Double Iron Main Gate',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig



def create_wooden_door(width=3, height=7, thickness=0.2, color='saddlebrown'):
    door_traces = []
    # Door frame
    frame_thickness = 0.3
    for x in [0, width]:
        door_traces.append(go.Mesh3d(
            x=[x, x, x, x],
            y=[0, thickness, thickness, 0],
            z=[0, 0, height, height],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=1
        ))
    for z in [0, height]:
        door_traces.append(go.Mesh3d(
            x=[0, width, width, 0],
            y=[0, 0, thickness, thickness],
            z=[z, z, z, z],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color=color, opacity=1
        ))
    # Door panel
    door_traces.append(go.Mesh3d(
        x=[frame_thickness, width-frame_thickness, width-frame_thickness, frame_thickness],
        y=[thickness/2, thickness/2, thickness/2, thickness/2],
        z=[frame_thickness, frame_thickness, height-frame_thickness, height-frame_thickness],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=1
    ))
    # Door handle
    handle_radius = 0.1
    handle_height = height / 2
    handle_depth = thickness + 0.2
    door_traces.append(go.Mesh3d(
        x=[width-0.5]*20 + [width-0.5-handle_radius]*20,
        y=[handle_depth*np.cos(t) for t in np.linspace(0, 2*np.pi, 20)] + 
          [handle_depth*np.cos(t) for t in np.linspace(0, 2*np.pi, 20)],
        z=[handle_height + handle_radius*np.sin(t) for t in np.linspace(0, 2*np.pi, 20)] + 
          [handle_height + handle_radius*np.sin(t) for t in np.linspace(0, 2*np.pi, 20)],
        i=[0, 0] + list(range(20, 39)), j=[1, 20] + list(range(21, 40)),
        k=[20, 21] + list(range(22, 41)),
        color='gold', opacity=1
    ))
    # Wood grain lines
    num_lines = 10
    for i in range(num_lines):
        z = height * (i + 1) / (num_lines + 1)
        door_traces.append(go.Scatter3d(
            x=[frame_thickness, width-frame_thickness],
            y=[thickness/2, thickness/2],
            z=[z, z],
            mode='lines',
            line=dict(color='sienna', width=2),
            showlegend=False
        ))
    return door_traces

def add_plane(fig, x1, y1, x2, y2, height_low, height_high, color, layer_name):
    if layer_name in ['Garden', 'Parking', 'Entrance_Staircase', 'Garden_Staircase']:
        # Add only the outline of the area at the base
        fig.add_trace(go.Scatter3d(
            x=[x1, x2, x2, x1, x1],
            y=[y1, y1, y2, y2, y1],
            z=[height_low, height_low, height_low, height_low, height_low],
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='text',
            hovertext=f'Layer: {layer_name}',
            showlegend=False
        ))
    elif layer_name == 'Balcony' or layer_name == 'Main_gate':
        # For balcony, show lines including height and 5 equally spaced horizontal lines
        # Calculate heights for 5 equally spaced lines
        heights = [height_low + i * (height_high - height_low) / 6 for i in range(7)]
        
        for height in heights:
            fig.add_trace(go.Scatter3d(
                x=[x1, x2, x2, x1, x1],
                y=[y1, y1, y2, y2, y1],
                z=[height, height, height, height, height],
                mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='text',
                hovertext=f'Layer: {layer_name}',
                showlegend=False
            ))
        
        # Add vertical lines at the corners
        for x, y in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[height_low, height_high],
                mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='text',
                hovertext=f'Layer: {layer_name}',
                showlegend=False
            ))

    else:
        # Create the vertical plane for other layers
        fig.add_trace(go.Mesh3d(
            x=[x1, x2, x2, x1],
            y=[y1, y2, y2, y1],
            z=[height_low, height_low, height_high, height_high],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color=color,
            hoverinfo='text',
            hovertext=f'Layer: {layer_name}',
            opacity=1
        ))
        # Add the edges of the plane for better visibility
        fig.add_trace(go.Scatter3d(
            x=[x1, x2, x2, x1, x1],
            y=[y1, y2, y2, y1, y1],
            z=[height_low, height_low, height_high, height_high, height_low],
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='text',
            hovertext=f'Layer: {layer_name}',
            showlegend=False
        ))
        # Add vertical lines at the corners
        for x, y in [(x1, y1), (x2, y2)]:
            fig.add_trace(go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[height_low, height_high],
                mode='lines',
                line=dict(color=color, width=2),
                hoverinfo='text',
                hovertext=f'Layer: {layer_name}',
                showlegend=False
            ))

def add_text(fig, x, y, z, text, color):
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='text',
        text=[text],
        textfont=dict(color='black', size=10),
        showlegend=False
    ))

def plot_2d_boundary(fig, floor, gap, z_value, i,no_of_floors):
    if i == 0:
        boundary_data = floor[floor['Layer'] == 'Boundary']

        if boundary_data.empty:
            print(f"No boundary data found for floor {i}")
            return

        # Calculate the min and max coordinates for the boundary layer
        min_x = boundary_data['X_start'].min()
        max_x = boundary_data['X_start'].max()
        min_y = boundary_data['Y_start'].min()
        max_y = boundary_data['Y_start'].max()

        # Use the z_value and i to set the correct height for this floor's boundary
        z = z_value * i

        # Plot boundary lines
        for _, row in boundary_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['X_start'], row['X_end']],
                y=[row['Y_start'], row['Y_end']],
                z=[z, z],  # Use the calculated z value for both start and end
                mode='lines',
                line=dict(color='#292323', width=2),
                showlegend=False
            ))

        # Optionally, you can add a floor plane
        fig.add_trace(go.Mesh3d(
            x=[min_x-100, max_x+100, max_x+100, min_x-100],
            y=[min_y-100, min_y-100, max_y+100, max_y+100],
            z=[z, z, z, z],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='#41980a',
            opacity=0.4
        ))
        for _, row in boundary_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['X_start'], row['X_end']],
                y=[row['Y_start'], row['Y_end']],
                z=[z, z],  # Use the calculated z value for both start and end
                mode='lines',
                line=dict(color='#292323', width=2),
                showlegend=False
            ))

        # Optionally, you can add a floor plane
        fig.add_trace(go.Mesh3d(
            x=[min_x, max_x, max_x, min_x],
            y=[min_y, min_y, max_y, max_y],
            z=[z, z, z, z],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            color='Black',
            opacity=0.6
        ))
    else:
        boundary_data = floor[floor['Layer'] != 'Boundary']
        if boundary_data.empty:
            print(f"No boundary data found for floor {i}")
            return
        # Calculate the min and max coordinates for the boundary layer
        min_x = boundary_data['X_start'].min()
        max_x = boundary_data['X_start'].max()
        min_y = boundary_data['Y_start'].min()
        max_y = boundary_data['Y_start'].max()
        # Use the z_value and i to set the correct height for this floor's boundary
        z = z_value * i
        z_up = z + 6  # Set the height of the cuboid to 8 units
        z_up2 = z+102
        z_up3 = z+108
        # Plot boundary lines
        for _, row in boundary_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['X_start'], row['X_end']],
                y=[row['Y_start'], row['Y_end']],
                z=[z, z],  # Use the calculated z value for both start and end
                mode='lines',
                line=dict(color='#808080', width=2),
                showlegend=False
            ))

        # Create a 3D cuboid instead of a 2D plane
        vertices = np.array([
            [min_x, min_y, z],
            [max_x, min_y, z],
            [max_x, max_y, z],
            [min_x, max_y, z],
            [min_x, min_y, z_up],
            [max_x, min_y, z_up],
            [max_x, max_y, z_up],
            [min_x, max_y, z_up]
        ])

        I = [0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
        J = [1, 2, 6, 5, 6, 5, 2, 3, 7, 6, 7, 6, 3, 0, 4, 7, 4, 7, 0, 1, 5, 4, 5, 4]
        K = [0, 1, 5, 1, 4, 0, 1, 2, 6, 2, 5, 1, 2, 3, 7, 3, 6, 2, 3, 0, 4, 0, 7, 3]

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=I, j=J, k=K,
            color='#808080',
            opacity=1))
        
        for _, row in boundary_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['X_start'], row['X_end']],
                y=[row['Y_start'], row['Y_end']],
                z=[z, z],  # Use the calculated z value for both start and end
                mode='lines',
                line=dict(color='#808080', width=2),
                showlegend=False
            ))

        # Create a 3D cuboid instead of a 2D plane
        vertices = np.array([
            [min_x, min_y, z_up2],
            [max_x, min_y, z_up2],
            [max_x, max_y,z_up2],
            [min_x, max_y, z_up2],
            [min_x, min_y, z_up3],
            [max_x, min_y, z_up3],
            [max_x, max_y, z_up3],
            [min_x, max_y, z_up3]
        ])

        I = [0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
        J = [1, 2, 6, 5, 6, 5, 2, 3, 7, 6, 7, 6, 3, 0, 4, 7, 4, 7, 0, 1, 5, 4, 5, 4]
        K = [0, 1, 5, 1, 4, 0, 1, 2, 6, 2, 5, 1, 2, 3, 7, 3, 6, 2, 3, 0, 4, 0, 7, 3]

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=I, j=J, k=K,
            color='#808080',
            opacity=1))
        
    if no_of_floors == 1:
        print('no_of_floors',no_of_floors)
        boundary_data = floor[floor['Layer'] == 'Boundary']
        if boundary_data.empty:
            print(f"No boundary data found for floor {i}")
            return
        # Calculate the min and max coordinates for the boundary layer
        min_x = boundary_data['X_start'].min()
        max_x = boundary_data['X_start'].max()
        min_y = boundary_data['Y_start'].min()
        max_y = boundary_data['Y_start'].max()
        # Use the z_value and i to set the correct height for this floor's boundary
        z = z_value * i  # Set the height of the cuboid to 8 units
        z_up2 = z+108
        z_up3 = z_up2+6
        # Plot boundary lines
        for _, row in boundary_data.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row['X_start'], row['X_end']],
                y=[row['Y_start'], row['Y_end']],
                z=[z, z],  # Use the calculated z value for both start and end
                mode='lines',
                line=dict(color='#808080', width=2),
                showlegend=False
            ))

        # Create a 3D cuboid instead of a 2D plane
        vertices = np.array([
            [min_x, min_y, z_up2],
            [max_x, min_y, z_up2],
            [max_x, max_y, z_up2],
            [min_x, max_y, z_up2],
            [min_x, min_y, z_up3],
            [max_x, min_y, z_up3],
            [max_x, max_y, z_up3],
            [min_x, max_y, z_up3]
        ])

        I = [0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
        J = [1, 2, 6, 5, 6, 5, 2, 3, 7, 6, 7, 6, 3, 0, 4, 7, 4, 7, 0, 1, 5, 4, 5, 4]
        K = [0, 1, 5, 1, 4, 0, 1, 2, 6, 2, 5, 1, 2, 3, 7, 3, 6, 2, 3, 0, 4, 0, 7, 3]

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=I, j=J, k=K,
            color='#808080',
            opacity=1))



def final_3d(df, number):
    layer_colors = {
        'Garden': '#90EE90', 
        'WashArea': '#C5C6C7',
        'WashArea_Staircase': '#C5C6C7', 
        'Parking': '#964B00', 
        'DiningRoom': '#C5C6C7', 
        'LivingRoom1': '#C5C6C7', 
        'BathRoom1': '#C5C6C7', 
        'BedRoom1': '#C5C6C7', 
        'Kitchen': '#C5C6C7',
        'PoojaRoom': '#C5C6C7', 
        'BedRoom3': '#C5C6C7', 
        'BedRoom2': '#C5C6C7', 
        'Balcony': '#964B00', 
        'BathRoom2a': '#C5C6C7',
        'BedRoom2a': '#C5C6C7', 
        'BathRoom2b': '#C5C6C7',
        'BathRoom2': '#C5C6C7', 
        'LivingRoom2': '#C5C6C7', 
        '0': 'Red',
        'Terrace': '#C5C6C7',
        'Door4fPanel': '#964B00', 
        'Door3fPanel': '#964B00'
        , 'Door2.6fPanel': '#964B00',
        'Staircase_wall2':'#C5C6C7',
        'Staircase_outertwall':'#C5C6C7',
        'Main_gate': '#964B00',
        'Pantry':'#C5C6C7',
        'DrawingRoom':'#C5C6C7',
        'Office':'#C5C6C7',
        'StoreRoom':'#C5C6C7',
        'BedRoom3a': '#C5C6C7'
    }

    df = floor_main(df)
    df = df.set_index('floor').sort_index()
    Gaps = []
    unique_floors = df.index.unique()

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    for idx, i in enumerate(unique_floors):
        no_of_floors = len(unique_floors)
        print('no_of_floors2',no_of_floors)
        floor = df.loc[i]
        floor = floor.reset_index(drop=True)
        floor['Length2'] = ((floor['X_end'] - floor['X_start'])**2 + (floor['Y_end'] - floor['Y_start'])**2)**0.5


        gap = floor['X_start'].min()
        Gaps.append(gap)
        z_value = 108
        floor[['X_start', 'X_end', 'X_insert']] = floor[['X_start', 'X_end', 'X_insert']] - gap
        floor[['Z_start', 'Z_end', 'Z_insert']] = floor[['Z_start', 'Z_end', 'Z_insert']] + (z_value * i)
        floor = floor[floor['Layer'] != 'Staircase_innerwall']
        if (floor['Layer'] == 'Staircase').any():
            staircase_layer = floor[floor['Layer'] == 'Staircase']
            floor = floor[floor['Layer'] != 'Staircase']
            stairs_line = staircase_layer[staircase_layer['Type'] == 'LINE']
            stairs_MTEXT = staircase_layer[staircase_layer['Type'] == 'MTEXT']
            stairs_line['Length'] = ((stairs_line['X_end'] - stairs_line['X_start'])**2 + (stairs_line['Y_end'] - stairs_line['Y_start'])**2)**0.5

            # Group lines by their length and filter groups with more than one line
            common_length_lines = stairs_line.groupby('Length').filter(lambda x: len(x) > 1)
            common_length_lines['Length'] = common_length_lines['Length'].round(2)
            length_counts = common_length_lines['Length'].value_counts()

            if not length_counts.empty:
                max_count_length = length_counts.idxmax()
                # Filter the DataFrame to keep only lines with the most common length
                max_length_lines = common_length_lines[common_length_lines['Length'] == max_count_length]

                # Save the rest of the lines
                rest_lines = stairs_line[~stairs_line.index.isin(max_length_lines.index)]
                rest_lines['Layer'] = rest_lines['Layer'].replace('Staircase','Staircase_wall2')
            else:
                max_count_length = None 
                max_length_lines = pd.DataFrame()  # Empty DataFrame
                rest_lines = stairs_line  # All lines are in rest_lines if no common length

            print('max_length_lines', max_length_lines)
            print('*************************************************')
            print('rest_lines', rest_lines)

            staires = pd.concat([max_length_lines, stairs_MTEXT])
            print('staires', staires)

            # Manage the staircase layer before duplicating lines
            staircase_layer = stairecase_manage(staires)
            print('staircase_layer', staircase_layer)

            floor = pd.concat([floor, staircase_layer])

            staircase_len = staircase_layer[staircase_layer['Layer'] == 'Staircase']['X_start'].count()

            diff = z_value / staircase_len
            sum_z = 0

            for j in floor.index.values:
                if floor.loc[j, "Layer"] == "Staircase":
                    sum_z += diff
                    floor.loc[j, "Z_start"] = floor.loc[j, "Z_start"] + sum_z
                    floor.loc[j, "Z_end"] = floor.loc[j, "Z_end"] + sum_z

            staircase_layer2 = floor[floor['Layer'] == 'Staircase']

            # Now, create duplicates of the staircase lines and connect them with the original lines
            connecting_lines = []

            duplicated_lines = staircase_layer2[staircase_layer2['Type'] == 'LINE'].copy()
            print('duplicated_lines:',duplicated_lines)
            print('staircase_layer2:',staircase_layer2)
            for _, orig_row in staircase_layer2[staircase_layer2['Type'] == 'LINE'].iterrows():
                dup_row = orig_row.copy()

                if abs(orig_row['X_start'] - orig_row['X_end']) <= 2:
                    # Vertical line, increment X values
                    dup_row['X_start'] += 12
                    dup_row['X_end'] += 12

                    # Add connection lines between original and duplicate
                    connecting_lines.append({
                        'X_start': orig_row['X_start'],
                        'Y_start': orig_row['Y_start'],
                        'Z_start': orig_row['Z_end'],
                        'X_end': dup_row['X_start'],
                        'Y_end': dup_row['Y_start'],
                        'Z_end': dup_row['Z_end'],
                        'Layer': 'Connected_staire_lines',
                        'Type': 'LINE'
                    })
                    connecting_lines.append({
                        'X_start': orig_row['X_end'],
                        'Y_start': orig_row['Y_end'],
                        'Z_start': orig_row['Z_end'],
                        'X_end': dup_row['X_end'],
                        'Y_end': dup_row['Y_end'],
                        'Z_end': dup_row['Z_end'],
                        'Layer': 'Connected_staire_lines',
                        'Type': 'LINE'
                    })

                elif abs(orig_row['Y_start'] - orig_row['Y_end']) <= 2:
                    # Horizontal line, increment Y values
                    dup_row['Y_start'] += 12
                    dup_row['Y_end'] += 12

                    # Add connection lines between original and duplicate
                    connecting_lines.append({
                        'X_start': orig_row['X_start'],
                        'Y_start': orig_row['Y_start'],
                        'Z_start': orig_row['Z_end'],
                        'X_end': dup_row['X_start'],
                        'Y_end': dup_row['Y_start'],
                        'Z_end': dup_row['Z_end'],
                        'Layer': 'Connected_staire_lines',
                        'Type': 'LINE'
                    })
                    connecting_lines.append({
                        'X_start': orig_row['X_end'],
                        'Y_start': orig_row['Y_end'],
                        'Z_start': orig_row['Z_end'],
                        'X_end': dup_row['X_end'],
                        'Y_end': dup_row['Y_end'],
                        'Z_end': dup_row['Z_end'],
                        'Layer':'Connected_staire_lines',
                        'Type': 'LINE'
                    })

                # Add the duplicated line to the list
                duplicated_lines = pd.concat([duplicated_lines, pd.DataFrame([dup_row])])

            # Convert the connecting lines list to a DataFrame and add to staircase_layer2
            connecting_lines_df = pd.DataFrame(connecting_lines)
            print('connecting_lines_df:',connecting_lines_df)
            staircase_layer2 = pd.concat([staircase_layer2, duplicated_lines, connecting_lines_df,rest_lines])
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('staircase_layer2:',staircase_layer2)
        
            # Add the updated staircase layer back to the floor DataFrame
            floor = pd.concat([floor, staircase_layer2])
            print('Floor_stairs:',floor[floor['Layer']=='Connected_staire_lines'])

        if i == df.index.unique()[-1]:
            floor = floor[floor["Layer"] != "Staircase"]
            floor = floor[floor["Layer"] != "Connected_staire_lines"]

        z_base = i * z_value  # Base height for this floor
        z_ceiling = z_base + 108  # Ceiling height for this floor
        plot_2d_boundary(fig, floor, gap, z_value, i,no_of_floors)
        if i > 0:
            floor = floor[floor['Layer'] != 'Boundary']
        for layer in floor['Layer'].unique():
            layer_data = floor[floor['Layer'] == layer]
            color = layer_colors.get(layer, '#808080')
            layer = str(layer)
            if 'Door' in layer or layer == 'Main_gate':
                for _, row in layer_data[layer_data['Type'] == 'LINE'].iterrows():
                    # Calculate gate/door dimensions and position
                    gate_width = np.sqrt((row['X_end'] - row['X_start'])**2 + (row['Y_end'] - row['Y_start'])**2)
                    gate_height = 60 if layer == 'Main_gate' else 60  # You can adjust the main gate height if needed
                    gate_thickness = 1  # Default thickness
                    total_height = 108  # Total height of the wall/plane

                    # Calculate gate/door position and rotation
                    gate_x = (row['X_start'] + row['X_end']) / 2
                    gate_y = (row['Y_start'] + row['Y_end']) / 2
                    gate_z = z_base

                    # Calculate rotation angle
                    angle = np.arctan2(row['Y_end'] - row['Y_start'], row['X_end'] - row['X_start'])

                    # Create and add the gate/door
                    if layer == 'Main_gate':
                        gate_fig = create_iron_main_gate(width=gate_width, height=gate_height, thickness=gate_thickness)
                        gate_traces = gate_fig.data  # Extract traces from the Figure object
                    else:
                        gate_traces = create_wooden_door(width=gate_width, height=gate_height, thickness=gate_thickness)

                    # Create a plane above the door (but not for the main gate)
                    if layer != 'Main_gate':
                        plane_height = total_height - gate_height
                        plane_trace = go.Mesh3d(
                            x=[0, gate_width, gate_width, 0, 0, gate_width, gate_width, 0],
                            y=[0, 0, 0, 0, gate_thickness, gate_thickness, gate_thickness, gate_thickness],
                            z=[gate_height, gate_height, gate_height + plane_height, gate_height + plane_height,
                               gate_height, gate_height, gate_height + plane_height, gate_height + plane_height],
                            i=[0, 0, 0, 1],
                            j=[1, 2, 3, 2],
                            k=[2, 3, 1, 3],
                            color='#C5C6C7',
                            opacity=1
                        )
                        gate_traces = list(gate_traces) + [plane_trace]

                    # Rotate and translate gate/door (and plane if it exists)
                    for trace in gate_traces:
                        # Convert trace coordinates to numpy arrays
                        x = np.array(trace.x)
                        y = np.array(trace.y)
                        z = np.array(trace.z)

                        # Rotate
                        rotated_x = np.cos(angle) * (x - gate_width/2) - np.sin(angle) * y
                        rotated_y = np.sin(angle) * (x - gate_width/2) + np.cos(angle) * y

                        # Translate
                        trace.x = rotated_x + gate_x
                        trace.y = rotated_y + gate_y
                        trace.z = z + gate_z

                        fig.add_trace(trace)

            else:
                for _, row in layer_data[layer_data['Type'] == 'LINE'].iterrows():
                    if layer == 'Staircase' or layer == 'Connected_staire_lines':
                        # For staircase, use the actual Z values from the data
                        add_plane(fig, row['X_start'], row['Y_start'], row['X_end'], row['Y_end'], 
                                  row['Z_start'], row['Z_end'], 'Black', layer)
                    elif layer == 'Staircase_wall2':
                        add_plane(fig, row['X_start'], row['Y_start'], row['X_end'], row['Y_end'], 
                                  z_base, z_base + 10, color, layer)
                    elif layer == 'Boundary' or layer == 'Main_gate':
                        add_plane(fig, row['X_start'], row['Y_start'], row['X_end'], row['Y_end'], 
                                  z_base, z_base + 60, color, layer)
                    elif layer == 'Balcony':
                        add_plane(fig, row['X_start'], row['Y_start'], row['X_end'], row['Y_end'], 
                                  z_base, z_base + 36, color, layer)
                    else:
                        # For other layers, use the floor's base and ceiling
                        add_plane(fig, row['X_start'], row['Y_start'], row['X_end'], row['Y_end'], 
                                  z_base, z_ceiling, '#C5C6C7', layer)

            # Add text labels (keep this part the same)
            for _, row in layer_data[layer_data['Type'] == 'MTEXT'].iterrows():
                fig.add_trace(go.Scatter3d(
                    x=[row['X_insert']],
                    y=[row['Y_insert']],
                    z=[z_value*i],  # Place text in middle of room height
                    mode='text',
                    text=[row['Text']],
                    textfont=dict(color='black', size=10),
                    hoverinfo='text',
                    hovertext=f'Layer: {layer}',
                    showlegend=False
                ))

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis=dict(
                visible=True,
                range=[0, 250],
                title='Height(inch)',
                titlefont=dict(size=12),
                dtick=108
            )
        ),
        title=f'3D Plot for All Floors',
        autosize=True
    )


    # Show the plot
    # fig.show()
    file_name = f'{project_name}_{number}.html'
    file_path = os.path.join(settings.MEDIA_ROOT, 'gifs', file_name)

    # Ensure the directory exists

    # Save the file
    fig.write_html(file_path)
    # If you want to save as HTML for interactivity



#3D

# ########################################################


base_dir = settings.BASE_DIR / 'assets'
full_path = base_dir / 'MetaData.csv'
print(f"Attempting to read file from: {full_path}")
print(f"File exists: {full_path.exists()}")

nearest_neighbors, Sorted_points = Similarity_fuc_main(new_point, str(full_path))
Sorted_points = Sorted_points.to_list()
print(Sorted_points)

for file in Sorted_points:
    
    # Adjust the original DXF file coordinates to the origin (0,0)
    print(file)
    adjust_dxf_coordinates_to00(os.path.join(data_folder, file))

    # Convert the modified DXF file to a DataFrame for further processing
    output_filename = os.path.splitext(os.path.basename(file))[0] + project_name+"_new.dxf"
    # output_path = os.path.join(settings.BASE_DIR, 'Temp', 'dxfCache', output_filename)
    testing1 = Dxf_to_DF(os.path.join(settings.BASE_DIR, 'Temp', 'dxfCache', output_filename))

        # Adjust the start and end coordinates within the DataFrame
    testing2 = adjust_Xstart_ystart(testing1)
    testing3 = floor_main(testing2)
    testing4 = testing3.set_index('floor').sort_index()

    testing5 = pd.DataFrame(columns = ['Type', 'Layer', 'X_start', 'Y_start', 'Z_start', 'X_end', 'Y_end',
           'Z_end', 'Length', 'Text', 'X_insert', 'Y_insert', 'Z_insert'])
    ine3 = pd.DataFrame(columns = ['Type', 'Layer', 'X_start', 'Y_start', 'Z_start', 'X_end', 'Y_end',
           'Z_end', 'Length', 'Text', 'X_insert', 'Y_insert', 'Z_insert'])
    Gaps = []
    for i in testing4.index.unique():
#         print(i)
        floor = testing4.loc[i]
        #print(floor['X_start'].min())
        gap = floor['X_start'].min()
        Gaps.append(gap)
        #print(gap)
        floor[['X_start', 'X_end','X_insert']] = floor[['X_start', 'X_end','X_insert']] - gap
        testing5 = pd.concat([testing5,floor],axis = 0)

    #Calculate the difference in dimensions requested by the user
    info = {}
    for j in testing4.index.unique():
#         print(100*'*')
#         print(j)
        testing_floor = testing4.loc[j]
#         print(testing_floor)
        
        # Calculate the difference in dimensions requested by the user
        original_x = testing_floor['X_start'].max() - testing_floor['X_start'].min()
#         print("original_x:" ,original_x)
        original_y = testing_floor['Y_start'].max() - testing_floor['Y_start'].min()
#         print("original_y:" ,original_y)
        diff_Y = original_y - (User_LB[1]*12)
#         print("diff_Y:" ,diff_Y)
        diff_X = original_x - (User_LB[0]*12)
#         print(User_LB[0])
#         print("diff_X:" ,diff_X)

        # Add horizontal lines based on computed diff_X
        step1 = add_horizontal_lines_for_X_updated(testing5)
#         print(step1)

        # Group lines and distribute units proportionally based on diff_X
        step2 = find_line_groups_for_X(step1)
#         print(step2)
        step3 = distribute_units_proportionally_for_X(step2, diff_X)
#         print(step3)
        selected_pairs_for_X = distribute_units_between_lines_for_X(step3)
#         print(selected_pairs_for_X)
        for k in range(len(selected_pairs_for_X)):
#             print(Gaps[j])
            selected_pairs_for_X[k] = (selected_pairs_for_X[k][0]+Gaps[j],selected_pairs_for_X[k][1]+Gaps[j])
#         print(selected_pairs_for_X)

        # Multiple trimming operations along X coordinates
        step4 = Multiple_trim_for_X(testing_floor)

        # Similar operations are repeated for Y coordinates
        step5 = add_horizontal_lines_for_Y_updated(testing5)
#         print(step5)
        step6 = find_line_groups_for_Y(step5)
#         print(step6)
        step7 = distribute_units_proportionally_for_Y(step6, diff_Y)
#         print(step7)
        selected_pairs_for_Y = distribute_units_between_lines_for_Y(step7)

        # Multiple trimming operations along Y coordinates
        step8 = Multiple_trim_for_Y(step4)
        inmage_name = project_name+file+'_{}'.format(j)+'.png'
        plot_dataframe(step8,inmage_name)
        boq = area_main(step8)

        floor_info = {inmage_name:boq}
        info.update(floor_info)
        # step8[['X_start','X_end','X_insert']] = step8[['X_start','X_end','X_insert']] + (Gaps[j])/12 
        ine3 = pd.concat([ine3,step8])

    

    digit = ''.join([char for char in file if char.isdigit()])
    final_filename = project_name+'_{}'.format(digit)+'.dxf'
    create_dxf_from_dataframe(ine3, final_filename)
    
    trimmed_dxf_path=os.path.join(settings.MEDIA_ROOT , 'dxfs')
    plot_dxf(os.path.join(trimmed_dxf_path,final_filename))
    final_json = {final_filename.replace('.dxf','.png'):info}
    contrains_df = constrains(os.path.join(trimmed_dxf_path,final_filename))
    area_true_percentage, area_false_percentage = calculate_percentage(contrains_df, 'Condition_Area')
    length_true_percentage, length_false_percentage = calculate_percentage(contrains_df, 'Condition_Length')
    width_true_percentage, width_false_percentage = calculate_percentage(contrains_df, 'Condition_Width')
    avg = (area_true_percentage+length_true_percentage+width_true_percentage)/3
    avg = round(avg,2)
    print("INFO:",final_json)
    final_3d(ine3,digit)
    # print(final_filename , ':' , avg)
    # print(f"AVG:{avg}")
    # boqs = area_main(ine3)
    # print(boqs)

















