import logging
import folium
from geopy.distance import geodesic
import googlemaps
from datetime import datetime
import os, numpy as np,pandas 
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from django.conf import settings
from pathlib import Path
from PIL import Image, ImageSequence
from geopy.distance import geodesic
from helper.uuidGenerator import generate_short_uuid
import math
from folium.raster_layers import ImageOverlay
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Replace with your Google Maps API key
GOOGLE_MAPS_API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY')
base_dir = settings.BASE_DIR / 'assets'
excel_path = base_dir / 'soil_type.xlsx'
img_path = base_dir / 'sitemapanaloverlay.png'
homeIcon=base_dir/'home_icon.png'
# Initialize the Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_user_input():
    while True:
        try:
            latitude = float(input("Enter latitude: "))
            longitude = float(input("Enter longitude: "))
            if -90 <= latitude <= 90 and -180 <= longitude <= 180:
                return latitude, longitude
            else:
                print("Invalid latitude or longitude. Please enter valid values.")
        except ValueError:
            print("Invalid input. Please enter numerical values for latitude and longitude.")

def calculate_bearing(pointA, pointB):
    """
    Calculate the bearing between two points.
    """
    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])
    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def rotate_gif(input_path, output_path, rotation_angle):
    """
    Rotate the GIF by the specified angle and save it as a new file.
    """
    try:
        with Image.open(input_path) as img:
            frames = [frame.rotate(rotation_angle, expand=True) for frame in ImageSequence.Iterator(img)]
            frames[0].save(
                output_path, save_all=True, append_images=frames[1:], loop=img.info.get('loop', 0),
                duration=img.info.get('duration', 100), disposal=img.info.get('disposal', 2)
            )
    except Exception as e:
        logging.error(f"Error rotating GIF: {e}")


def create_map_with_gif_overlay(latitude, longitude, gif_path, zoom=20, min_zoom=16):
    """
    Create a Folium map and overlay a GIF at specified coordinates.
    """
    # Create the base map
    folium_map = folium.Map(location=[latitude, longitude], zoom_start=zoom, min_zoom=min_zoom, max_zoom=22,tiles='CartoDB positron', mapattr='CartoDB positron')

    # Define the bounds where the GIF will be overlaid
    lat_offset = 0.0005  # Adjust these offsets based on the desired size and positioning
    lon_offset = 0.0005
    south = latitude - lat_offset
    north = latitude + lat_offset
    west = longitude - lon_offset
    east = longitude + lon_offset

    # Create the ImageOverlay using the rotated GIF path
    image_overlay = ImageOverlay(
        image=gif_path,
        bounds=[[south, west], [north, east]],
        interactive=True,
        cross_origin=False,
        zindex=1,
        
    )

    # Add overlay to the map
    image_overlay.add_to(folium_map)

    return folium_map
def create_map(latitude, longitude, zoom=22, min_zoom=16):
    folium_map = folium.Map(
        location=[latitude, longitude],
        zoom_start=zoom,
        min_zoom=min_zoom,
        max_zoom=22,  # Increase max zoom level
    )
    return folium_map

def add_marker(map_obj, latitude, longitude, color, popup, icon_path=None,place_type=None):
    icons={"park":"tree-conifer",
           "shopping_mall":"shopping-cart",
           "bank":"usd",
           "school":"book",
           "hotel":"briefcase"
           }
    if icon_path:
        icon = folium.CustomIcon(icon_path, icon_size=(30, 30))
    else:
        icon = folium.Icon(color=color, icon=icons.get(place_type, 'info-sign'),prefix='glyphicon')

    folium.Marker(
        [latitude, longitude], 
        icon=icon,
        popup=popup
    ).add_to(map_obj)

def find_nearby_places(map_obj, latitude, longitude, place_types, radius):
    # Add marker for user's location with home icon
    # add_marker(map_obj, latitude, longitude, 'red', 'Your Location', icon_path=str(homeIcon))
    
    colors = {
        "park": "green",
        "shopping_mall": "blue",
        "bank": "purple",
        "hotel": "orange",
        "school": "gray"
    }
    
    for place_type in place_types:
        print(f"Searching for nearby {place_type}s within {radius} meters...")
        
        # Use Google Maps Places API to search for nearby places
        places_result = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=radius,
            type=place_type
        )
        
        if places_result['status'] == 'OK':
            for place in places_result['results']:
                place_lat = place['geometry']['location']['lat']
                place_lon = place['geometry']['location']['lng']
                place_name = place['name']
                place_address = place.get('vicinity', 'Address not available')
                
                place_distance = geodesic((latitude, longitude), (place_lat, place_lon)).meters
                
                popup_text = f"<b><span style='color:red;'>NAME:</span></b> {place_name}<br>" \
                             f"<b><span style='color:red;'>CATEGORY:</span></b> {place_type}<br>" \
                             f"<b><span style='color:red;'>DISTANCE:</span></b> {place_distance:.2f} meters"
                add_marker(map_obj, place_lat, place_lon, colors.get(place_type, 'blue'), popup_text, place_type=place_type)
            
            print(f"Found {len(places_result['results'])} {place_type}(s)")
        else:
            print(f"No nearby {place_type}s found or error in API response.")
    
    # Find nearby roads (using reverse geocoding)
    print("Searching for nearby roads...")
    reverse_geocode_result = gmaps.reverse_geocode((latitude, longitude))
    if reverse_geocode_result:
        for result in reverse_geocode_result:
            if 'route' in result['types']:
                road_name = result['address_components'][0]['long_name']
                road_lat = result['geometry']['location']['lat']
                road_lon = result['geometry']['location']['lng']
                road_distance = geodesic((latitude, longitude), (road_lat, road_lon)).meters
                add_marker(map_obj, road_lat, road_lon, 'gray', f"{road_name} ({road_distance:.2f} meters)")
                break  # Just add the nearest road
    else:
        print("No nearby roads found.")
    
    return map_obj

def load_image(image_path):
    try:
        from matplotlib import pyplot as plt
        import matplotlib.image as mpimg
        img = mpimg.imread(image_path)
        return img
    except FileNotFoundError:
        logging.error(f"Error: Image file '{image_path}' not found.")
        return None
    except Exception as e:
        logging.error(f"Error loading image '{image_path}': {e}")
        return None

def add_image_overlay(map_obj, img, latitude, longitude, size_factor=0.0005):
    lat_offset = size_factor
    lon_offset = size_factor
    south = latitude - lat_offset
    north = latitude + lat_offset
    west = longitude - lon_offset
    east = longitude + lon_offset

    image_overlay = folium.raster_layers.ImageOverlay(
        image=img,
        bounds=[[south, west], [north, east]],
        interactive=True,
        cross_origin=False,
        zindex=1
    )
    image_overlay.add_to(map_obj)
    
    return image_overlay

def add_compass_markers(map_obj, latitude, longitude, size_factor=0.0005):
    distance = size_factor * 2  # Adjust distance based on image size factor
    compass_points = {
        'N': [latitude + distance, longitude],
        'S': [latitude - distance, longitude],
        'E': [latitude, longitude + distance],
        'W': [latitude, longitude - distance]
    }
    for direction, coords in compass_points.items():
        folium.Marker(
            coords,
            tooltip=direction,
            icon=folium.DivIcon(html=f'<div style="color: red; font-size: 24px; font-weight: bold;">{direction}</div>')
        ).add_to(map_obj)

def add_zoom_handler(folium_map, latitude, longitude, size_factor):
    # JavaScript code to dynamically update the image overlay bounds on zoom
    script = f"""
    <script>
        function updateImageOverlay() {{
            var map = document.querySelector('.leaflet-container')._leaflet_map;
            var zoom = map.getZoom();
            var sizeFactor = {size_factor} * Math.pow(2, 21 - zoom);
            var south = {latitude} - sizeFactor;
            var north = {latitude} + sizeFactor;
            var west = {longitude} - sizeFactor;
            var east = {longitude} + sizeFactor;
            imageOverlay.setBounds([[south, west], [north, east]]);
        }}
        
        var map = document.querySelector('.leaflet-container')._leaflet_map;
        var imageOverlay;
        map.eachLayer(function (layer) {{
            if (layer instanceof L.ImageOverlay) {{
                imageOverlay = layer;
            }}
        }});
        map.on('zoomend', updateImageOverlay);
        updateImageOverlay();
    </script>
    """
    folium_map.get_root().html.add_child(folium.Element(script))

def overlay_image_on_map(map_obj, latitude, longitude, image_path, size_factor=0.005):
    img = load_image(image_path)
    if img is None:
        print("Unable to load image. Exiting.")
        return None

    image_overlay = add_image_overlay(map_obj, img, latitude, longitude, size_factor=size_factor)
    add_compass_markers(map_obj, latitude, longitude, size_factor=size_factor)
    add_zoom_handler(map_obj, latitude, longitude, size_factor)

    return map_obj
def determine_rotation(front_of_house):
    """
    Determine rotation angle based on the front of the house.
    """
    direction_map = {
        'n': 0, 'nne': 22.5, 'ne': 45, 'ene': 67.5,
        'e': 90, 'ese': 112.5, 'se': 135, 'sse': 157.5,
        's': 180, 'ssw': 202.5, 'sw': 225, 'wsw': 247.5,
        'w': 270, 'wnw': 292.5, 'nw': 315, 'nnw': 337.5
    }

    front_of_house = front_of_house.lower()
    if front_of_house in direction_map:
        return direction_map[front_of_house]
    else:
        raise ValueError(f"Unknown direction '{front_of_house}'")
    
def add_boundary(map_obj, latitude, longitude, boundary_coords, fill_color='red', fill_opacity=0.02):
    """
    Add a boundary around the specified point with a defined offset.
    """

    folium.Polygon(
        locations=boundary_coords,
        color='red',
        weight=0.5,
        fill=True,
        fill_color=fill_color,
        fill_opacity=fill_opacity,
        opacity=0.8
    ).add_to(map_obj)

def main(output_file, front_of_house, latitude, longitude, boundary_coords, gif_path):
    # Determine rotation based on the front of the house
    rotation_needed = determine_rotation(front_of_house)
    logging.info(f"Rotation needed: {rotation_needed} degrees")

    # Rotate the GIF and save it as a temporary file
    shortID=generate_short_uuid()
    rotated_gif_path = str(settings.BASE_DIR/'Temp'/'mapgifCache'/f'rotated_direction_{shortID}.gif')
    rotate_gif(gif_path, rotated_gif_path, rotation_needed)

    # Create the map with the rotated GIF overlay
    folium_map = create_map_with_gif_overlay(latitude, longitude, rotated_gif_path)

    # Add a boundary around the house
    add_boundary(folium_map, latitude, longitude, boundary_coords, fill_color='red', fill_opacity=0.2)

    # Find nearby places
    place_types = ["park", "shopping_mall", "bank", "hotel", "school"]
    find_nearby_places(folium_map, latitude, longitude, place_types, radius=2000)

    media_maps_dir = os.path.join(settings.MEDIA_ROOT, 'maps')
    
    # Save the map file within media/maps directory
    map_file_path = os.path.join(media_maps_dir, output_file)
    folium_map.save(map_file_path)
    
    logging.info(f"Map saved as '{map_file_path}'")
    os.remove(rotated_gif_path)
    # Return the relative path to be stored in the database
    return os.path.relpath(map_file_path, settings.MEDIA_ROOT)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    
    return distance

def soil_type(df, latitude, longitude):
    inp_lat = latitude
    inp_long = longitude

    dist = []

    for index, rows in df.iterrows():
        soil_lat = rows['Latitude']
        soil_long = rows['Longitude']

        val = haversine(inp_lat, inp_long, soil_lat, soil_long)
        dist.append(val)

    df['Distance'] = dist
    closest = df.nsmallest(1, 'Distance')
    close = closest[['Soil Type', 'Ground Water Depth', 'Foundation Type']].copy()
    return close


__all__ = ['main', 'soil_type']