'''
This code creates a new shapefile that selects the blocks that covers the maximum flood inundated area, then creates a csv file of the xll and yll of those selected blocks with Watershed's name and ID.
'''

import pandas as pd
import geopandas as gpd
import configparser
import os
from shapely.geometry import Point

# Reading the configuration file
config_path = r'./directories.cfg'

config = configparser.ConfigParser()
config.read(config_path)  # Make sure this path is correct

# Accessing the directories and file paths
shapefile_dir = config.get('Directories', 'ShapefileDir')
output_dir = config.get('Directories', 'OutputDir')
shapefile_path = config.get('Directories', 'shapefile_path')
output_csv_path = config.get('Directories', 'output_csv_path')

# Using the directories to load shapefiles
river_polygons = gpd.read_file(os.path.join(shapefile_dir, "MOM_raster_dissolved.shp"))
grid_polygons = gpd.read_file(os.path.join(shapefile_dir, "blocks_conasauga.shp"))

# Perform a spatial join to find intersects
intersects = gpd.sjoin(grid_polygons, river_polygons, how="inner", predicate="intersects")

# Calculate lower left coordinates directly from intersected polygons
def lower_left_coordinates(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    return Point(minx, miny)

# Apply the function and create a GeoDataFrame from the results
coordinates_gdf = gpd.GeoDataFrame(geometry=intersects['geometry'].apply(lower_left_coordinates))

# Load the watershed shapefile to find which polygon each point falls into
watershed_polygons = gpd.read_file(shapefile_path)

# Initialize a list to store the results
results = []

# Iterate over each point to determine which polygon it falls into
for _, point in coordinates_gdf.iterrows():
    point_found = False
    for _, polygon in watershed_polygons.iterrows():
        if point.geometry.within(polygon.geometry):
            # Add the polygon's name and the unique ID to the results list
            results.append({
                'X': point.geometry.x, 
                'Y': point.geometry.y, 
                'Name': polygon['Name']
            })
            point_found = True
            break
    if not point_found:
        # If the point doesn't fall within any polygon
        results.append({'X': point.geometry.x, 'Y': point.geometry.y, 'Name': 'Outside'})

# Convert the results to a DataFrame
df_results = pd.DataFrame(results)

# Generate a unique ID for each unique name
unique_names = df_results['Name'].unique()
name_to_id = {name: idx for idx, name in enumerate(unique_names, start=1)}

# Map the names to IDs
df_results['ID'] = df_results['Name'].apply(lambda name: name_to_id[name])

# Save the results to a new CSV file
df_results.to_csv(output_csv_path, index=False)

print("Processing complete. Output saved to:", output_csv_path)
