# Project: TRITON-Lite, A Deep Learning Surrogate Model for Flood Inundation Modeling

# Water Depth Data Processing, which is the output of triton lite

import rasterio
from shapely.geometry import box
import geopandas as gpd
import configparser
import os

# Reading the configuration file
config = configparser.ConfigParser()
config.read('./directories.cfg')

# Accessing the directories and file paths
# example path: /lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl
dem_path = config.get('Directories', 'DEMPath')
large_grid_shapefile_path = config.get('Directories', 'LargeGridShapefilePath')

# Size of the larger polygon in terms of number of cells (e.g., 80x80)
cell_group_size = 80

# Open the raster file using rasterio
with rasterio.open(dem_path) as dataset:
    # Get the dimensions of a single raster cell
    cell_width, cell_height = dataset.res

    # Get the bounds of the raster
    left, bottom, right, top = dataset.bounds

    # Generate the large grid cells (boxes) for groups of pixels
    large_grid_cells = []
    for i in range(0, dataset.width, cell_group_size):
        for j in range(0, dataset.height, cell_group_size):
            # Calculate the bounds of the large polygon
            minx = left + i * cell_width
            maxx = minx + cell_group_size * cell_width
            maxy = top - j * cell_height
            miny = maxy - cell_group_size * cell_height

            # Ensure the large polygon does not extend beyond the raster bounds
            if maxx > right:
                maxx = right
            if miny < bottom:
                miny = bottom

            # Create the large polygon and add it to the list
            large_grid_cells.append(box(minx, miny, maxx, maxy))

    # Create a GeoDataFrame with the large grid cells
    large_grid_gdf = gpd.GeoDataFrame(geometry=large_grid_cells)

# Save to a new shapefile
large_grid_gdf.to_file(large_grid_shapefile_path)

print("The large grid shapefile has been created and saved.")