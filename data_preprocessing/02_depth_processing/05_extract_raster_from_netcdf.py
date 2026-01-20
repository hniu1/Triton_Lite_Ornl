import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os
import configparser
from pathlib import Path
from shapely.geometry import box


script_path = Path(__file__).parent.resolve()
os.chdir(script_path)

# Base directory and file format
watershed_name = 'Sugar Creek'  # Example watershed name
watershed_name_lower = watershed_name.replace(" ", "_").lower()

# Reading the configuration file
config_path = 'directories.cfg'

config = configparser.ConfigParser()
config.read(config_path)  # Make sure this path is correct

base_dir = config.get('Directories', 'NetCDFInputDir')
shapefile_path = config.get('Directories', 'shapefile_path')
output_dir = config.get('Directories', 'water_depth_rasters')

output_dir = f'{output_dir}_{watershed_name_lower}'
os.makedirs(f'{output_dir}', exist_ok=True)

# Load the shapefile using GeoPandas
shapes_all = gpd.read_file(shapefile_path)

# Filter the shapefile for the specific watershed
shapes = shapes_all[shapes_all['Name'] == watershed_name]

# Loop through the files
for i in range(1, 41):
    try:
        # Construct file names
        netcdf_file_name = f'D{i:03d}_ACC_future.nc'
        netcdf_file_path = os.path.join(base_dir, netcdf_file_name)
        output_tiff_name = f'ACC_D{i:03d}_{watershed_name_lower}.tif'
        output_tiff_path = os.path.join(f'{output_dir}', output_tiff_name)

        # # if file exists, skip
        # if os.path.exists(output_tiff_path):
        #     print(f"File {output_tiff_path} already exists. Skipping...")
        #     continue

        # Open the NetCDF file
        with rasterio.open(netcdf_file_path) as src:
            # Clip the raster with the shapefile
            out_image, out_transform = mask(src, shapes.geometry, crop=True)

            # Copy the metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })

            # Save the clipped raster to a TIFF file
            with rasterio.open(output_tiff_path, "w", **out_meta) as dest:
                dest.write(out_image)

        print(f"Clipped raster saved to {output_tiff_path}")
    
    except Exception as e:
        print(f"Error processing file {netcdf_file_name}: {e}")
        continue
