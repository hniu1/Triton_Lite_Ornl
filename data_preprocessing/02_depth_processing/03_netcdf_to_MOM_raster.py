import os
import rasterio
import numpy as np
import configparser
import rasterio
import xarray as xr
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

config_path = r'./directories.cfg'

config = configparser.ConfigParser()
config.read(config_path)

input_directory  = config.get('Directories', 'NetCDFInputDir')
output_directory = config.get('Directories', 'MOMRasterOutputDir')

variable_of_interest = "output_depth"

os.makedirs(output_directory, exist_ok=True)


# -------------------------------
# Find NetCDF files (sorted for reproducibility)
# -------------------------------
netcdf_file_names = sorted([f for f in os.listdir(input_directory) if f.endswith(".nc")])
if not netcdf_file_names:
    raise FileNotFoundError(f"No .nc files found in {input_directory}")

# -------------------------------
# Use rasterio/GDAL to grab georeferencing metadata (same source as code #1)
# -------------------------------
first_nc_path = os.path.join(input_directory, netcdf_file_names[0])
with rasterio.open(f'NETCDF:"{first_nc_path}":{variable_of_interest}') as src:
    out_meta = src.meta.copy()

overall_max = None

# -------------------------------
# Loop through NetCDF files (xarray for data, but matching GDAL-style behavior)
# -------------------------------
for nc_name in netcdf_file_names:
    nc_path = os.path.join(input_directory, nc_name)
    print(f"\nProcessing: {nc_path}")

    # Key part to match GDAL/rasterio more closely:
    # - mask_and_scale=False avoids turning _FillValue into NaN and avoids applying scale/offset
    # - decode_times=False avoids time decoding overhead (not needed here)
    ds = xr.open_dataset(nc_path, engine="netcdf4", mask_and_scale=False, decode_times=False)

    try:
        var = ds[variable_of_interest]

        # Mimic "bands" by iterating over the first dimension (usually time)
        time_dim = var.dims[0]
        nt = var.sizes[time_dim]
        print(f"Variable dims: {var.dims}, shape: {tuple(var.shape)} (iterating {time_dim}={nt})")

        file_max = None

        for t in range(nt):
            print(f"  reading slice {t+1}/{nt}")

            # Force float32 so it matches your intended output dtype in code #1
            slice_np = np.asarray(var.isel({time_dim: t}).values, dtype=np.float32)

            if file_max is None:
                file_max = slice_np.copy()
            else:
                np.maximum(file_max, slice_np, out=file_max)

            del slice_np

        if overall_max is None:
            overall_max = file_max.copy()
        else:
            np.maximum(overall_max, file_max, out=overall_max)

        del file_max

    finally:
        ds.close()

# # -------------------------------
# # Save final raster (same as code #1’s intent)
# # -------------------------------
# os.makedirs(output_directory, exist_ok=True)
final_output_raster_path = os.path.join(output_directory, "Final_MOM_ACC_baseline.tif")

out_meta.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    height=overall_max.shape[0],
    width=overall_max.shape[1],
)

with rasterio.open(final_output_raster_path, "w", **out_meta) as dst:
    dst.write(overall_max, 1)

print(f"\nWrote: {final_output_raster_path}")

# Reset for the next part
input_raster_path = final_output_raster_path
output_raster_path = os.path.join(output_directory, 'Final_MOM_ACC_baseline_zero.tif')

# Process the existing raster
with rasterio.open(input_raster_path) as src:
    meta = src.meta
    no_data_value = meta.get("nodata", -9999)
    if no_data_value is None:
        no_data_value = -9999

    meta.update(dtype=rasterio.int32, nodata=int(no_data_value))
    # no_data_value = meta.get('nodata', -9999)
    # meta.update(dtype=rasterio.int32, nodata=int(no_data_value))
    
    for i in range(1, src.count + 1):
        band = src.read(i).astype(rasterio.int32)
        mask = band != no_data_value
        band[mask] = 0
        
        if i == 1:
            with rasterio.open(output_raster_path, 'w', **meta) as dst:
                dst.write(band, i)
        else:
            with rasterio.open(output_raster_path, 'r+', **meta) as dst:
                dst.write(band, i)


def raster_to_polygon_and_dissolve(config_path):
    # Reading the configuration file
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Extracting paths from the configuration file
    raster_path = config.get('Directories', 'MOM_Raster_Path')
    output_shapefile = config.get('Directories', 'OutputShapefilePath')
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the raster data
        image = src.read(1)  # Read the first band
        
        # Adjust the mask to ignore no data values (-9999)
        mask = (image != 0) & (image != -9999)
        
        # Extract shapes and values from the raster
        results = list(shapes(image, mask=mask, transform=src.transform))
        
        # Convert the results to a GeoDataFrame
        geoms = [shape(geom) for geom, value in results if value != -9999]
        gdf = gpd.GeoDataFrame({'geometry': geoms})
        
        # Set the coordinate reference system (CRS) from the raster to the GeoDataFrame
        gdf.crs = src.crs

        # Dissolve all polygons into a single polygon
        dissolved_gdf = gdf.dissolve()
        
        # Save the dissolved GeoDataFrame to a shapefile
        dissolved_gdf.to_file(output_shapefile, driver='ESRI Shapefile')

    print(f"Conversion and dissolve completed. Output saved to {output_shapefile}")

# Example usage
raster_to_polygon_and_dissolve(config_path)
