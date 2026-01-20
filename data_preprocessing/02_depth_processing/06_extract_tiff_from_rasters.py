import math
import numpy as np
import pandas as pd
import rasterio
import os
import configparser
from pathlib import Path

script_path = Path(__file__).parent.resolve()
os.chdir(script_path)

watershed_name = 'Sugar Creek'
watershed_name_lower = watershed_name.replace(" ", "_").lower()

config_path = r'./directories.cfg'

config = configparser.ConfigParser()
config.read(config_path)  # Make sure this path is correct

base_dir = config.get('Directories', 'NetCDFInputDir')

# Base path to your TIFF and CSV files
full_ras_path = config.get('Directories', 'water_depth_rasters')+f'_{watershed_name_lower}'

csv_file = config.get('Directories', 'csv_file_path')
out_folder_chunk = config.get('Directories', 'out_folder_tiff')+f'_{watershed_name_lower}'

# Ensure the output directory exists
os.makedirs(out_folder_chunk, exist_ok=True)

# Load CSV file for coordinates
coords_all = pd.read_csv(csv_file)
coords_df = coords_all[coords_all['Name'] == watershed_name].reset_index(drop=True)

# count total number of rasters to process under full_ras_path where it starts with 'ACC_D' and ends with '{lower}.tif'
total_rasters = len([f for f in os.listdir(full_ras_path) if f.startswith('ACC_D') and f.endswith(f'{watershed_name_lower}.tif')])

# Dynamically generate filenames of the TIFFs to process
filenames = [f'ACC_D{str(i).zfill(3)}_{watershed_name_lower}.tif' for i in range(1,total_rasters+1)]

# Process each TIFF file
for filename in filenames:
    full_ras_name = os.path.join(full_ras_path, filename)
    
    # Check if the TIFF file exists
    if not os.path.exists(full_ras_name):
        print(f"File {filename} does not exist. Skipping...")
        continue

    print(f"Processing {filename}...")
    
    # Process each row in the DataFrame for the current TIFF
    for index, row in coords_df.iterrows():
        xll = row['X']
        yll = row['Y']
        
        with rasterio.open(full_ras_name) as src:
            # Determine the position of the chunk in the raster grid
            transform = src.transform
            curr_col, curr_row = ~transform * (xll, yll)
            curr_col, curr_row = map(math.ceil, [curr_col, curr_row])
            
            out_ht = 80
            out_wd = 80
            final_col = curr_col + out_wd
            initial_row = curr_row - out_ht

            # Generate output filename
            outfname = f'{watershed_name_lower}_{filename[:-4]}_block_{index}.tif'
            out_file = os.path.join(out_folder_chunk, outfname)

            if curr_col >= 0 and final_col <= src.width and initial_row >= 0 and (initial_row + out_ht) <= src.height:
                window = rasterio.windows.Window(col_off=curr_col, row_off=initial_row, width=out_wd, height=out_ht)
                data = src.read(window=window)
                if src.nodata is not None:
                    data = np.where(data == src.nodata, 0, data)
                
                with rasterio.open(
                    out_file,
                    'w',
                    driver='GTiff',
                    height=out_ht,
                    width=out_wd,
                    count=src.count,
                    dtype=data.dtype,
                    crs=src.crs,
                    transform=rasterio.windows.transform(window, src.transform)) as dst:
                    dst.write(data)
                print(f'Raster data extracted and saved for {filename}, index {index}, to {outfname}')
            else:
                print(f"Window for {filename}, index {index} is out of raster bounds. Creating empty TIFF file...")
                empty_data = np.full((src.count, out_ht, out_wd), src.nodata, dtype=src.read(1).dtype)
                with rasterio.open(
                    out_file,
                    'w',
                    driver='GTiff',
                    height=out_ht,
                    width=out_wd,
                    count=src.count,
                    dtype=src.read(1).dtype,
                    crs=src.crs,
                    transform=src.transform) as dst:
                    dst.write(empty_data)
                print(f'Empty TIFF file created for {filename}, index {index}, to {outfname}')


