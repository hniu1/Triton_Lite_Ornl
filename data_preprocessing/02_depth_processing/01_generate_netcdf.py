#!/usr/bin/env python3
"""
Water Depth Data Processing (Triton Lite Output → NetCDF)
Author: ORNL Triton Team
"""

import os
import glob
import time
import zipfile
import subprocess
import numpy as np
import netCDF4 as nc4
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def unzip_selected(zip_path, output_path, output_type):
    """Extract only files matching e.g., H*.dat from a ZIP archive."""
    os.makedirs(output_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        selected = [f for f in zf.namelist() if f.split("/")[-1].startswith(output_type) and f.endswith(".dat")]
        for f in selected:
            zf.extract(f, output_path)
    print(f"Extracted {len(selected)} {output_type}*.dat files to {output_path}")
    return selected


def get_output_files(dat_path, output_type):
    """Return sorted list of binary output files."""
    files = sorted(glob.glob(os.path.join(dat_path, f"D00*/output/flood2d/bin/{output_type}*")), key=len)
    print(f"Found {len(files)} output files.")
    return files


def get_dir_size_human(path):
    """Return directory size in human-readable format."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024:
            return f"{total:.2f}{unit}"
        total /= 1024


def bin2array(bin_file, rows, cols, threshold=0, pad=1):
    """Convert a Triton Lite binary depth file into a 2D numpy array."""
    array = np.fromfile(bin_file)
    # array = np.reshape(array[2:], (int(array[0]), int(array[1]))) # This is for the current triton output format
    array = array.reshape((rows+2*pad, cols+2*pad))  # this is for the Conassauga output format
    array = array[pad:rows+pad, pad:cols+pad]
    array = np.where(array > threshold, array, np.nan)  # Take out no-flood values
    
    return array


def create_time_variable(nc, name, n_steps, start_time, interval_sec, bound_width_hr, ref_time):
    """Generic helper to create a CF-compliant time variable with bounds."""
    nc.createDimension(name)
    time_var = nc.createVariable(name, "f4", (name,))
    time_var.standard_name = "time"
    time_var.long_name = "time"
    time_var.calendar = "standard"
    time_var.units = ref_time
    time_var.bounds = f"{name}_bnds"

    # nc.createDimension("nv", 2)
    time_bnds = nc.createVariable(f"{name}_bnds", "f4", (name, "nv"))
    time_bnds.units = ref_time

    for i in range(n_steps):
        # input time in_dates = [time_origin + timedelta(seconds=(i)*10800) for i in range(80)]
        # output time out_dates = [time_origin + timedelta(seconds=(i+1)*1800) for i in range(480)]
        date = start_time + timedelta(seconds=(i + (1 if "out" in name else 0)) * interval_sec)
        delta = (date - start_time)
        hrs = delta.days * 24 + delta.seconds / 3600
        time_var[i] = hrs
        time_bnds[i] = [hrs - bound_width_hr, hrs + bound_width_hr]
        # in_time_bnds[i] = [(hours_since_origin)-0.5, (hours_since_origin) + 0.5] # Assign time bounds.
        # out_time_bnds[i] = [(hours_since_origin)-0.25, (hours_since_origin) + 0.25] # Assign time bounds.
    return nc


# ---------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------

def process_depth_data(zip_path, base_out_dir, output_type, nc_name,
                       cols, rows, gt, ref_time, time_origin):
    start_time = time.time()
    dat_path = os.path.join(base_out_dir, "temp_bin")
    nc_path = os.path.join(base_out_dir, 'nc')

    print(f"\n[1/5] Extracting {output_type}*.dat from {zip_path}")
    unzip_selected(zip_path, dat_path, output_type)
    print(f"Disk usage: {get_dir_size_human(dat_path)}")

    print("[2/5] Listing binary outputs...")
    output_files = get_output_files(dat_path, output_type)

    print(f"[3/5] Creating NetCDF: {nc_path}")
    os.makedirs(nc_path, exist_ok=True)

    xmin, xres, xrot, ymax, yrot, yres = gt
    xarr = np.array([xmin + xres * 0.5 + i * xres for i in range(cols)])
    yarr = np.array([ymax + yres * 0.5 + i * yres for i in range(rows)])

    with nc4.Dataset(f'{nc_path}/{nc_name}', "w", format="NETCDF4", diskless=False) as nc:
        nc.Conventions = "CF-1.6"
        # Create the dimensions for the rootgroup (nc)
        nc.createDimension("x", cols)
        nc.createDimension("y", rows)
        nc.createDimension("nv", 2)

        # Create x variable.
        x = nc.createVariable("x", "f8", ("x",))
        x.units = "m"
        x.long_name = "x coordinate of projection"
        x.standard_name = "projection_x_coordinate"
        x[:] = xarr

        # Create y variable.
        y = nc.createVariable("y", "f8", ("y",))
        y.units = "m"
        y.long_name = "y coordinate of projection"
        y.standard_name = "projection_y_coordinate"
        y[:] = yarr

        # Create the CRS variable. 
        grid_mapping_name = "transverse_mercator"  # This is a CF standard.
        crs = nc.createVariable(grid_mapping_name, "i2")

        # Populate the attributes using the osr.SRS. Some required by GMN.
        crs.grid_mapping_name = grid_mapping_name
        crs.false_easting = 500000.0
        crs.false_northing = 0.0
        crs.inverse_flattening = 298.257222101004
        crs.crs_wkt = 'PROJCS["NAD_1983_UTM_Zone_16N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-87],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26916"]]'
        crs.longitude_of_central_meridian = -87.0
        crs.latitude_of_projection_origin = 0.0
        crs.semi_major_axis = 6378137.0
        crs.semi_minor_axis = 6356752.314140356
        crs.scale_factor_at_central_meridian = 0.9996

        # Time variables
        print("[4/5] Creating time variables...")
        nc.createDimension("in_time") # input time dimension

        in_time = nc.createVariable("in_time", "f4", ("in_time",)) # input time variable
        in_time.standard_name = "time"
        in_time.long_name = "time"
        in_time.calendar = "standard"
        in_time.units = ref_time
        in_time.bounds = "in_time_bnds"

        in_time_bnds = nc.createVariable("in_time_bnds", "f4", ("in_time","nv",))
        in_time_bnds.units = ref_time

        # Create dates and time to populate the input time variable
        in_dates = [time_origin + timedelta(seconds=(i)*10800) for i in range(80)]
        for i, date in enumerate(in_dates):
            since_origin = (date-time_origin)
            hours_since_origin = since_origin.days*24 + (since_origin.seconds)/3600
            in_time[i] = hours_since_origin
            in_time_bnds[i] = [(hours_since_origin)-0.5, (hours_since_origin) + 0.5] # Assign time bounds.



        nc.createDimension("out_time") # input time dimension

        out_time = nc.createVariable("out_time", "f4", ("out_time",))
        out_time.standard_name = "time"
        out_time.long_name = "time"
        out_time.calendar = "standard"
        out_time.units = ref_time
        out_time.bounds = "out_time_bnds"

        out_time_bnds = nc.createVariable("out_time_bnds", "f4", ("out_time","nv",))
        out_time_bnds.units = ref_time

        # Create dates and time to populate the output time variable
        out_dates = [time_origin + timedelta(seconds=(i+1)*1800) for i in range(480)]
        for i, date in enumerate(out_dates):
            since_origin = (date-time_origin)
            hours_since_origin = since_origin.days*24 + (since_origin.seconds)/3600
            out_time[i] = hours_since_origin
            out_time_bnds[i] = [(hours_since_origin)-0.25, (hours_since_origin) + 0.25] # Assign time bounds.

        # Create output variable
        out_depth = nc.createVariable(
            "output_depth",        # variable name
            "f4",               # Specify data type.
            ("out_time", "y", "x"),         # Specify dimensions.
            zlib=True,          # Specify compression with zlib.
            complevel=4,        # Specify compression level.
            fill_value=-9999.,  # Specify the fill value.
            )
        out_depth.grid_mapping = grid_mapping_name   # CF required.
        out_depth.standard_name = "Depth"
        out_depth.long_name = f"flood_depth_{os.path.basename(nc_name)}"
        out_depth.units = "m"

        print("[5/5] Writing flood depth data...")
        tic = time.time()
        for i, t in enumerate(output_files):        # Loop over outputs.
        # for i, t in enumerate(output_files[:2]):  # Loop over outputs.
            arr = bin2array(t, rows, cols)

            arr[np.isnan(arr)] = -9999.     # Set missing values to -9999.
            # arr = arr.T                     # Transpose array for the dimensions to match the x,y format.
            out_depth[i,:,:] = arr          # Add the array to the current timestep.
            nc.sync()                       # Write changes to disk.
            if (i+1)%50==0:
                print(t, "done! {}/{}".format(i+1, len(output_files)))
        print("Finished writing output depth data. Duration: {}s".format(time.time()-tic))
        


    # Clean up
    subprocess.run(["rm", "-rf", dat_path])
    print(f"Temporary folder deleted: {dat_path}")
    print(f"✅ NetCDF saved to {nc_path}")
    print(f"⏱️ Total runtime: {time.time() - start_time:.2f}s")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Static grid setup (same for all D001, D002, etc.)
    # ---------------------------------------------------------------------
    cols, rows = 5474, 7976
    gt = (676222.46162369, 9.50608224323499, 0.0, 3891564.837097742, 0.0, -9.506082243234989)
    ref_time = "hours since 2013-02-02 03:00:00 -5:00"
    time_origin = datetime(1966, 2, 2, 3, tzinfo=timezone(-timedelta(hours=5)))

    # ---------------------------------------------------------------------
    # Input and output configuration
    # ---------------------------------------------------------------------
    zip_dir = "/lustre/orion/cli190/world-shared/Conasauga_Paper/DataAndMethods/4GCMFloodSimulations/2_OutputData/0_Simulation_Outputs/2BaseHygs/ACCESS_RegCM_baseline_flood_3hr"
    out_dir = "../../processed_data/netcdf"
    os.makedirs(out_dir, exist_ok=True)
    output_type = "H"

    # ---------------------------------------------------------------------
    # Loop through all ZIPs matching D*.zip
    # ---------------------------------------------------------------------
    zip_files = sorted(glob.glob(os.path.join(zip_dir, "D*.zip")))
    print(f"Found {len(zip_files)} zip files to process.")

    for zip_path in zip_files:
        basename = os.path.splitext(os.path.basename(zip_path))[0]  # e.g., D001
        nc_name = f"{basename}_ACC_future.nc"

        print(f"\n=== Processing {basename} ===")
        process_depth_data(zip_path, out_dir, output_type, nc_name,
                        cols, rows, gt, ref_time, time_origin)

    print("\n✅ All ZIPs processed successfully!")