import os
import gc
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import configparser
from pathlib import Path
from shapely.geometry import box
import numpy as np

# ------------------ safety (important) ------------------
os.environ["GDAL_NUM_THREADS"] = "1"
os.environ["GDAL_CACHEMAX"] = "64"

# ------------------ setup ------------------
script_path = Path(__file__).parent.resolve()
os.chdir(script_path)

watershed_name = "Sugar Creek"
watershed_key = watershed_name.strip().lower()
watershed_name_lower = watershed_key.replace(" ", "_")

config = configparser.ConfigParser()
config.read("directories.cfg")

base_dir = config.get("Directories", "NetCDFInputDir")
shapefile_path = config.get("Directories", "shapefile_path")
output_dir = f"{config.get('Directories', 'water_depth_rasters')}_{watershed_name_lower}"
os.makedirs(output_dir, exist_ok=True)

TMP_DIR = os.environ.get("SLURM_TMPDIR", "/tmp")

# ------------------ load watershed ------------------
shapes = gpd.read_file(shapefile_path)
shapes["Name_norm"] = shapes["Name"].str.strip().str.lower()
shapes = shapes[shapes["Name_norm"] == watershed_key]

if shapes.empty:
    raise RuntimeError(f"Watershed '{watershed_name}' not found")

# clean geometry (prevents hangs)
shapes["geometry"] = shapes.geometry.buffer(0)
shapes = shapes.explode(ignore_index=True)

# ------------------ helpers ------------------
def nc_to_tif(nc_path, tif_path):
    """Read NetCDF via GDAL and write GeoTIFF (NO masking here)."""
    with rasterio.open(f"NETCDF:{nc_path}:output_depth") as src:
        data = src.read(1)
        meta = src.meta.copy()
        if np.all(data == src.nodata):
            print(f"[INFO] No flood for {nc_name}, skipping output")
        else:
            print(f"[INFO] Writing data for {nc_name}")

        if meta.get("crs") is None:
            meta["crs"] = "EPSG:26916"  # NAD83 / UTM 16N (from your grid_mapping)

    meta.update(driver="GTiff", count=1, compress="lzw")

    with rasterio.open(tif_path, "w", **meta) as dst:
        dst.write(data, 1)

def clip_tif(src_tif, out_tif, shapes):
    with rasterio.open(src_tif) as src:
        # reproject shapes if needed
        if shapes.crs and src.crs and shapes.crs != src.crs:
            shapes = shapes.to_crs(src.crs)

        geom = shapes.geometry.intersection(box(*src.bounds))
        geom = geom[~geom.is_empty]
        if geom.empty:
            return False

        out_img, out_tr = mask(src, geom, crop=True, all_touched=True)

        meta = src.meta.copy()
        meta.update(
            height=out_img.shape[1],
            width=out_img.shape[2],
            transform=out_tr,
            compress="lzw",
        )

    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(out_img)

    return True

# ------------------ main loop ------------------
for i in range(1, 41):
    nc_name = f"D{i:03d}_ACC_future.nc"
    nc_path = os.path.join(base_dir, nc_name)

    out_tif = os.path.join(output_dir, f"ACC_D{i:03d}_{watershed_name_lower}.tif")
    if os.path.exists(out_tif):
        print(f"[SKIP] {out_tif}")
        continue

    tmp_tif = os.path.join(TMP_DIR, f"tmp_{i:03d}.tif")
    print(f"[PROCESS] {nc_name}")

    try:
        nc_to_tif(nc_path, tmp_tif)
        ok = clip_tif(tmp_tif, out_tif, shapes)
        if ok:
            print(f"[DONE] {out_tif}")
        else:
            print(f"[WARN] No overlap for {nc_name}")

    except Exception as e:
        print(f"[ERROR] {nc_name}: {e}")

    finally:
        if os.path.exists(tmp_tif):
            os.remove(tmp_tif)
        gc.collect()
