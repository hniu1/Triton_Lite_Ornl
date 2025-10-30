# depth_steps.py
"""
Unified depth preprocessing (cfg-driven) replacing:
  01_conasauga_block_creation.ipynb
  02_netcdf_to_MOM_raster.ipynb
  03_block_selection_to_csv_export.ipynb
  04_extract_raster_from_netcdf.ipynb

Subcommands:
  block-grid        -> build pixel block grid + block index CSV
  mom               -> NetCDF 'output_depth' → overall MOM GeoTIFF + zeroed variant
  dissolve          -> raster→polygons (ignore 0/nodata), dissolve → shapefile
  block-ll-csv      -> blocks ∩ dissolved polygons → lower-left XY + watershed ID CSV
  extract           -> per-event clip of NetCDF raster by polygon (writes GeoTIFF)

Config: reads your existing 'directories.cfg'.
"""

from __future__ import annotations
import argparse, configparser, csv, math, os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.features import shapes
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, shape, Point
import geopandas as gpd


# ----------------------------- cfg helpers -----------------------------

def _read_cfg(cfg_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not cfg.read(cfg_path):
        raise FileNotFoundError(f"CFG not found: {cfg_path}")
    return cfg

def _get(cfg: configparser.ConfigParser, section: str, key: str, fallback: str | None = None) -> str:
    return cfg.get(section, key, fallback=fallback)


# ----------------------------- 1) block-grid -----------------------------

def step_block_grid(cfg: configparser.ConfigParser) -> tuple[Path, Path]:
    """
    Create block grid shapefile + block index CSV from DEM.
    Uses [Directories]:
      DEMPath, LargeGridShapefilePath, BlockIndexCSV (optional),
      BlockWidth (default=80), BlockHeight (default=80)
    """
    dem_path = Path(_get(cfg, "Directories", "DEMPath"))
    shp_out  = Path(_get(cfg, "Directories", "LargeGridShapefilePath"))
    csv_out  = Path(_get(cfg, "Directories", "BlockIndexCSV", fallback=str(shp_out.with_suffix(".csv"))))
    block_w  = int(_get(cfg, "Directories", "BlockWidth", fallback="80"))
    block_h  = int(_get(cfg, "Directories", "BlockHeight", fallback="80"))

    shp_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(dem_path) as ds:
        H, W = ds.height, ds.width
        transform, crs = ds.transform, ds.crs
        n_by = math.ceil(H / block_h)
        n_bx = math.ceil(W / block_w)

        geoms = []
        records = []

        bid = 0
        for by in range(n_by):
            row_off = by * block_h
            h = min(block_h, H - row_off)
            if h <= 0: continue
            for bx in range(n_bx):
                col_off = bx * block_w
                w = min(block_w, W - col_off)
                if w <= 0: continue

                win = Window(col_off=col_off, row_off=row_off, width=w, height=h)
                win_transform = rio.windows.transform(win, transform)
                minx, maxy = (win_transform * (0, 0))
                maxx, miny = (win_transform * (w, h))
                geoms.append(box(minx, miny, maxx, maxy))

                # lower-left pixel coordinate of this block
                x_ll, y_ll = rio.transform.xy(transform, row_off + h - 1, col_off, offset="ul")
                records.append((bid, row_off, col_off, h, w, x_ll, y_ll))
                bid += 1

    gdf = gpd.GeoDataFrame(
        {"block_id":[r[0] for r in records], "row_off":[r[1] for r in records],
         "col_off":[r[2] for r in records], "height":[r[3] for r in records],
         "width":[r[4] for r in records]},
        geometry=geoms, crs=crs
    )
    gdf.to_file(shp_out)
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_id","row_off","col_off","height","width","x_ll","y_ll"])
        w.writerows(records)

    print(f"[block-grid] wrote {shp_out} (blocks={len(gdf)})")
    print(f"[block-grid] wrote {csv_out}")
    return shp_out, csv_out


# ----------------------------- 2) MOM raster -----------------------------

def step_mom(cfg: configparser.ConfigParser) -> tuple[Path, Path]:
    """
    Open each NetCDF in NetCDFInputDir using variable 'output_depth' (configurable),
    take max across time and across files → Final_MOM_ACC_baseline.tif (float32).
    Also writes Final_MOM_ACC_baseline_zero.tif where non-nodata pixels are 0.

    Uses [Directories]:
      NetCDFInputDir, MOMRasterOutputDir, NetCDFVariable (default=output_depth)
    """
    in_dir   = Path(_get(cfg, "Directories", "NetCDFInputDir"))
    out_dir  = Path(_get(cfg, "Directories", "MOMRasterOutputDir"))
    var_name = _get(cfg, "Directories", "NetCDFVariable", fallback="output_depth")

    out_dir.mkdir(parents=True, exist_ok=True)
    netcdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower()==".nc"])
    if not netcdfs:
        raise FileNotFoundError(f"No .nc files in {in_dir}")

    overall = None
    out_meta = None

    for nc in netcdfs:
        with rio.open(f'NETCDF:"{nc}":{var_name}') as src:
            if overall is None:
                out_meta = src.meta.copy()
                out_meta.update(driver="GTiff", count=1, dtype="float32")
                overall = np.full((src.height, src.width), -np.inf, dtype=np.float32)

            local_max = None
            for b in range(1, src.count+1):
                arr = src.read(b).astype(np.float32)
                local_max = np.maximum(local_max, arr) if local_max is not None else arr

            overall = np.maximum(overall, local_max)

    # replace -inf with nodata if available
    nod = out_meta.get("nodata", -9999)
    overall[np.isneginf(overall)] = nod

    mom_path = out_dir / "Final_MOM_ACC_baseline.tif"
    with rio.open(mom_path, "w", **out_meta) as dst:
        dst.write(overall.astype(np.float32), 1)
    print(f"[mom] wrote {mom_path}")

    # zero variant (non-nodata -> 0)
    zero_path = out_dir / "Final_MOM_ACC_baseline_zero.tif"
    with rio.open(mom_path) as src:
        meta = src.meta.copy()
        nod = src.nodata if src.nodata is not None else -9999
        band = src.read(1)
        mask = band != nod
        band[mask] = 0
        meta.update(dtype=rio.int32, nodata=int(nod))
        with rio.open(zero_path, "w", **meta) as dst:
            dst.write(band.astype(rio.int32), 1)
    print(f"[mom] wrote {zero_path}")

    return mom_path, zero_path


# ----------------------------- 3) raster → polygons (dissolve) -----------------------------

def step_dissolve(cfg: configparser.ConfigParser) -> Path:
    """
    Convert MOM raster to polygons (ignore 0 and nodata), dissolve into one shapefile.

    Uses [Directories]:
      MOM_Raster_Path (else uses MOMRasterOutputDir/Final_MOM_ACC_baseline.tif)
      OutputShapefilePath
    """
    mom_dir  = Path(_get(cfg, "Directories", "MOMRasterOutputDir"))
    raster_path = Path(_get(cfg, "Directories", "MOM_Raster_Path",
                            fallback=str(mom_dir / "Final_MOM_ACC_baseline.tif")))
    out_shp = Path(_get(cfg, "Directories", "OutputShapefilePath"))

    out_shp.parent.mkdir(parents=True, exist_ok=True)

    with rio.open(raster_path) as src:
        img = src.read(1)
        nod = src.nodata if src.nodata is not None else -9999
        m = (img != 0) & (img != nod)
        polys = list(shapes(img, mask=m, transform=src.transform))
        geoms = [shape(g) for g, v in polys if v != nod]

        if not geoms:
            raise RuntimeError("No polygons generated from raster (all zeros/nodata?).")

        gdf = gpd.GeoDataFrame({"value": 1}, geometry=[g for g in geoms], crs=src.crs)
        dissolved = gdf.dissolve(by="value")
        dissolved = dissolved.drop(columns=["value"]).reset_index(drop=True)
        dissolved.to_file(out_shp, driver="ESRI Shapefile")

    print(f"[dissolve] wrote {out_shp}")
    return out_shp


# ----------------------------- 4) block selection → LL CSV -----------------------------

def step_block_ll_csv(cfg: configparser.ConfigParser) -> Path:
    """
    Blocks that intersect dissolved polygons → CSV with lower-left XY + watershed Name + numeric ID.

    Uses [Directories]:
      ShapefileDir (optional root), or:
      DissolvedShapefilePath,
      BlocksShapefilePath,
      shapefile_path (watershed polygons with 'Name' field),
      output_csv_path
    """
    shp_dir     = Path(_get(cfg, "Directories", "ShapefileDir", fallback=""))
    dissolved   = Path(_get(cfg, "Directories", "DissolvedShapefilePath",
                            fallback=str(shp_dir / "MOM_raster_dissolved.shp")))
    blocks_shp  = Path(_get(cfg, "Directories", "BlocksShapefilePath",
                            fallback=str(shp_dir / "blocks_conasauga.shp")))
    watersheds  = Path(_get(cfg, "Directories", "shapefile_path"))  # must have 'Name'
    out_csv     = Path(_get(cfg, "Directories", "output_csv_path"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    river_polygons = gpd.read_file(dissolved)
    grid_polygons  = gpd.read_file(blocks_shp).to_crs(river_polygons.crs)

    # spatial join to intersect blocks
    intersects = gpd.sjoin(grid_polygons, river_polygons, how="inner", predicate="intersects")

    # lower-left point of each intersected block
    def ll_pt(poly):
        minx, miny, maxx, maxy = poly.bounds
        return Point(minx, miny)

    ll_points = intersects.geometry.apply(ll_pt)
    coords_gdf = gpd.GeoDataFrame(geometry=ll_points, crs=river_polygons.crs)

    # tag watershed Name
    ws_polys = gpd.read_file(watersheds).to_crs(coords_gdf.crs)

    names = []
    for pt in coords_gdf.geometry:
        hit = ws_polys[ws_polys.contains(pt)]
        names.append(hit.iloc[0]["Name"] if not hit.empty else "Outside")

    X = [p.x for p in coords_gdf.geometry]
    Y = [p.y for p in coords_gdf.geometry]
    df = gpd.pd.DataFrame({"X": X, "Y": Y, "Name": names})

    # numeric IDs by name
    ids = {n: i for i, n in enumerate(df["Name"].unique(), start=1)}
    df["ID"] = df["Name"].map(ids)

    df.to_csv(out_csv, index=False)
    print(f"[block-ll-csv] wrote {out_csv} (rows={len(df)})")
    return out_csv


# ----------------------------- 5) extract per-event raster by polygon -----------------------------

def step_extract(cfg: configparser.ConfigParser) -> None:
    """
    Clip each NetCDF event to a polygon (e.g., a watershed), write GeoTIFF.

    Uses [Directories]:
      NetCDFInputDir, ExtractOutputDir, ExtractShapefilePath,
      NetCDFVariable (default=output_depth),
      ExtractEventNameTemplate (default: 'D{num:03d}_test.nc' for input),
      ExtractOutputNameTemplate (default: 'ACC_D{num:03d}_clipped.tif'),
      ExtractEventStart (default=1), ExtractEventEnd (default=40)
    """
    in_dir   = Path(_get(cfg, "Directories", "NetCDFInputDir"))
    out_dir  = Path(_get(cfg, "Directories", "ExtractOutputDir", fallback=str(in_dir / "clipped")))
    shp_path = Path(_get(cfg, "Directories", "ExtractShapefilePath"))
    var_name = _get(cfg, "Directories", "NetCDFVariable", fallback="output_depth")

    name_in  = _get(cfg, "Directories", "ExtractEventNameTemplate", fallback="D{num:03d}_test.nc")
    name_out = _get(cfg, "Directories", "ExtractOutputNameTemplate", fallback="ACC_D{num:03d}_clipped.tif")
    ev_start = int(_get(cfg, "Directories", "ExtractEventStart", fallback="1"))
    ev_end   = int(_get(cfg, "Directories", "ExtractEventEnd", fallback="40"))

    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = gpd.read_file(shp_path)

    for i in range(ev_start, ev_end + 1):
        in_nc  = in_dir / name_in.format(num=i)
        out_tif = out_dir / name_out.format(num=i)
        if not in_nc.exists():
            print(f"[extract] skip (missing): {in_nc}")
            continue

        try:
            with rio.open(f'NETCDF:"{in_nc}":{var_name}') as src:
                # If multi-band (time), clip each band and take max (MOM) of the clipped stack
                clipped_max = None
                for b in range(1, src.count + 1):
                    arr, tr = rio_mask(src, shapes.geometry, crop=True, filled=True)
                    # arr shape: (1, h, w) per read; when looping bands, we read band-by-band
                    arr = arr.astype(np.float32)
                    clipped_max = np.maximum(clipped_max, arr) if clipped_max is not None else arr

                meta = src.meta.copy()
                meta.update(driver="GTiff", count=1, height=clipped_max.shape[1],
                            width=clipped_max.shape[2], transform=tr, dtype="float32")

                with rio.open(out_tif, "w", **meta) as dst:
                    dst.write(clipped_max[0], 1)

            print(f"[extract] wrote {out_tif}")

        except Exception as e:
            print(f"[extract] ERROR {in_nc}: {e}")


# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser("Depth preprocessing pipeline")
    ap.add_argument("--cfg", required=True, help="Path to directories.cfg")
    ap.add_argument("--step", choices=["block-grid","mom","dissolve","block-ll-csv","extract","all"], default="all")
    args = ap.parse_args()

    cfg = _read_cfg(args.cfg)

    steps = [args.step] if args.step != "all" else ["block-grid","mom","dissolve","block-ll-csv","extract"]
    for s in steps:
        print(f"\n=== {s} ===")
        if s == "block-grid":       step_block_grid(cfg)
        elif s == "mom":            step_mom(cfg)
        elif s == "dissolve":       step_dissolve(cfg)
        elif s == "block-ll-csv":   step_block_ll_csv(cfg)
        elif s == "extract":        step_extract(cfg)

if __name__ == "__main__":
    main()

'''
# Everything in sequence
python depth_steps.py --cfg directories.cfg --step all

# Or individual steps
python depth_steps.py --cfg directories.cfg --step block-grid
python depth_steps.py --cfg directories.cfg --step mom
python depth_steps.py --cfg directories.cfg --step dissolve
python depth_steps.py --cfg directories.cfg --step block-ll-csv
python depth_steps.py --cfg directories.cfg --step extract
'''