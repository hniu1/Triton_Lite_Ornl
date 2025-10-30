# ---------- STEP 1: Generate NetCDF from TRITON binary frames (H/QX/QY/...) ----------
import netCDF4 as nc4
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import glob, os

def _events_from_cfg(E):
    start_ev = int(E["start_event"]); end_ev = int(E["end_event"])
    skip = [s.strip() for s in E.get("skip_events", "").split(",") if s.strip()]
    evs = [f"D{i:03d}" for i in range(start_ev, end_ev + 1)]
    return [e for e in evs if e not in set(skip)]

def _gt_from_cfg(G):
    xmin = float(G["xmin"]); xres = float(G["xres"]); xrot = float(G["xrot"])
    ymax = float(G["ymax"]); yrot = float(G["yrot"]); yres = float(G["yres"])
    return (xmin, xres, xrot, ymax, yrot, yres)

def _make_xy(cols, rows, gt):
    xmin, xres, xrot, ymax, yrot, yres = gt
    xarr = np.array([xmin + xres*0.5 + i*xres for i in range(cols)], dtype=np.float64)
    yarr = np.array([ymax + yres*0.5 + i*yres for i in range(rows)], dtype=np.float64)
    return xarr, yarr

def _time_origin_from_cfg(T):
    tz = timezone(-timedelta(hours=int(T.get("origin_utc_offset_hours", 0))))
    return datetime(int(T["origin_year"]), int(T["origin_month"]), int(T["origin_day"]),
                    int(T["origin_hour"]), tzinfo=tz)

def _binary_to_array(fp, rows, cols, threshold=0.0, pad=1, dtype="float32"):
    arr = np.fromfile(fp, dtype=np.dtype(dtype))
    # Conasauga format: array contains a (rows+2*pad) x (cols+2*pad) grid
    try:
        arr = arr.reshape((rows + 2*pad, cols + 2*pad))
    except ValueError:
        raise ValueError(f"Binary length {arr.size} not compatible with shape {(rows+2*pad, cols+2*pad)} for file {fp}")
    arr = arr[pad:rows+pad, pad:cols+pad]
    if threshold is not None:
        arr = np.where(arr > threshold, arr, np.nan)
    return arr

def step_generate_netcdf_from_binaries(cfg):
    """
    Read TRITON binary frames for each event and write a NetCDF file with:
      dims: x, y, in_time, out_time
      vars: x, y, <CRS>, in_time(+bounds), out_time(+bounds), {var_name} [out_time,y,x]
    Matches your original metadata and timing.
    """
    P = cfg["Paths"]; E = cfg["Events"]; O = cfg["Output"]; G = cfg["Grid"]; T = cfg["Time"]; B = cfg["BinaryReader"]

    bin_root   = Path(P["bin_root"])
    netcdf_dir = Path(P["netcdf_dir"]); netcdf_dir.mkdir(parents=True, exist_ok=True)

    output_type = O.get("output_type", "H").upper()
    name_tmpl   = O.get("data_name_tmpl", "{event}.nc")
    var_name    = O.get("var_name", "output_depth")
    fill_value  = float(O.get("fill_value", -9999.0))

    cols = int(G["cols"]); rows = int(G["rows"])
    gt   = _gt_from_cfg(G)
    xarr, yarr = _make_xy(cols, rows, gt)

    ref_time   = T["ref_time"]
    time_origin = _time_origin_from_cfg(T)
    in_steps   = int(T.get("in_steps", 80))
    in_step_s  = int(T.get("in_step_sec", 10800))
    out_steps  = int(T.get("out_steps", 480))
    out_step_s = int(T.get("out_step_sec", 1800))

    threshold  = float(B.get("threshold", 0.0))
    pad        = int(B.get("pad", 1))
    dtype      = B.get("dtype", "float32")

    events = _events_from_cfg(E)

    for ev in events:
        # Locate binary directory for this event
        bin_dir = bin_root / ev / "output" / "flood2d" / "bin"
        if not bin_dir.exists():
            print(f"[generate_netcdf] MISSING dir: {bin_dir}")
            continue

        pattern = str(bin_dir / f"{output_type}*")
        output_files = sorted(sorted(glob.glob(pattern)), key=len)
        if len(output_files) == 0:
            print(f"[generate_netcdf] No binaries matching {pattern}")
            continue

        data_name = name_tmpl.format(event=ev)
        nc_path = netcdf_dir / data_name
        if nc_path.exists():
            print(f"[generate_netcdf] exists, skip: {nc_path}")
            continue

        print(f"[generate_netcdf] {ev}: frames={len(output_files)} → {nc_path}")

        # --- Create NetCDF
        # NOTE: memory keyword is kept from your script; if netCDF4 version ignores, it's harmless.
        nc = nc4.Dataset(str(nc_path), "w", memory=20560)

        # dimensions
        nc.createDimension("x", cols)
        nc.createDimension("y", rows)
        nc.createDimension("nv", 2)
        nc.createDimension("in_time")
        nc.createDimension("out_time")

        # coords
        x = nc.createVariable("x", "f8", ("x",))
        x.units = "m"; x.long_name = "x coordinate of projection"; x.standard_name = "projection_x_coordinate"
        x[:] = xarr

        y = nc.createVariable("y", "f8", ("y",))
        y.units = "m"; y.long_name = "y coordinate of projection"; y.standard_name = "projection_y_coordinate"
        y[:] = yarr

        # CRS (kept same as your original CF attributes)
        grid_mapping_name = "transverse_mercator"
        crs = nc.createVariable(grid_mapping_name, "i2")
        crs.grid_mapping_name = grid_mapping_name
        crs.false_easting = 500000.0
        crs.false_northing = 0.0
        crs.inverse_flattening = 298.257222101004
        crs.crs_wkt = ('PROJCS["NAD_1983_UTM_Zone_16N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",'
                       'SPHEROID["GRS 1980",6378137,298.257222101004,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],'
                       'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],'
                       'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-87],'
                       'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],'
                       'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26916"]]')
        crs.longitude_of_central_meridian = -87.0
        crs.latitude_of_projection_origin = 0.0
        crs.semi_major_axis = 6378137.0
        crs.semi_minor_axis = 6356752.314140356
        crs.scale_factor_at_central_meridian = 0.9996

        # in_time (+bounds)
        in_time = nc.createVariable("in_time", "f4", ("in_time",))
        in_time.standard_name = "time"; in_time.long_name = "time"; in_time.calendar = "standard"; in_time.units = ref_time
        in_time.bounds = "in_time_bnds"
        in_time_bnds = nc.createVariable("in_time_bnds", "f4", ("in_time","nv",))
        in_time_bnds.units = ref_time

        in_dates = [time_origin + timedelta(seconds=i*in_step_s) for i in range(in_steps)]
        in_vals = []
        for d in in_dates:
            delta = d - time_origin
            in_vals.append(delta.days*24 + delta.seconds/3600.0)
        in_time[:] = np.array(in_vals, dtype=np.float32)
        in_time_bnds[:,0] = np.array(in_vals, dtype=np.float32) - 0.5
        in_time_bnds[:,1] = np.array(in_vals, dtype=np.float32) + 0.5

        # out_time (+bounds)
        out_time = nc.createVariable("out_time", "f4", ("out_time",))
        out_time.standard_name = "time"; out_time.long_name = "time"; out_time.calendar = "standard"; out_time.units = ref_time
        out_time.bounds = "out_time_bnds"
        out_time_bnds = nc.createVariable("out_time_bnds", "f4", ("out_time","nv",))
        out_time_bnds.units = ref_time

        out_dates = [time_origin + timedelta(seconds=(i+1)*out_step_s) for i in range(out_steps)]
        out_vals = []
        for d in out_dates:
            delta = d - time_origin
            out_vals.append(delta.days*24 + delta.seconds/3600.0)
        out_time[:] = np.array(out_vals, dtype=np.float32)
        out_time_bnds[:,0] = np.array(out_vals, dtype=np.float32) - 0.25
        out_time_bnds[:,1] = np.array(out_vals, dtype=np.float32) + 0.25

        # data variable (matches your metadata)
        out_depth = nc.createVariable(var_name, "f4", ("out_time","y","x"),
                                      zlib=True, complevel=4, fill_value=fill_value)
        out_depth.grid_mapping = grid_mapping_name
        out_depth.standard_name = "Depth"
        out_depth.long_name = f"flood_depth_{data_name}"
        out_depth.units = "m"

        # write frames
        for i, bf in enumerate(output_files[:out_steps]):  # safeguard if more files than out_steps
            arr = _binary_to_array(bf, rows, cols, threshold=threshold, pad=pad, dtype=dtype)
            arr = np.nan_to_num(arr, nan=fill_value).astype(np.float32)
            out_depth[i, :, :] = arr
            nc.sync()
            if (i+1) % 50 == 0 or (i+1) == len(output_files[:out_steps]):
                print(f"[generate_netcdf] {ev}: wrote frame {i+1}/{min(len(output_files), out_steps)}")

        nc.Conventions = "CF-1.6"
        nc.archive     = data_name

        # finalize
        nc.close()
        print(f"[generate_netcdf] DONE → {nc_path}")


'''
python convert_netcdf.py --cfg depth_process.cfg --step generate_netcdf
'''