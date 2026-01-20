import os
import xarray as xr
import numpy as np

nc_dir = "/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/processed_data/netcdf"

results = []

for fname in sorted(os.listdir(nc_dir)):
    if not fname.endswith("_ACC_future.nc"):
        continue

    path = os.path.join(nc_dir, fname)
    print(f"[CHECK] {fname}")

    ds = xr.open_dataset(path)

    if "output_depth" not in ds:
        print("  ❌ missing output_depth")
        results.append((fname, "MISSING"))
        ds.close()
        continue

    da = ds["output_depth"]
    fill = da.attrs.get("_FillValue", -9999.0)

    has_data = False

    # ---- FAST CHECK: stop at first timestep with any valid value ----
    for t in range(da.sizes["out_time"]):
        slab = da.isel(out_time=t).values   # read ONE 2D slice only
        if np.nanmax(slab) > 0:
            has_data = True
            break

    if has_data:
        print("  ✅ HAS DATA")
        results.append((fname, "HAS_DATA"))
    else:
        print("  ⚠️ ALL FILL (-9999)")
        results.append((fname, "ALL_FILL"))

    ds.close()

print("\n===== SUMMARY =====")
for r in results:
    print(r)
