import geopandas as gpd
from shapely.geometry import Point



shapefile_Cona = "/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/shapefiles/shapefiles/ConasaugaBasinHUC12.shp"
# Load the shapefile using GeoPandas
shapes_Cona = gpd.read_file(shapefile_Cona)


shapefile_flow = "/lustre/orion/proj-shared/cli138/7hn/triton/Triton_Lite_Ornl/shapefiles/flow_locations/flow_locations.shp"
# Load the shapefile using GeoPandas
shapes_flow = gpd.read_file(shapefile_flow)

print(shapes_flow.head())


# Assuming your DataFrame is called shapes_all
gdf_points = gpd.GeoDataFrame(
    shapes_flow,
    geometry=gpd.points_from_xy(shapes_flow["X"], shapes_flow["Y"]),
    crs=shapes_flow.crs  # <-- replace with correct CRS
)

print(shapes_Cona.crs)

if gdf_points.crs != shapes_Cona.crs:
    gdf_points = gdf_points.to_crs(shapes_Cona.crs)

points_with_watershed = gpd.sjoin(
    gdf_points,
    shapes_Cona,
    how="left",
    predicate="within"   # use "intersects" if points lie on boundaries
)


points_with_watershed[
    ["Loc", "X", "Y", "HUC12", "Name"]
].head()

# save the result to a csv file
points_with_watershed[
    ["Loc", "X", "Y", "Name"]
].to_csv("/lustre/orion/proj-shared/cli138/7hn/triton/ttu/processed_data/hyg/loc_watershed.csv", index=False)