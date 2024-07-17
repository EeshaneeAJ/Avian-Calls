import geopandas as gpd

# Load country shapefiles from the downloaded dataset
world = gpd.read_file('data/ne_110m_admin_0_countries.shp')

# Print unique country names
print(world['NAME'].unique())
