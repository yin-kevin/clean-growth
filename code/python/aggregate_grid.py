# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:12:23 2022

@author: ky297
"""

#%% install packages

import subprocess
import sys
import os
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

# check system environment for packages and install if necessary
# package versions that were used at time of writing are commented above

# Pandas 2.0.0
try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
    import pandas as pd

# Numpy 1.24.2
try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'numpy'])
finally:
    import numpy as np

# GeoPandas 1.1.1
try:
    import geopandas as gpd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopandas'])
finally:
    import geopandas as gpd

# Matplotlib 3.7.1
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
finally:
    import matplotlib.pyplot as plt

# Shapely 2.0.1
try:
    from shapely.geometry import Point, LineString
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'shapely'])
finally:
    from shapely.geometry import Point, LineString
    
# GeoPy 2.3.0
try:
    from geopy.distance import distance
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopy'])
finally:
    from geopy.distance import distance


#%% parameters

# 1. root directory
directory_path = os.path.realpath(__file__)[:-30]
os.chdir(directory_path)

# 2. data path
data_path = os.path.join(directory_path, "data")


#%% override option

# if running this script from the master, override the directory path
try:
    directory_path = master_directory_path
    os.chdir(master_directory_path)
    data_path = os.path.join(master_directory_path, "data")
except NameError:
    pass



#%% load shapefile

csr_path = os.path.join(data_path, "shapefile", "selected_regions", "selected_regions.shp")
csr_shp = gpd.read_file(csr_path)


# plot shapefile
# =============================================================================
# fig, ax = plt.subplots(figsize=(20,20), frameon=False)
# csr_shp.plot(ax=ax,
#             figsize = (20, 20), 
#             color = "white", 
#             edgecolor = "black", 
#             linewidth = 0.1)
# plt.show(block=False)
# =============================================================================


#%% process line data

# load data
df_stations = pd.read_csv(os.path.join(data_path, "sql_output", "vertices.csv"))
df_lines = pd.read_csv(os.path.join(data_path, "sql_output", "links_cap.csv"))
df_intersections = pd.read_csv(os.path.join(data_path, "sql_output", "intersections.csv"))

# keep only intersections that touch two stations
df_intersections = df_intersections.drop_duplicates(subset=['line_id', 'station_id'], keep='first')
df_intersections = df_intersections[df_intersections['line_id'].duplicated(keep=False)]

# keep only lines that touch stations on both ends
df_s2slines = df_lines.merge(df_intersections, 
                             on=['line_id'], 
                             how='left', 
                             indicator=True)

df_s2slines = df_s2slines[df_s2slines['_merge'] == 'both']
df_s2slines = df_s2slines.drop_duplicates('line_id')
df_s2slines = df_s2slines[['line_id', 'longitude_1', 'latitude_1', 'longitude_2', 'latitude_2', 'voltage_kv', 'max_capacity_mw']]

# turn endpoints into point geometries
df_s2slines['point_A'] = df_s2slines.apply(lambda x: Point(x['longitude_1'], x['latitude_1']), axis = 1)
df_s2slines['point_B'] = df_s2slines.apply(lambda x: Point(x['longitude_2'], x['latitude_2']), axis = 1)
df_s2slines['line'] = df_s2slines.apply(lambda x: LineString([x['point_A'], x['point_B']]), axis = 1)



#%% find edges between zones

# make the line dataframe a geodataframe and change the CRS
df_s2slines = gpd.GeoDataFrame(df_s2slines, geometry = 'point_A')
df_s2slines = df_s2slines.set_crs("EPSG:4326")
csr_shp = csr_shp.to_crs("EPSG:4326")

# calculate which region intersects point_A in a given line
edges = gpd.sjoin(df_s2slines, csr_shp, how='left', predicate='intersects')
edges = edges[['line_id', 'point_A', 'point_B', 'line', 'csr_id', 'voltage_kv', 'max_capacity_mw']]

# set point_B to the geometry and calculate which region intersects point_B
edges = gpd.GeoDataFrame(edges, geometry = 'point_B')
edges = edges.set_crs("EPSG:4326")
edges = gpd.sjoin(edges, csr_shp, how='left', predicate='intersects')


# clean up the edges table 
edges = edges[['csr_id_left', 'csr_id_right', 'voltage_kv', 'max_capacity_mw']]
edges = edges.rename(columns={ 'csr_id_left': 'csr_id_A', 'csr_id_right': 'csr_id_B'})

# delete if loop from zone to itself
edges = edges[edges['csr_id_A'] != edges['csr_id_B']]
edges.reset_index(inplace=True)
edges.drop(columns=['index'], inplace=True)



#%% construct final edges table

# construct edges table (for both those with and without voltage data)
df_csr_edges = edges.dropna(subset=['csr_id_A', 'csr_id_B'])
df_sort_csr_edges = pd.DataFrame(np.sort(df_csr_edges[['csr_id_A', 'csr_id_B']])[:,::1], 
                               columns=['csr_id_A', 'csr_id_B']
                               )

# recombine voltages and capacity after sorting each row lowest to highest by id
voltages = df_csr_edges[['voltage_kv','max_capacity_mw']].reset_index().drop(columns='index')
df_csr_edges = pd.concat([df_sort_csr_edges, voltages], axis='columns')

# replace NaNs with a '0' string, to be turned back into NaNs after string processing and record which rows this was done
df_csr_edges['v_data'] = 1
df_csr_edges.loc[df_csr_edges['voltage_kv'].isnull(), 'v_data'] = 0




#%% line aggregation

# aggregate over zones
df_csr_edges['num_of_edge'] = 1
#df_csr_edges.drop(columns='voltage', inplace=True)
df_csr_edges = df_csr_edges.groupby(['csr_id_A','csr_id_B']).agg({'voltage_kv':'sum','max_capacity_mw':'sum','num_of_edge':'sum'})


# turn id indexes into columns
df_csr_edges.reset_index(inplace=True)

# calculate centroids
m_centroids = pd.DataFrame(csr_shp.to_crs("EPSG:4326").centroid)
m_centroids = m_centroids.rename(columns={ 0:'centroid' })
m_centroids['csr_id'] = csr_shp['csr_id']

# remove Canada and Unalaska centroids
canada_ids = csr_shp[csr_shp['ctry_code'] == 'CA']
canada_ids = canada_ids['csr_id']
unalaska_ids = csr_shp[csr_shp['region'] == 'Unalaska']
unalaska_ids = unalaska_ids['csr_id']
m_centroids = m_centroids[~m_centroids['csr_id'].isin(canada_ids)]
m_centroids = m_centroids[~m_centroids['csr_id'].isin(unalaska_ids)]

# import Canadian population raster
canada_pop_path = os.path.join(data_path, "shapefile", "canada_population", "griddedPopulationCanada10km_2016.shp")
df_canada_pop =  gpd.read_file(canada_pop_path)
df_canada_pop = df_canada_pop.to_crs("EPSG:4326")

# calculate the center of each raster block
df_canada_pop['rast_centroid'] = df_canada_pop.centroid
df_canada_pop = gpd.GeoDataFrame(df_canada_pop, geometry = 'rast_centroid')
df_canada_pop.to_crs(csr_shp.crs, inplace=True)
df_canada_pop['x'] = df_canada_pop['rast_centroid'].x
df_canada_pop['y'] = df_canada_pop['rast_centroid'].y

# assign block-centroids to provinces
canada_shp = csr_shp[csr_shp['ctry_code'] == 'CA']
canada_shp = gpd.GeoDataFrame(canada_shp, geometry = 'geometry')
df_canada_pop = gpd.sjoin(df_canada_pop, canada_shp[['csr_id', 'geometry']], how='left', predicate='within')
df_canada_pop.dropna(subset=['csr_id'], inplace=True)

# sum population by province
df_province_pop = df_canada_pop[['TOT_POP2A','csr_id']].groupby('csr_id').sum()
df_province_pop.rename(columns={'TOT_POP2A':'total_pop'},inplace=True)

# calculate weighted values
df_canada_pop = df_canada_pop.merge(df_province_pop, how='left', on='csr_id')
df_canada_pop['weighted_x'] = (df_canada_pop['TOT_POP2A'] / df_canada_pop['total_pop']) * df_canada_pop['x']
df_canada_pop['weighted_y'] = (df_canada_pop['TOT_POP2A'] / df_canada_pop['total_pop']) * df_canada_pop['y']

# find aggregate weighted centroids by province
w_centroids = df_canada_pop[['weighted_x','weighted_y','csr_id']].groupby('csr_id',as_index=False).sum()
w_centroids['centroid'] = w_centroids.apply(lambda x: Point(x['weighted_x'], x['weighted_y']), axis = 1)
w_centroids = w_centroids[['csr_id', 'centroid']]

# hardcode Unalaska centroid
unalaska_centroid = gpd.GeoDataFrame(index=[0], geometry = [Point(-166.725, 53.8)], crs="EPSG:4326")
unalaska_ids = unalaska_ids.reset_index().drop(columns=['index'])
unalaska_centroid = pd.concat([unalaska_centroid, unalaska_ids], axis=1)
unalaska_centroid.rename(columns={'geometry':'centroid'}, inplace=True)

# merge Canadian and Unalaskan centroids with m_centroids
m_centroids = pd.concat([m_centroids, unalaska_centroid, w_centroids], axis=0)
m_centroids.reset_index(inplace=True)
m_centroids.drop(columns=['index'],inplace=True)

# merge by zone id
df_csr_edges = df_csr_edges.merge(m_centroids, how='left', left_on='csr_id_A', right_on='csr_id')
df_csr_edges.drop(columns='csr_id', inplace=True)
df_csr_edges = df_csr_edges.merge(m_centroids, how='left', left_on='csr_id_B', right_on='csr_id')
df_csr_edges.drop(columns='csr_id', inplace=True)

# stripping geoms into lat and lon for geopy distance function
df_csr_edges['latA'] = df_csr_edges['centroid_x'].apply(lambda a: a.y) 
df_csr_edges['lonA'] = df_csr_edges['centroid_x'].apply(lambda a: a.x) 
df_csr_edges['latB'] = df_csr_edges['centroid_y'].apply(lambda a: a.y) 
df_csr_edges['lonB'] = df_csr_edges['centroid_y'].apply(lambda a: a.x) 

# turning into tuples
df_csr_edges['tupleA'] = list(zip(df_csr_edges.latA, df_csr_edges.lonA))
df_csr_edges['tupleB'] = list(zip(df_csr_edges.latB, df_csr_edges.lonB))

# calculate distance
df_csr_edges['distance_km'] = df_csr_edges[['tupleA', 'tupleB']].apply(lambda x: distance(x['tupleA'], x['tupleB']).km, axis=1)
df_csr_edges = df_csr_edges[['csr_id_A', 
                             'csr_id_B', 
                             'num_of_edge', 
                             'voltage_kv', 
                             'max_capacity_mw', 
                             'distance_km']]



#%% construct matrix

# save a copy of the edges with voltage and capacity data
edges_v = edges
#edges = edges_v

# don't need voltage for matrix
edges = edges.drop(columns=['voltage_kv','max_capacity_mw'])

# sort each row such that the highest zone id comes first (1 or -1 determines low first or high first respectively)
high2low_edges = pd.DataFrame(np.sort(edges[['csr_id_A', 'csr_id_B']])[:,::-1], 
                              columns=['csr_id_A', 'csr_id_B']
                              )
low2high_edges = pd.DataFrame(np.sort(edges[['csr_id_A', 'csr_id_B']])[:,::1], 
                              columns=['csr_id_A', 'csr_id_B']
                              )


# sort low to high to examine frequency of certain edges
high2low_edges.sort_values(by='csr_id_A', axis='index', inplace=True)
low2high_edges.sort_values(by='csr_id_A', axis='index', inplace=True)

# for all zone ids not in either column, create a connection between that zone and a new dummy zone 9999
# EXPL: doesn't matter whether we use high2low or the opposite, the unique values across A and B are the same
# EXPL: first run left merges to identify which ids occur in both, then delete those, then do it again for B

# find unique ids in A and B
unique_A = pd.DataFrame(high2low_edges['csr_id_A'].unique())
unique_B = pd.DataFrame(high2low_edges['csr_id_B'].unique())
unique_A.rename(columns={ 0: 'csr_id' }, inplace=True)
unique_B.rename(columns={ 0: 'csr_id' }, inplace=True)


# remove those that are in A
csr_id_leftover = csr_shp[['csr_id']].merge(unique_A, 
                                         on=['csr_id'], 
                                         how='left', 
                                         indicator=True)
csr_id_leftover = csr_id_leftover[csr_id_leftover['_merge'] == 'left_only']
csr_id_leftover.drop(columns=['_merge'], inplace=True)

# remove those in B
csr_id_leftover = csr_id_leftover.merge(unique_B, 
                                      on=['csr_id'], 
                                      how='left', 
                                      indicator=True)
csr_id_leftover = csr_id_leftover[csr_id_leftover['_merge'] == 'left_only']
csr_id_leftover.drop(columns=['_merge'], inplace=True)

# create another column where every id is 9999, copy that dataframe and flip
csr_id_leftover['csr_id_dummy'] = 9999
csr_leftover_A = csr_id_leftover.rename(columns={ 'csr_id': 'csr_id_A', 'csr_id_dummy': 'csr_id_B' })
csr_leftover_B = csr_id_leftover.rename(columns={ 'csr_id': 'csr_id_B', 'csr_id_dummy': 'csr_id_A' })

# concatenate all
df_edges = pd.concat([high2low_edges, low2high_edges, csr_leftover_A, csr_leftover_B], axis='index')
df_edges.fillna(9999, inplace=True)

# construct matrix
df_adjacency_mat = pd.crosstab(df_edges.csr_id_A, df_edges.csr_id_B)
df_adjacency_mat.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat.drop(columns=9999.0, inplace=True)
df_adjacency_mat.index.rename('csr_id', inplace=True)

df_01_adjacency_mat = pd.DataFrame(np.where(df_adjacency_mat > 0, 1, 0))



#%% construct number-of-edges matrix

# construct adjacency matrix with distances (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['num_of_edge'] = 0
leftover_B['num_of_edge'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_count = pd.concat([df_csr_edges, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_count_sym = df_edges_count.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_count = pd.concat([df_edges_count, df_edges_count_sym], axis='index')

# construct matrix (we can use np.mean because there's only 1)
df_adjacency_mat_count = pd.crosstab(df_edges_count.csr_id_A, df_edges_count.csr_id_B, values=df_edges_count.num_of_edge, aggfunc=np.mean)
df_adjacency_mat_count.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_count.drop(columns=9999.0, inplace=True)
df_adjacency_mat_count.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_count.drop(columns=0, inplace=True)
df_adjacency_mat_count = df_adjacency_mat_count.fillna(0)



#%% construct distance matrix

# construct adjacency matrix with distances (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['distance_km'] = 0
leftover_B['distance_km'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_dist = pd.concat([df_csr_edges, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_dist_sym = df_edges_dist.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_dist = pd.concat([df_edges_dist, df_edges_dist_sym], axis='index')

# construct matrix
df_adjacency_mat_dist = pd.crosstab(df_edges_dist.csr_id_A, df_edges_dist.csr_id_B, values=df_edges_dist.distance_km, aggfunc=np.mean)
df_adjacency_mat_dist.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_dist.drop(columns=9999.0, inplace=True)
df_adjacency_mat_dist.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_dist.drop(columns=0, inplace=True)



#%% construct voltage matrix

# construct adjacency matrix with capacities (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['voltage_kv'] = 0
leftover_B['voltage_kv'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_volt = pd.concat([df_csr_edges, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_volt_sym = df_edges_volt.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_volt = pd.concat([df_edges_volt, df_edges_volt_sym], axis='index')

# construct matrix
df_adjacency_mat_volt = pd.crosstab(df_edges_volt.csr_id_A, df_edges_volt.csr_id_B, values=df_edges_volt.voltage_kv, aggfunc=np.sum)
df_adjacency_mat_volt.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_volt.drop(columns=9999.0, inplace=True)
df_adjacency_mat_volt.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_volt.drop(columns=0, inplace=True)




#%% construct line capacity matrix

# construct adjacency matrix with capacities (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['max_capacity_mw'] = 0
leftover_B['max_capacity_mw'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_cap = pd.concat([df_csr_edges, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_cap_sym = df_edges_cap.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_cap = pd.concat([df_edges_cap, df_edges_cap_sym], axis='index')

# construct matrix
df_adjacency_mat_cap = pd.crosstab(df_edges_cap.csr_id_A, df_edges_cap.csr_id_B, values=df_edges_cap.max_capacity_mw, aggfunc=np.sum)
df_adjacency_mat_cap.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_cap.drop(columns=9999.0, inplace=True)
df_adjacency_mat_cap.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_cap.drop(columns=0, inplace=True)


# =============================================================================
# # TEST: entire matrix comparison
# df_mat_cap_0 = df_adjacency_mat_cap.fillna(0)
# df_mat_cap_0[df_mat_cap_0 != 0] = 1
# df_mat_0 = df_adjacency_mat
# df_mat_0[df_mat_0 != 0] = 1
# df_mat_0 = df_mat_0.astype(float)
# df_mat_0.equals(df_mat_cap_0)
# # returns false for some reason
# 
# # TEST: cell by cell comparison
# size = len(df_mat_0)
# df_mat_dist_0 = df_adjacency_mat_dist.fillna(0)
# df_mat_dist_0[df_mat_dist_0 != 0] = 1
# 
# # errors will occur if and only if there are line capacities that are 0
# error_tuples = []
# for i in range(1,size+1):
#     for j in range(1,size+1):
#         if df_mat_cap_0.loc[i,j] != df_mat_dist_0.loc[i,j]:
#             print('Row:', i)
#             print('Column:', j)
#             error_tuples.append((i,j))
# 
# # remove any entries from the aggregate lines dataframe and adjacency matrices where the capacity is 0
# for i,j in error_tuples:
#     df_adjacency_mat.loc[i,j] = 0
#     df_adjacency_mat_dist.loc[i,j] = np.nan
#     df_adjacency_mat_count.loc[i,j] = 0
#     # should only delete one because csr_edges has already sorted them with the lowest id first
#     df_csr_edges = df_csr_edges[(df_csr_edges['csr_id_A'] != i) | (df_csr_edges['csr_id_B'] != j)]
# =============================================================================
    




#%% plant capacity aggregation

# load data
df_gpp = pd.read_csv(os.path.join(data_path, "gpp", "gpp_bleed.csv"))

# sort types of energy sources (define hydro as fossil per Connor's request)
df_gpp_renw = df_gpp[(df_gpp['primary_fuel'] == 'Solar') |
                     (df_gpp['primary_fuel'] == 'Wind') |
                     (df_gpp['primary_fuel'] == 'Biomass') |
                     (df_gpp['primary_fuel'] == 'Waste') |
                     (df_gpp['primary_fuel'] == 'Geothermal')]
df_gpp_fossil = df_gpp[(df_gpp['primary_fuel'] == 'Gas') |
                      (df_gpp['primary_fuel'] == 'Oil') |
                      (df_gpp['primary_fuel'] == 'Coal') |
                      (df_gpp['primary_fuel'] == 'Hydro') ]


# create geodataframe
df_gpp['coordinates'] = [Point(xy) for xy in zip(df_gpp['longitude'], df_gpp['latitude'])]
df_gpp = gpd.GeoDataFrame(df_gpp, geometry = 'coordinates')

# label each power plant with its commuter zone (by finding intersections)
csr_shp = csr_shp.to_crs("EPSG:4326")
df_gpp = df_gpp.set_crs("EPSG:4326")
df_gpp = gpd.sjoin(df_gpp, csr_shp[['csr_id', 'geometry']], how='left', predicate='intersects')

# drop those with no csr_id because they're in a country outside of the shapefile
df_gpp = df_gpp[df_gpp['csr_id'].notna()]

# create aggregate capacity dataframe
df_agg_capacity = df_gpp.groupby(['csr_id']).sum('capacity_mw')
df_agg_capacity = df_agg_capacity[['capacity_mw']]


# find commuter zone intersections and aggregate capacity for renewables
df_gpp_renw['coordinates'] = [Point(xy) for xy in zip(df_gpp_renw['longitude'], df_gpp_renw['latitude'])]
df_gpp_renw = gpd.GeoDataFrame(df_gpp_renw, geometry = 'coordinates')
df_gpp_renw = df_gpp_renw.set_crs("EPSG:4326")
df_gpp_renw = gpd.sjoin(df_gpp_renw, csr_shp[['csr_id', 'geometry']], how='left', predicate='intersects')

df_agg_renw = df_gpp_renw.groupby(['csr_id']).sum('capacity_mw')
df_agg_renw = df_agg_renw[['capacity_mw']]
df_agg_renw = df_agg_renw.rename(columns={ 'capacity_mw': 'rnw_capacity_mw' })


# same for fossil fuels
df_gpp_fossil['coordinates'] = [Point(xy) for xy in zip(df_gpp_fossil['longitude'], df_gpp_fossil['latitude'])]
df_gpp_fossil = gpd.GeoDataFrame(df_gpp_fossil, geometry = 'coordinates')
df_gpp_fossil = df_gpp_fossil.set_crs("EPSG:4326")
df_gpp_fossil = gpd.sjoin(df_gpp_fossil, csr_shp[['csr_id', 'geometry']], how='left', predicate='intersects')

df_agg_fossil = df_gpp_fossil.groupby(['csr_id']).sum('capacity_mw')
df_agg_fossil = df_agg_fossil[['capacity_mw']]
df_agg_fossil = df_agg_fossil.rename(columns={ 'capacity_mw': 'ffl_capacity_mw' })


# get dataframe of just commuter zone ids
csr_index = csr_shp[['csr_id']]
csr_index = csr_index.set_index(keys=['csr_id'])

# combine all aggregate capacity data
df_aggcap_all = pd.concat([csr_index, df_agg_capacity, df_agg_renw, df_agg_fossil], 
                          axis=1, 
                          join='outer', 
                          ignore_index=False)

# fill nan values with 0
df_aggcap_all.fillna(0, inplace=True)




#%% counterfactual setup

# map reg_ids to csr_ids
id_map = csr_shp[['reg_id', 'csr_id']]

def csr(id_map, reg_id):
    csr = int(id_map[id_map['reg_id']==reg_id]['csr_id'])
    return csr


# transwest
transwest = pd.DataFrame([[csr(id_map, 'US34602'), # wymoning to utah 
                           csr(id_map, 'US36000'),
                           1,
                           500,
                           1500,
                           np.nan],
                          [csr(id_map, 'US36000'), # utah to nevada
                           csr(id_map, 'US37901'),
                           1,
                           500,
                           3000,
                           np.nan]
                          ], columns=df_csr_edges.columns)



# sunzia
sunzia = pd.DataFrame([[csr(id_map, 'US35001'), # pinal AZ to graham AZ
                           csr(id_map, 'US35002'),
                            1,
                            525,
                            3000,
                            np.nan],
                           [csr(id_map, 'US35002'), #  graham AZ to hidalgo NM
                            csr(id_map, 'US30604'),
                            1,
                            525,
                            3000,
                            np.nan],
                           [csr(id_map, 'US30604'), #  hidalgo NM to socorro NM
                            csr(id_map, 'US34902'),
                            1,
                            525,
                            3000,
                            np.nan],
                           [csr(id_map, 'US34902'), #  socorro NM to lincoln NM
                            csr(id_map, 'US30602'),
                            1,
                            525,
                            3000,
                            np.nan]
                      ], columns=df_csr_edges.columns)

# grainbelt
grainbelt = pd.DataFrame([[csr(id_map, 'US29003'), # ford KS to monroe MS
                            csr(id_map, 'US26101'),
                            1,
                            600,
                            5000,
                            np.nan],
                           [csr(id_map, 'US26101'), #  monroe MS to callaway MS (Kingdom City end)
                            csr(id_map, 'US29601'),
                            1,
                            600,
                            2500,
                            np.nan],
                           [csr(id_map, 'US26101'), #  monroe MS to ralls MS
                            csr(id_map, 'US25000'),
                            1,
                            600,
                            5000,
                            np.nan],
                           [csr(id_map, 'US25000'), #  ralls MS to clark IL
                            csr(id_map, 'US23301'),
                            1,
                            600,
                            5000,
                            np.nan],
                           [csr(id_map, 'US23301'), #  clark IL to sullivan IN
                            csr(id_map, 'US14400'),
                            1,
                            600,
                            5000,
                            np.nan]
                      ], columns=df_csr_edges.columns)

# champlain-hudson
champlain = pd.DataFrame([[csr(id_map, 'CA24'), # quebec to upstate NY 
                           csr(id_map, 'US18600'),
                           1,
                           320,
                           1250,
                           np.nan],
                          [csr(id_map, 'US18600'), # upstate to downstate
                           csr(id_map, 'US19600'),
                           1,
                           320,
                           1250,
                           np.nan],
                          [csr(id_map, 'US19600'), #  downstate to NYC
                           csr(id_map, 'US19400'),
                           1,
                           320,
                           1250,
                           np.nan]
                          ], columns=df_csr_edges.columns)

# combine
df_grid_improvements = pd.concat([transwest, sunzia, grainbelt, champlain], axis=0)

# calculate distances
# merge by zone id
df_grid_improvements = df_grid_improvements.merge(m_centroids, how='left', left_on='csr_id_A', right_on='csr_id')
df_grid_improvements.drop(columns='csr_id', inplace=True)
df_grid_improvements = df_grid_improvements.merge(m_centroids, how='left', left_on='csr_id_B', right_on='csr_id')
df_grid_improvements.drop(columns='csr_id', inplace=True)

# stripping geoms into lat and lon for geopy distance function
df_grid_improvements['latA'] = df_grid_improvements['centroid_x'].apply(lambda a: a.y) 
df_grid_improvements['lonA'] = df_grid_improvements['centroid_x'].apply(lambda a: a.x) 
df_grid_improvements['latB'] = df_grid_improvements['centroid_y'].apply(lambda a: a.y) 
df_grid_improvements['lonB'] = df_grid_improvements['centroid_y'].apply(lambda a: a.x) 

# turning into tuples
df_grid_improvements['tupleA'] = list(zip(df_grid_improvements.latA, df_grid_improvements.lonA))
df_grid_improvements['tupleB'] = list(zip(df_grid_improvements.latB, df_grid_improvements.lonB))


# calculate distance
df_grid_improvements['distance_km'] = df_grid_improvements[['tupleA', 'tupleB']].apply(lambda x: distance(x['tupleA'], x['tupleB']).km, axis=1)

# trim dataframe
df_grid_improvements = df_grid_improvements[df_csr_edges.columns]


# check which id pairs are already in df_csr_edges
match_list = []

for i in range(len(df_grid_improvements)):
    row = df_grid_improvements.iloc[[i]]
    matches = df_csr_edges[((df_csr_edges['csr_id_A'] == int(row['csr_id_A'])) & 
                            (df_csr_edges['csr_id_B'] == int(row['csr_id_B']))   ) | 
                           ((df_csr_edges['csr_id_A'] == int(row['csr_id_B'])) & 
                            (df_csr_edges['csr_id_B'] == int(row['csr_id_A'])   ))]
    if matches.empty:
        match_list.append(0)
    else:
        match_list.append(1)

# record whether the edges already exist in the data
df_grid_improvements = pd.concat([df_grid_improvements, pd.DataFrame(match_list)], axis=1)
df_grid_improvements.rename(columns={0:'match'}, inplace=True)

# concatenate edges that don't exist
unmatched = df_grid_improvements[df_grid_improvements['match'] == 0]
unmatched.drop(columns='match', inplace=True)
df_csr_edges_imp = pd.concat([df_csr_edges, unmatched])

# for edges that already exist, sum num_edges, max_capacity and voltages
matched  = df_grid_improvements[df_grid_improvements['match'] == 1]
matched.reset_index(inplace=True)
matched.drop(columns=['index', 'match'], inplace=True)

# initialize empty dataframe to collect matches
repeat_edges = pd.DataFrame(columns=matched.columns)

for i in range(len(matched)):
    row = matched.iloc[[i]]
    repeat = df_csr_edges[((df_csr_edges['csr_id_A'] == int(row['csr_id_A'])) & 
                           (df_csr_edges['csr_id_B'] == int(row['csr_id_B']))   ) | 
                          ((df_csr_edges['csr_id_A'] == int(row['csr_id_B'])) & 
                           (df_csr_edges['csr_id_B'] == int(row['csr_id_A'])   ))]
    
    # delete that line from the original data, to be updated
    df_csr_edges_imp.drop(index=repeat.index, inplace=True)
    
    # add capacity
    repeat[['max_capacity_mw']] = float(repeat['max_capacity_mw']) + float(row['max_capacity_mw'])
    
    # add number of edges
    repeat[['num_of_edge']] = int(repeat['num_of_edge']) + int(row['num_of_edge'])
    
    # add voltage
    repeat[['voltage_kv']] = float(repeat['voltage_kv']) + float(row['voltage_kv'])
    
    # append the updated edge
    repeat_edges = pd.concat([repeat_edges, repeat], axis=0)
    
# add the updated edges
df_csr_edges_imp = pd.concat([df_csr_edges_imp, repeat_edges], axis=0)

# plot the regions connected by these lines as a sanity check
improved_reglist = list(df_grid_improvements['csr_id_A'])
improved_reglist = improved_reglist + list(df_grid_improvements['csr_id_B'])
improved_reglist = [*set(improved_reglist)]

improved_shp = csr_shp[csr_shp['csr_id'].isin(improved_reglist)]
improved_shp.plot()





#%% construct improved number of edge matrix

# construct adjacency matrix with distances (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges_imp.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges_imp.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['num_of_edge'] = 0
leftover_B['num_of_edge'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_count = pd.concat([df_csr_edges_imp, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_count_sym = df_edges_count.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_count = pd.concat([df_edges_count, df_edges_count_sym], axis='index')

# construct matrix (we can use np.mean because there's only 1)
df_adjacency_mat_count_imp = pd.crosstab(df_edges_count.csr_id_A, df_edges_count.csr_id_B, values=df_edges_count.num_of_edge, aggfunc=np.mean)
df_adjacency_mat_count_imp.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_count_imp.drop(columns=9999.0, inplace=True)
df_adjacency_mat_count_imp.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_count_imp.drop(columns=0, inplace=True)
df_adjacency_mat_count_imp = df_adjacency_mat_count_imp.fillna(0)




#%% construct improved distance matrix

# construct adjacency matrix with distances (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges_imp.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges_imp.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['distance_km'] = 0
leftover_B['distance_km'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_dist = pd.concat([df_csr_edges_imp, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_dist_sym = df_edges_dist.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_dist = pd.concat([df_edges_dist, df_edges_dist_sym], axis='index')

# construct matrix
df_adjacency_mat_dist_imp = pd.crosstab(df_edges_dist.csr_id_A, df_edges_dist.csr_id_B, values=df_edges_dist.distance_km, aggfunc=np.mean)
df_adjacency_mat_dist_imp.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_dist_imp.drop(columns=9999.0, inplace=True)
df_adjacency_mat_dist_imp.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_dist_imp.drop(columns=0, inplace=True)




#%% construct improved voltage matrix

# construct adjacency matrix with capacities (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges_imp.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges_imp.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['voltage_kv'] = 0
leftover_B['voltage_kv'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_volt = pd.concat([df_csr_edges_imp, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_volt_sym = df_edges_volt.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_volt = pd.concat([df_edges_volt, df_edges_volt_sym], axis='index')

# construct matrix
df_adjacency_mat_volt_imp = pd.crosstab(df_edges_volt.csr_id_A, df_edges_volt.csr_id_B, values=df_edges_volt.voltage_kv, aggfunc=np.sum)
df_adjacency_mat_volt_imp.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_volt_imp.drop(columns=9999.0, inplace=True)
df_adjacency_mat_volt_imp.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_volt_imp.drop(columns=0, inplace=True)






#%% construct improved line capacity matrix

# construct adjacency matrix with capacities (start by finding the leftovers)
leftover_A = []
leftover_B = []
for i in range(len(csr_shp)):
    if i not in list(df_csr_edges_imp.csr_id_A):
        leftover_A.append(i)
    if i not in list(df_csr_edges_imp.csr_id_B):
        leftover_B.append(i)

# create dummy connections
leftover_A = pd.DataFrame(leftover_A)
leftover_B = pd.DataFrame(leftover_B)
leftover_A['csr_id_B'] = 9999
leftover_B['csr_id_A'] = 9999
leftover_A['max_capacity_mw'] = 0
leftover_B['max_capacity_mw'] = 0
leftover_A.rename(columns={ 0: 'csr_id_A' }, inplace=True)
leftover_B.rename(columns={ 0: 'csr_id_B' }, inplace=True)

# concatenate
df_edges_cap = pd.concat([df_csr_edges_imp, leftover_A, leftover_B], axis='index')

# make the matrix symmetric (doesn't do this automatically when you specify values + aggfunc)
df_edges_cap_sym = df_edges_cap.rename(columns={ 'csr_id_A': 'csr_id_B', 'csr_id_B': 'csr_id_A' })
df_edges_cap = pd.concat([df_edges_cap, df_edges_cap_sym], axis='index')

# construct matrix
df_adjacency_mat_cap_imp = pd.crosstab(df_edges_cap.csr_id_A, df_edges_cap.csr_id_B, values=df_edges_cap.max_capacity_mw, aggfunc=np.sum)
df_adjacency_mat_cap_imp.drop(labels=9999.0, axis=0, inplace=True)
df_adjacency_mat_cap_imp.drop(columns=9999.0, inplace=True)
df_adjacency_mat_cap_imp.drop(labels=0, axis=0, inplace=True)
df_adjacency_mat_cap_imp.drop(columns=0, inplace=True)



#%% construct basic improved adjacency matrix

df_adjacency_mat_imp = df_adjacency_mat_count_imp
df_adjacency_mat_imp[df_adjacency_mat_imp > 0] = 1


#%% export

# edges between zones with voltages and distances
df_csr_edges.to_csv(os.path.join(data_path, "agg_output", "csr_edges.csv"), index=False)

# aggregate power production capacities by zone
df_aggcap_all.to_csv(os.path.join(data_path, "agg_output", "csr_aggcap.csv"))


# adjacency matrix
df_adjacency_mat.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat.csv"))
# edge counts 
df_adjacency_mat_count.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_count.csv"))
# distances
df_adjacency_mat_dist.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_dist.csv"))
# voltages
df_adjacency_mat_volt.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_volt.csv"))
# line capacities
df_adjacency_mat_cap.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_lcap.csv"))





#%% export counterfactuals

# edges between zones with voltages and distances
df_csr_edges_imp.to_csv(os.path.join(data_path, "agg_output", "csr_edges_imp.csv"), index=False)

# only the grid improvements
df_grid_improvements.drop(columns='match', inplace=True)
df_grid_improvements.to_csv(os.path.join(data_path, "agg_output", "csr_edges_imp_only.csv"), index=False)


# adjacency matrix
df_adjacency_mat_imp.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_imp.csv"))
# edge counts 
df_adjacency_mat_count_imp.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_count_imp.csv"))
# distances
df_adjacency_mat_dist_imp.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_dist_imp.csv"))
# voltages
df_adjacency_mat_volt_imp.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_volt_imp.csv"))
# line capacities
df_adjacency_mat_cap_imp.to_csv(os.path.join(data_path, "agg_output", "csr_adjmat_lcap_imp.csv"))


# reset warnings
warnings.filterwarnings('default')


