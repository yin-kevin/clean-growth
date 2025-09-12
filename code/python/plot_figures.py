# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:23:58 2022

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
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.lines import Line2D
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
finally:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.lines import Line2D

# Shapely 2.0.1
try:
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import split
    from shapely.affinity import translate
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'shapely'])
finally:
    from shapely.geometry import Point, LineString, Polygon
    from shapely.ops import split
    from shapely.affinity import translate
    
    


#%% user parameters

# 1. root directory
directory_path = os.path.realpath(__file__)[:-28]
os.chdir(directory_path)

# 2. data path
data_path = os.path.join(directory_path, "data")

# 3. Image output path
output_path = os.path.join(directory_path, "figures")



#%% override option

# if running this script from the master, override the directory path
try:
   directory_path = master_directory_path
   os.chdir(master_directory_path)
   data_path = os.path.join(master_directory_path, "data")
   output_path = os.path.join(master_directory_path, "figures")
except NameError:
    pass



#%% import data

# import sub regions
csr_path = os.path.join(directory_path, "data", "shapefile", "selected_regions", "selected_regions.shp")
csr_shp = gpd.read_file(csr_path)

# import edge data
edge_path = os.path.join(data_path, "agg_output", "csr_edges.csv")
edges = pd.read_csv(edge_path)

# import grid improvements
improvement_path = os.path.join(data_path, "agg_output", "csr_edges_imp_only.csv")
improvements = pd.read_csv(improvement_path )

# import capacity data
capacity_path = os.path.join(data_path, "agg_output", "csr_aggcap.csv")
capacities = pd.read_csv(capacity_path)

# import individual lines
lines_path = os.path.join(data_path, "sql_output", "links_cap.csv")
lines = pd.read_csv(lines_path)

# import all stations
gpp_path = os.path.join(data_path, "gpp", "gpp_bleed.csv")
df_gpp = pd.read_csv(gpp_path)

# import model outputs (potentials, shares etc)
modeldata_path = os.path.join(data_path, "mod_output", "model_dataset.csv")
df_modeldata = pd.read_csv(modeldata_path)

# import Canadian population raster
canada_pop_path = os.path.join(data_path, "shapefile", "canada_population", "griddedPopulationCanada10km_2016.shp")
df_canada_pop =  gpd.read_file(canada_pop_path)

# strip country codes from the reg_ids
csr_shp['ctry_code'] = csr_shp['reg_id'].str[:2]

# sort country codes
china_codes = ['CN']          
us_codes = ['US']
india_codes = ['IN']
russia_codes = ['RU']
brazil_codes = ['BR']
australia_codes = ['AU']
southkorea_codes = ['KR']
colombia_codes = ['CO']
turkey_codes = ['TR']
japan_codes = ['JP']

NA_codes = ['US', # united states
            'CA', # canada
            'ME'] # mexico

EU_codes = ['AT', # austria
            'DE', # germany
            'ES', # spain
            'TR', # turkey
            'CH', # switzerland
            'FR', # france
            'BE', # belgium
            'PT', # portugal
            'NL', # netherlands
            'HR', # croatia
            'IT', # italy
            'FI', # finland
            'CZ', # czech republic
            'DK', # denmark
            'PL', # poland
            'UK', # united kingdom
            'IE', # ireland
            'MT', # malta
            'LT', # lithuania
            'BG', # bulgaria
            'EE', # estonia
            'EL', # greece
            'LV', # latvia
            'SE', # sweden
            'HU', # hungary
            'SK', # slovakia
            'RO', # romania
            'SI', # slovenia
            'NO', # norway
            'LU'] # luxembourg



#%% fix russia

# russia shapefile needs to be adjusted since a small piece is on the other side of the map
russia_shp = csr_shp[csr_shp['ctry_code'].isin(russia_codes)]
chukotka_id = russia_shp.loc[russia_shp.region == 'Chukotka Autonomous Okrug', 'csr_id'].squeeze()
chukotka_idx = russia_shp.index[russia_shp['csr_id']==chukotka_id].to_list()[0]
chukotka_shp = gpd.GeoDataFrame(pd.DataFrame(russia_shp.loc[chukotka_idx,:]).transpose(), geometry = 'geometry')

# function which shifts the center (it also breaks everything into pieces so they need to be recombined)
def shift_geom(shift, gdataframe, plotQ=False):
    # this code is adapted from stackoverflow
    shift -= 180
    moved_map = []
    splitted_map = []
    border = LineString([(shift,90),(shift,-90)])

    for row in gdataframe["geometry"]:
        splitted_map.append(split(row, border))
    for element in splitted_map:
        items = list(element.geoms) # for earlier versions of Shapely (1.7.1), `element.geoms' is just `element'
        for item in items:
            minx, miny, maxx, maxy = item.bounds
            if minx >= shift:
                moved_map.append(translate(item, xoff=-180-shift))
            else:
                moved_map.append(translate(item, xoff=180-shift))

    # got `moved_map` as the moved geometry            
    gdf = gpd.GeoDataFrame({"geometry": moved_map})
    # can move back to original pos by rerun with -ve shift

    # plot the figure
    if plotQ:
        fig, ax = plt.subplots()
        gdf.plot(ax=ax)
        #plt.show()

    return gdf

# shift the region back and forth, which combines the two separate parts
chukotka_shp = shift_geom(90, chukotka_shp, False)
chukotka_shp = shift_geom(-90, chukotka_shp, False)

# combine all the parts again
chukotka_shp = chukotka_shp.dissolve()

# a very involved method to combine the geometry and the rest of the data
chukotka_shp['index'] = 655
chukotka_shp.set_index('index', inplace=True)

# combine chukotka region data with reconstructed shapefile
df_chukotka = russia_shp.loc[russia_shp.csr_id==chukotka_id, russia_shp.columns!='geometry']
df_chukotka = pd.concat([df_chukotka, chukotka_shp], axis=1)

# delete old chukotka and replace with new
csr_shp = csr_shp[csr_shp.csr_id != chukotka_id]
csr_shp = pd.concat([csr_shp, df_chukotka], axis=0)



#%% population-weighted Canada centroids

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

# merge with canada_shp
canada_shp = canada_shp.merge(w_centroids, how='left', on='csr_id')
canada_shp = canada_shp.set_crs(csr_shp.crs)



#%% pre-processing

# ---------- EDGES ----------
# calculate centroids (and move Unalaska centroid to where it should be)
unalaska_shp = csr_shp[csr_shp['region'] == 'Unalaska']
csr_shp = csr_shp[(csr_shp['ctry_code'] != 'CA') & (csr_shp['region'] != 'Unalaska')]
csr_shp['centroid'] = csr_shp.centroid
csr_shp = pd.concat([csr_shp, canada_shp], axis=0)

# hard code Unalaska centroid
unalaska_shp['centroid'] = Point(-166.725, 53.8)
unalaska_shp = gpd.GeoDataFrame(unalaska_shp, geometry='centroid')
unalaska_shp = unalaska_shp.set_crs(csr_shp.crs)
csr_shp = pd.concat([csr_shp, unalaska_shp], axis=0)
csr_shp = gpd.GeoDataFrame(csr_shp, geometry='centroid')
csr_shp = csr_shp.set_crs("EPSG:4326")
csr_shp = gpd.GeoDataFrame(csr_shp, geometry='geometry')

# find the centroids that are only connected to the grid improvements
imp_shp = csr_shp[csr_shp['csr_id'].isin(improvements['csr_id_A']) | csr_shp['csr_id'].isin(improvements['csr_id_B'])]

# match centroid to zone id A
edges = edges.merge(csr_shp[['csr_id','centroid']], 
                    left_on=['csr_id_A'], 
                    right_on=['csr_id'],
                    how='left', 
                    indicator=True)

# same for grid improvements
improvements = improvements.merge(csr_shp[['csr_id','centroid']], 
                                  left_on=['csr_id_A'], 
                                  right_on=['csr_id'],
                                  how='left', 
                                  indicator=True)

# clean up
edges = edges.rename(columns={ 'centroid': 'centroid_A' })
edges = edges.drop(columns=['csr_id', '_merge'])

improvements = improvements.rename(columns={ 'centroid': 'centroid_A' })
improvements = improvements.drop(columns=['csr_id', '_merge'])

# match centroid to zone id B
edges = edges.merge(csr_shp[['csr_id','centroid']], 
                    left_on=['csr_id_B'], 
                    right_on=['csr_id'],
                    how='left', 
                    indicator=True)

# same for grid improvements
improvements = improvements.merge(csr_shp[['csr_id','centroid']], 
                                  left_on=['csr_id_B'], 
                                  right_on=['csr_id'],
                                  how='left', 
                                  indicator=True)

# clean up
edges = edges.rename(columns={ 'centroid': 'centroid_B' })
edges = edges.drop(columns=['csr_id', '_merge'])

improvements = improvements.rename(columns={ 'centroid': 'centroid_B' })
improvements = improvements.drop(columns=['csr_id', '_merge'])

# TEMP: drop edges that have 'none' in one of the centroids
edges = edges.dropna()
edges['line'] = edges.apply(lambda x: LineString([x['centroid_A'], x['centroid_B']]), axis = 1)
edges = gpd.GeoDataFrame(edges, geometry = 'line')

improvements = improvements.dropna()
improvements['line'] = improvements.apply(lambda x: LineString([x['centroid_A'], x['centroid_B']]), axis = 1)
improvements = gpd.GeoDataFrame(improvements, geometry = 'line')

improvements_save = improvements


# ---------- CAPACITIES ----------
na_eu_shp = csr_shp[(csr_shp['ctry_code'].isin(NA_codes)) | (csr_shp['ctry_code'].isin(EU_codes))]
na_eu_shp = gpd.clip(na_eu_shp, Polygon([[-175, 10], [50, 10], [50, 85], [-175, 85]]))

northamer_shp = csr_shp[csr_shp['ctry_code'].isin(NA_codes)]
northamer_shp = gpd.clip(northamer_shp, Polygon([[-175, 10], [50, 10], [50, 85], [-175, 85]]))

europe_shp = csr_shp[csr_shp['ctry_code'].isin(EU_codes)]
europe_shp = gpd.clip(europe_shp, Polygon([[-34, 35], [50, 35], [50, 75], [-34, 75]]))

us_shp = csr_shp[csr_shp['ctry_code'].isin(us_codes)]
us_shp = gpd.clip(us_shp, Polygon([[-130, 10], [50, 10], [50, 52], [-130, 52]]))

china_shp = csr_shp[csr_shp['ctry_code'].isin(china_codes)]
india_shp = csr_shp[csr_shp['ctry_code'].isin(india_codes)]
russia_shp = csr_shp[csr_shp['ctry_code'].isin(russia_codes)]
brazil_shp = csr_shp[csr_shp['ctry_code'].isin(brazil_codes)]
australia_shp = csr_shp[csr_shp['ctry_code'].isin(australia_codes)]
southkorea_shp = csr_shp[csr_shp['ctry_code'].isin(southkorea_codes)]
japan_shp = csr_shp[csr_shp['ctry_code'].isin(japan_codes)]
colombia_shp = csr_shp[csr_shp['ctry_code'].isin(colombia_codes)]
turkey_shp = csr_shp[csr_shp['ctry_code'].isin(turkey_codes)]

# merge centroids with capacity data
us_capacities = capacities.merge(us_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
us_capacities = us_capacities[us_capacities['centroid'].notnull()]

#  add the connected Canada and Mexico capacities to US
us_ids = list(us_shp['csr_id'])
us_edges = edges[(edges['csr_id_A'].isin(us_ids)) | (edges['csr_id_B'].isin(us_ids))]
external = us_edges[(~us_edges['csr_id_A'].isin(us_ids)) | (~us_edges['csr_id_B'].isin(us_ids)) ]
external = pd.concat([external[['csr_id_A']].rename(columns={'csr_id_A':'csr_id'}), 
                      external[['csr_id_B']].rename(columns={'csr_id_B':'csr_id'})], axis=0)
external = external[~external['csr_id'].isin(us_ids)]
external.drop_duplicates(inplace=True)

ex_capacities = csr_shp[['csr_id','centroid']].merge(external, 
                                                     how='right', 
                                                     on='csr_id')
ex_capacities = capacities.merge(ex_capacities,
                                 how='right',
                                 on='csr_id')
us_capacities = pd.concat([us_capacities, ex_capacities], axis=0)
us_capacities = gpd.GeoDataFrame(us_capacities, geometry='centroid')

# merge centroids-capacities for other regions
na_eu_capacities = capacities.merge(na_eu_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
na_eu_capacities = na_eu_capacities[na_eu_capacities['centroid'].notnull()]

na_capacities = capacities.merge(northamer_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
na_capacities = na_capacities[na_capacities['centroid'].notnull()]

eu_capacities = capacities.merge(europe_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
eu_capacities = eu_capacities[eu_capacities['centroid'].notnull()]

cn_capacities = capacities.merge(china_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
cn_capacities = cn_capacities[cn_capacities['centroid'].notnull()]


# capacities for grid improvements only (US)
imp_capacities = capacities.merge(imp_shp[['csr_id', 'centroid']], 
                                  how='left', 
                                  left_on='csr_id', 
                                  right_on='csr_id')
imp_capacities = imp_capacities[imp_capacities['centroid'].notnull()]

in_capacities = capacities.merge(india_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
in_capacities = in_capacities[in_capacities['centroid'].notnull()]

ru_capacities = capacities.merge(russia_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
ru_capacities = ru_capacities[ru_capacities['centroid'].notnull()]

br_capacities = capacities.merge(brazil_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
br_capacities = br_capacities[br_capacities['centroid'].notnull()]

au_capacities = capacities.merge(australia_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
au_capacities = au_capacities[au_capacities['centroid'].notnull()]

kr_capacities = capacities.merge(southkorea_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
kr_capacities = kr_capacities[kr_capacities['centroid'].notnull()]

jp_capacities = capacities.merge(japan_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
jp_capacities = jp_capacities[jp_capacities['centroid'].notnull()]

co_capacities = capacities.merge(colombia_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
co_capacities = co_capacities[co_capacities['centroid'].notnull()]

tr_capacities = capacities.merge(turkey_shp[['csr_id', 'centroid']], 
                                 how='left', 
                                 left_on='csr_id', 
                                 right_on='csr_id')
tr_capacities = tr_capacities[tr_capacities['centroid'].notnull()]




# ensure capacities is a geo_df
na_eu_capacities = gpd.GeoDataFrame(na_eu_capacities, geometry='centroid')
na_capacities = gpd.GeoDataFrame(na_capacities, geometry='centroid')
eu_capacities = gpd.GeoDataFrame(eu_capacities, geometry='centroid')
cn_capacities = gpd.GeoDataFrame(cn_capacities, geometry='centroid')
us_capacities = gpd.GeoDataFrame(us_capacities, geometry='centroid')
imp_capacities = gpd.GeoDataFrame(imp_capacities, geometry='centroid')
in_capacities = gpd.GeoDataFrame(in_capacities, geometry='centroid')
ru_capacities = gpd.GeoDataFrame(ru_capacities, geometry='centroid')
br_capacities = gpd.GeoDataFrame(br_capacities, geometry='centroid')
au_capacities = gpd.GeoDataFrame(au_capacities, geometry='centroid')
co_capacities = gpd.GeoDataFrame(co_capacities, geometry='centroid')
tr_capacities = gpd.GeoDataFrame(tr_capacities, geometry='centroid')
jp_capacities = gpd.GeoDataFrame(jp_capacities, geometry='centroid')
kr_capacities = gpd.GeoDataFrame(kr_capacities, geometry='centroid')
tr_capacities = gpd.GeoDataFrame(tr_capacities, geometry='centroid')


# clip the capacities
na_capacities = gpd.clip(na_capacities, Polygon([[-175, 10], [50, 10], [50, 85], [-175, 85]]))
eu_capacities = gpd.clip(eu_capacities, Polygon([[-34, 35], [50, 35], [50, 75], [-34, 75]]))


# ---------- LINES ----------
# create linestrings
lines['point_A'] = lines.apply(lambda x: Point(x['longitude_1'], x['latitude_1']), axis = 1)
lines['point_B'] = lines.apply(lambda x: Point(x['longitude_2'], x['latitude_2']), axis = 1)
lines['line'] = lines.apply(lambda x: LineString([x['point_A'], x['point_B']]), axis = 1)

# make into geo_df and re-project crs
lines = gpd.GeoDataFrame(lines, geometry='line')
lines = lines.set_crs("EPSG:4326")




#%% us, plot all stations

# create geodataframe
df_gpp['coordinates'] = [Point(xy) for xy in zip(df_gpp['longitude'], df_gpp['latitude'])]
df_gpp = gpd.GeoDataFrame(df_gpp, geometry = 'coordinates')

# label each power plant with its commuter zone (by finding intersections)
us_shp = us_shp.to_crs("EPSG:4326")
df_gpp = df_gpp.set_crs("EPSG:4326")
df_gpp = gpd.sjoin(df_gpp, us_shp[['csr_id', 'geometry']], how='left', predicate='intersects')

df_us_plants = df_gpp[df_gpp['csr_id'].notnull()]
df_us_plants = gpd.GeoDataFrame(df_us_plants, geometry = 'coordinates')

# plot 
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

us_shp.plot(ax=ax,
                    figsize = (20, 20), 
                    color = "lightskyblue", 
                    edgecolor = "white", 
                    linewidth = 0.5,
                    zorder = 1)


df_us_plants.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "palegoldenrod", 
                   edgecolor = "black",
                   markersize = 5 + df_us_plants.capacity_mw/10,
                   linewidth = 1,
                   zorder = 3)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_all_stations.png"), dpi=250, bbox_inches='tight')



#%% plot edges

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
csr_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.1,
            zorder = 1)


# centroid to centroid edges
edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.1 + edges.num_of_edge/50,
           zorder = 2)

# centroids
csr_shp['centroid'].plot(ax=ax,
                    figsize = (20, 20), 
                    color = "yellow", 
                    edgecolor = "black",
                    markersize = 0.75,
                    linewidth = 0.05,
                    zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "global_map_edges.png"), dpi=250, bbox_inches='tight')



#%% plot edges, china

# select the edges that are only in China
china_ids = list(china_shp['csr_id'])
china_edges = edges[(edges['csr_id_A'].isin(china_ids)) & (edges['csr_id_B'].isin(china_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
               figsize = (20, 20), 
               color = "lightskyblue", 
               edgecolor = "white", 
               linewidth = 0.5,
               zorder = 1)

# centroid to centroid edges
china_edges.plot(ax=ax, 
                 color='black', 
                 figsize = (20, 20),
                 # line width varies with number of edges
               linewidth = 1 + china_edges.num_of_edge/10,
               zorder = 2)

# centroids
china_shp['centroid'].plot(ax=ax,
                           figsize = (20, 20), 
                           color = "yellow", 
                           edgecolor = "black",
                           markersize = 100,
                           linewidth = 0.75,
                           zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_edges.png"), dpi=250, bbox_inches='tight')



#%% us: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.35,
                   zorder = 1)

# capacities
us_capacities_save = us_capacities
us_capacities = us_capacities[~us_capacities['csr_id'].isin(ex_capacities['csr_id'])]
us_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + us_capacities.capacity_mw/20,
                               linewidth = 1,
                               zorder = 2)
us_capacities = us_capacities_save

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_map_cap.png"), dpi=200, bbox_inches='tight')



#%% us: plot edges and capacities

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
us_edges.plot(ax=ax, 
              color='black', 
              figsize = (20, 20),
              # line width varies with number of edges
              linewidth = 0.2 + us_edges.num_of_edge/10,
              zorder = 2)

# capacities
us_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + us_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_map_edges_cap.png"), dpi=250, bbox_inches='tight')




#%% us: plot line capacities

# select the edges that are only in the US
us_ids = list(us_shp['csr_id'])
us_edges = edges[(edges['csr_id_A'].isin(us_ids)) | (edges['csr_id_B'].isin(us_ids))]

# cap capacities at 5000 for bin construction
us_edges_save = us_edges
us_edges['max_capacity_mw'].where(us_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = us_edges.columns
us_edges = pd.concat([us_edges, dummy], axis=0)
us_edges['capacity_bin'] = pd.cut(us_edges['max_capacity_mw'], 10, labels=False)
us_edges['capacity_bin'] = us_edges['capacity_bin'] + 1
us_edges['decile_label'] = us_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(us_edges['max_capacity_mw'])
cmax = np.max(us_edges['max_capacity_mw'])
us_edges = us_edges[us_edges['csr_id_A'] != 0]
us_edges = gpd.GeoDataFrame(us_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in us_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
us_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.45},
              #line width varies with capacity
              linewidth = 0.5 + us_edges.max_capacity_mw/2500,
              zorder = 2)



us_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 10,
                   linewidth = 0.8,
                   zorder = 3)


# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_map_edges_guide.png"), dpi=250, bbox_inches='tight')

# restore edges dataframe
us_edges = us_edges_save


#%% us: plot line capacities improvements

# select the edges that are only in the US
us_ids = list(us_shp['csr_id'])

# cap capacities at 5000 for bin construction
improvements = improvements_save
improvements['max_capacity_mw'].where(improvements['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = improvements.columns
improvements = pd.concat([improvements, dummy], axis=0)
improvements['capacity_bin'] = pd.cut(improvements['max_capacity_mw'], 10, labels=False)
improvements['capacity_bin'] = improvements['capacity_bin'] + 1
improvements['decile_label'] = improvements['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(improvements['max_capacity_mw'])
cmax = np.max(improvements['max_capacity_mw'])
improvements = improvements[improvements['csr_id_A'] != 0]
improvements = gpd.GeoDataFrame(improvements, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in improvements['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
improvements.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.45},
              #line width varies with capacity
              linewidth = 0.5 + improvements.max_capacity_mw/2500,
              zorder = 2)


imp_shp['centroid'].plot(ax=ax,
                    figsize = (20, 20), 
                    color = "black", 
                    edgecolor = "black",
                    markersize = 10,
                    linewidth = 0.8,
                    zorder = 3)


# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)
cbar.set_title('Capacity Change', size=16, pad=25)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_map_edges_imp.png"), dpi=250, bbox_inches='tight')




#%% us: plot line AND centroid capacities

# get geographic dimensions of the map
w = us_shp.total_bounds[2] - us_shp.total_bounds[0]
l = us_shp.total_bounds[3] - us_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(w,l) ** 2)

# select the edges that are only in the US
us_ids = list(us_shp['csr_id'])

# cap capacities at 5000 for bin construction
us_edges_save = us_edges
us_edges['max_capacity_mw'].where(us_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = us_edges.columns
us_edges = pd.concat([us_edges, dummy], axis=0)
us_edges['capacity_bin'] = pd.cut(us_edges['max_capacity_mw'], 10, labels=False)
us_edges['capacity_bin'] = us_edges['capacity_bin'] + 1
us_edges['decile_label'] = us_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(us_edges['max_capacity_mw'])
cmax = np.max(us_edges['max_capacity_mw'])
us_edges = us_edges[us_edges['csr_id_A'] != 0]
us_edges = gpd.GeoDataFrame(us_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in us_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
us_edges.plot(ax=ax, 
               column='max_capacity_mw',
               legend=True,
               cmap = new_cmap,
               figsize = (20, 20),
               legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
               #line width varies with capacity
               linewidth = 0.5 + us_edges.max_capacity_mw/2500,
               zorder = 2)

# centroid capacities
us_capacities['centroid'].plot(ax=ax,
                                figsize = (20, 20), 
                                color = "palegoldenrod", 
                                edgecolor = "black",
                                # marker size varies with capacity
                                markersize = geo_scale * np.sqrt(us_capacities.capacity_mw),
                                linewidth = 0.4,
                                zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=15)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.2, 
         s='Total Generating Capacity: \n' + f"{int(round(us_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])


#plt.show()
fig.savefig(os.path.join(output_path, "us_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% us: plot line improvements AND relevant centroid capacities

# get geographic dimensions of the map
w = us_shp.total_bounds[2] - us_shp.total_bounds[0]
l = us_shp.total_bounds[3] - us_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(w,l) ** 2)

# select the edges that are only in the US
us_ids = list(us_shp['csr_id'])

# cap capacities at 5000 for bin construction
improvements = improvements_save
improvements['max_capacity_mw'].where(improvements['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = improvements.columns
improvements = pd.concat([improvements, dummy], axis=0)
improvements['capacity_bin'] = pd.cut(improvements['max_capacity_mw'], 10, labels=False)
improvements['capacity_bin'] = improvements['capacity_bin'] + 1
improvements['decile_label'] = improvements['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(improvements['max_capacity_mw'])
cmax = np.max(improvements['max_capacity_mw'])
improvements = improvements[improvements['csr_id_A'] != 0]
improvements = gpd.GeoDataFrame(improvements, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in improvements['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)


# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
improvements.plot(ax=ax, 
               column='max_capacity_mw',
               legend=True,
               cmap = new_cmap,
               figsize = (20, 20),
               legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
               #line width varies with capacity
               linewidth = 0.5 + improvements.max_capacity_mw/2500,
               zorder = 2)

# centroid capacities
imp_capacities['centroid'].plot(ax=ax,
                                figsize = (20, 20), 
                                color = "palegoldenrod", 
                                edgecolor = "black",
                                # marker size varies with capacity
                                markersize = geo_scale * np.sqrt(imp_capacities.capacity_mw),
                                linewidth = 0.4,
                                zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Capacity Change', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=15)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])


#plt.show()
fig.savefig(os.path.join(output_path, "us_map_region_edge_cap_imp.png"), dpi=250, bbox_inches='tight')




#%% north america: plot aggregate line-capacities

# select the edges that are only in the northamer
northamer_ids = list(northamer_shp['csr_id'])
northamer_edges = edges[(edges['csr_id_A'].isin(northamer_ids)) | (edges['csr_id_B'].isin(northamer_ids))]

# cap capacities at 5000 for bin construction
northamer_edges_save = northamer_edges
northamer_edges['max_capacity_mw'].where(northamer_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = northamer_edges.columns
northamer_edges = pd.concat([northamer_edges, dummy], axis=0)
northamer_edges['capacity_bin'] = pd.cut(northamer_edges['max_capacity_mw'], 10, labels=False)
northamer_edges['capacity_bin'] = northamer_edges['capacity_bin'] + 1
northamer_edges['decile_label'] = northamer_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(northamer_edges['max_capacity_mw'])
cmax = np.max(northamer_edges['max_capacity_mw'])
northamer_edges = northamer_edges[northamer_edges['csr_id_A'] != 0]
northamer_edges = gpd.GeoDataFrame(northamer_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in northamer_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
northamer_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + northamer_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
northamer_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 7.5,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northamer_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% north america: plot line AND centroid capacities

# get geographic dimensions of the map
w = northamer_shp.total_bounds[2] - northamer_shp.total_bounds[0]
l = northamer_shp.total_bounds[3] - northamer_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(w,l) ** 2)

# select the edges that are only in the northamer
northamer_ids = list(northamer_shp['csr_id'])
northamer_edges = edges[(edges['csr_id_A'].isin(northamer_ids)) & (edges['csr_id_B'].isin(northamer_ids))]

# cap capacities at 5000 for bin construction
northamer_edges_save = northamer_edges
northamer_edges['max_capacity_mw'].where(northamer_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = northamer_edges.columns
northamer_edges = pd.concat([northamer_edges, dummy], axis=0)
northamer_edges['capacity_bin'] = pd.cut(northamer_edges['max_capacity_mw'], 10, labels=False)
northamer_edges['capacity_bin'] = northamer_edges['capacity_bin'] + 1
northamer_edges['decile_label'] = northamer_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(northamer_edges['max_capacity_mw'])
cmax = np.max(northamer_edges['max_capacity_mw'])
northamer_edges = northamer_edges[northamer_edges['csr_id_A'] != 0]
northamer_edges = gpd.GeoDataFrame(northamer_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in northamer_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
northamer_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale * northamer_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
na_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale * np.sqrt(na_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.2 * geo_scale * 4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.2 * geo_scale * 6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.2 * geo_scale * 8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.2 * geo_scale * 9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.2 * geo_scale * 11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.2 * geo_scale * 15)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.285, 
         s='Total Generating Capacity: \n' + f"{int(round(na_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northamer_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')




#%% north america: plot edges and capacities

# select the edges that are only in north america
na_ids = list(northamer_shp['csr_id'])
na_edges = edges[(edges['csr_id_A'].isin(na_ids)) & (edges['csr_id_B'].isin(na_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
na_edges.plot(ax=ax, 
              color='black', 
              figsize = (20, 20),
              # line width varies with number of edges
              linewidth = 0.2 + na_edges.num_of_edge/10,
              zorder = 2)

# capacities
na_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + na_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "na_map_edges_cap.png"), dpi=250, bbox_inches='tight')




#%% north america: plot edges and capacities

# select the edges that are only in north america
na_ids = list(northamer_shp['csr_id'])
na_edges = edges[(edges['csr_id_A'].isin(na_ids)) & (edges['csr_id_B'].isin(na_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
na_edges.plot(ax=ax, 
              color='black', 
              figsize = (20, 20),
              # line width varies with number of edges
              linewidth = 0.2 + na_edges.num_of_edge/10,
              zorder = 2)

# capacities
na_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + na_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "na_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% north america: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.35,
                   zorder = 1)

# capacities
na_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + na_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_map_cap.png"), dpi=200, bbox_inches='tight')




#%% north america: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
na_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "yellow", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 2 + na_capacities.rnw_capacity_mw/50,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_map_renewcap.png"), dpi=200, bbox_inches='tight')





#%% north america: plot regional fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
northamer_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
na_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 2 + na_capacities.ffl_capacity_mw/50,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% north america: plot capacity heatmap

df_na_capheat = northamer_shp.merge(na_capacities, how='left', on='csr_id')
df_na_capheat = gpd.GeoDataFrame(df_na_capheat, geometry='geometry')
df_na_capheat['cap_decile'] = pd.qcut(df_na_capheat['capacity_mw'].rank(method='first'), 10, labels=False)
df_na_capheat['ffl_cap_decile'] = pd.qcut(df_na_capheat['ffl_capacity_mw'].rank(method='first'), 10, labels=False)
df_na_capheat['cap_decile'] = df_na_capheat['cap_decile'] + 1
df_na_capheat['ffl_cap_decile'] = df_na_capheat['ffl_cap_decile'] + 1

# identify percentiles for capping the plot
np.mean(df_na_capheat['capacity_mw'])
np.percentile(df_na_capheat['capacity_mw'], 25)
np.percentile(df_na_capheat['capacity_mw'], 75)
np.percentile(df_na_capheat['capacity_mw'], 100)

# cap at 2500
df_na_capheat['capacity_mw'] = np.where(df_na_capheat['capacity_mw'] >= 5000, 5000, df_na_capheat['capacity_mw'])

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_capheat.plot(column='capacity_mw',
               ax=ax,
               figsize = (20, 20), 
               cmap = 'jet', 
               edgecolor = "black", 
               linewidth = 0.15,
               legend = True,
               legend_kwds={'shrink': 0.43, 'ticks': []}) # no ticks for now
               #legend_kwds={'shrink': 0.43, 'ticks': [0, 1000, 2000, 3000, 4000, 5000]})

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_capheat_map.png"), dpi=250, bbox_inches='tight')




#%% north america: plot solar potentials

# merge shapefile with model data
df_na_modeldata = northamer_shp.merge(df_modeldata, how='left', on='csr_id')

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_modeldata.plot(column='solar_mean',
               ax=ax,
               figsize = (20, 20), 
               cmap = 'jet', 
               edgecolor = "black", 
               linewidth = 0.15,
               legend = True,
               legend_kwds={'shrink': 0.43, 'ticks': [-0.95, -0.90, -0.85, -0.8, -0.75]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['-95%', '-90%', '-85%', '-80%', '-75%'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_solarpot_map.png"), dpi=250, bbox_inches='tight')




#%% north america: plot wind potentials

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_modeldata.plot(column='wind_mean',
               ax=ax,
               figsize = (20, 20), 
               cmap = 'jet', 
               edgecolor = "black", 
               linewidth = 0.15,
               legend = True,
               legend_kwds={'shrink': 0.43, 'ticks': [-0.95, -0.90, -0.85, -0.8, -0.75]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['-95%', '-90%', '-85%', '-80%', '-75%'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_windpot_map.png"), dpi=250, bbox_inches='tight')



#%% north america: plot capacity potentials (deciles)

# append one dummy observation
#dummy = {'cap_decile': 0}
#df_na_capheat = df_na_capheat.append(dummy, ignore_index=True)

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_capheat.plot(column='cap_decile',
                   ax=ax,
                   figsize = (20, 20), 
                   cmap = 'jet', 
                   edgecolor = "black", 
                   linewidth = 0.15,
                   legend = True,
                   legend_kwds = {'shrink': 0.43, 'ticks': [0,1,2,3,4,5,6,7,8,9,10]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_capacity_deciles.png"), dpi=250, bbox_inches='tight')




#%% north america: plot fossil capacity potentials (deciles)

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_capheat.plot(column='ffl_cap_decile',
                   ax=ax,
                   figsize = (20, 20), 
                   cmap = 'jet', 
                   edgecolor = "black", 
                   linewidth = 0.15,
                   legend = True,
                   legend_kwds = {'shrink': 0.43, 'ticks': [0,1,2,3,4,5,6,7,8,9,10]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_fossil_capacity_deciles.png"), dpi=250, bbox_inches='tight')



#%% north america: plot solar potentials (deciles)

# deciles
df_na_modeldata['solar_decile'] = pd.qcut(df_na_modeldata['solar_mean'], 10, labels=False)
df_na_modeldata['wind_decile'] = pd.qcut(df_na_modeldata['wind_mean'], 10, labels=False)
df_na_modeldata['solar_decile'] = df_na_modeldata['solar_decile'] + 1
df_na_modeldata['wind_decile'] = df_na_modeldata['wind_decile'] + 1

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_modeldata.plot(column='solar_decile',
                     ax=ax,
                     figsize = (20, 20), 
                     cmap = 'jet', 
                     edgecolor = "black", 
                     linewidth = 0.15,
                     legend = True,
                     legend_kwds = {'shrink': 0.43, 'ticks': [0,1,2,3,4,5,6,7,8,9,10]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_solarpot_deciles.png"), dpi=250, bbox_inches='tight')




#%% north america: plot wind potentials (deciles)

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_na_modeldata.plot(column='wind_decile',
                     ax=ax,
                     figsize = (20, 20), 
                     cmap = 'jet', 
                     edgecolor = "black", 
                     linewidth = 0.15,
                     legend = True,
                     legend_kwds = {'shrink': 0.43, 'ticks': [0,1,2,3,4,5,6,7,8,9,10]})

cbar = ax.get_figure().get_axes()[1]
cbar.set_yticklabels(['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
cbar.tick_params(labelsize=18)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "northam_windpot_deciles.png"), dpi=250, bbox_inches='tight')




#%% europe: plot aggregate line-capacities

# select the edges that are only in the europe
europe_ids = list(europe_shp['csr_id'])
europe_edges = edges[(edges['csr_id_A'].isin(europe_ids)) | (edges['csr_id_B'].isin(europe_ids))]

# cap capacities at 5000 for bin construction
europe_edges_save = europe_edges
europe_edges['max_capacity_mw'].where(europe_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = europe_edges.columns
europe_edges = pd.concat([europe_edges, dummy], axis=0)
europe_edges['capacity_bin'] = pd.cut(europe_edges['max_capacity_mw'], 10, labels=False)
europe_edges['capacity_bin'] = europe_edges['capacity_bin'] + 1
europe_edges['decile_label'] = europe_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(europe_edges['max_capacity_mw'])
cmax = np.max(europe_edges['max_capacity_mw'])
europe_edges = europe_edges[europe_edges['csr_id_A'] != 0]
europe_edges = gpd.GeoDataFrame(europe_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in europe_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
europe_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + europe_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
europe_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 7.5,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "europe_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% europe: plot line AND centroid capacities

# get geographic dimensions of the map
w = europe_shp.total_bounds[2] - europe_shp.total_bounds[0]
l = europe_shp.total_bounds[3] - europe_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(w,l) ** 2)

# select the edges that are only in the europe
europe_ids = list(europe_shp['csr_id'])
europe_edges = edges[(edges['csr_id_A'].isin(europe_ids)) & (edges['csr_id_B'].isin(europe_ids))]

# cap capacities at 5000 for bin construction
europe_edges_save = europe_edges
europe_edges['max_capacity_mw'].where(europe_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = europe_edges.columns
europe_edges = pd.concat([europe_edges, dummy], axis=0)
europe_edges['capacity_bin'] = pd.cut(europe_edges['max_capacity_mw'], 10, labels=False)
europe_edges['capacity_bin'] = europe_edges['capacity_bin'] + 1
europe_edges['decile_label'] = europe_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(europe_edges['max_capacity_mw'])
cmax = np.max(europe_edges['max_capacity_mw'])
europe_edges = europe_edges[europe_edges['csr_id_A'] != 0]
europe_edges = gpd.GeoDataFrame(europe_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in europe_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
europe_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale * europe_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
eu_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(eu_capacities.capacity_mw), # used to be divided by 50
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)



# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.73 * geo_scale * 4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.73 * geo_scale * 6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.73 * geo_scale * 8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.73 * geo_scale * 9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.73 * geo_scale * 11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.68 * geo_scale * 15)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.3, 
         s='Total Generating Capacity: \n' + f"{int(round(eu_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "europe_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%%
# =============================================================================
# eu_cap_save = eu_capacities
# 
# eu_capacities = eu_cap_save
# eu_capacities = eu_capacities.iloc[[(np.abs(eu_capacities['capacity_mw'] - 30000)).argmin(),]]
# 
# =============================================================================


#%% europe: plot edges and capacities

# select the edges that are only in China
eu_ids = list(europe_shp['csr_id'])
eu_edges = edges[(edges['csr_id_A'].isin(eu_ids)) & (edges['csr_id_B'].isin(eu_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# centroid to centroid edges
eu_edges.plot(ax=ax, 
              color='black', 
              figsize = (20, 20),
              # line width varies with number of edges
              linewidth = 0.2 + eu_edges.num_of_edge/10,
              zorder = 2)

# capacities
eu_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + eu_capacities.capacity_mw/40,
                               linewidth = 0.5,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "eu_map_edges_cap.png"), dpi=250, bbox_inches='tight')



#%% europe: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.35,
                zorder = 1)

# capacities
eu_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + eu_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "europe_map_cap.png"), dpi=200, bbox_inches='tight')





#%% europe: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.35,
                zorder = 1)

# capacities
eu_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + eu_capacities.rnw_capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "europe_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% europe: plot regional fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
europe_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
eu_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 2 + eu_capacities.ffl_capacity_mw/50,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "europe_map_fossilcap.png"), dpi=200, bbox_inches='tight')





#%% both: plot edges and capacities

# select the edges that are in either europe or north america
na_eu_edges = pd.concat([na_edges, eu_edges], axis=0)

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
na_eu_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.25,
                zorder = 1)

# centroid to centroid edges
na_eu_edges = na_eu_edges.set_crs(na_eu_shp.crs)
na_eu_edges.plot(ax=ax, 
              color='black', 
              figsize = (20, 20),
              # line width varies with number of edges
              linewidth = 0.1 + na_eu_edges.num_of_edge/11,
              zorder = 2)

# capacities
#na_eu_capacities = na_eu_capacities.set_crs(na_eu_shp.crs)
na_eu_capacities['centroid'].plot(ax=ax,
                                  figsize = (20, 20), 
                                  color = "palegoldenrod", 
                                  edgecolor = "black",
                                  # marker size varies with capacity
                                  markersize = 2 + na_eu_capacities.capacity_mw/175,
                                  linewidth = 0.3,
                                  zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "na_eu_map_edges_cap.png"), dpi=250, bbox_inches='tight')



#%% china: plot aggregate line-capacities


# select the edges that are only in the china
china_ids = list(china_shp['csr_id'])
china_edges = edges[(edges['csr_id_A'].isin(china_ids)) | (edges['csr_id_B'].isin(china_ids))]

# cap capacities at 5000 for bin construction
china_edges_save = china_edges
china_edges['max_capacity_mw'].where(china_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = china_edges.columns
china_edges = pd.concat([china_edges, dummy], axis=0)
china_edges['capacity_bin'] = pd.cut(china_edges['max_capacity_mw'], 10, labels=False)
china_edges['capacity_bin'] = china_edges['capacity_bin'] + 1
china_edges['decile_label'] = china_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(china_edges['max_capacity_mw'])
cmax = np.max(china_edges['max_capacity_mw'])
china_edges = china_edges[china_edges['csr_id_A'] != 0]
china_edges = gpd.GeoDataFrame(china_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in china_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
china_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.45},
              # line width varies with capacity
              linewidth = 0.5 + china_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
china_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% china: plot line AND centroid capacities

# get geographic dimensions of the map
w = china_shp.total_bounds[2] - china_shp.total_bounds[0]
l = china_shp.total_bounds[3] - china_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)

# select the edges that are only in the china
china_ids = list(china_shp['csr_id'])
china_edges = edges[(edges['csr_id_A'].isin(china_ids)) & (edges['csr_id_B'].isin(china_ids))]

# cap capacities at 5000 for bin construction
china_edges_save = china_edges
china_edges['max_capacity_mw'].where(china_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = china_edges.columns
china_edges = pd.concat([china_edges, dummy], axis=0)
china_edges['capacity_bin'] = pd.cut(china_edges['max_capacity_mw'], 10, labels=False)
china_edges['capacity_bin'] = china_edges['capacity_bin'] + 1
china_edges['decile_label'] = china_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(china_edges['max_capacity_mw'])
cmax = np.max(china_edges['max_capacity_mw'])
china_edges = china_edges[china_edges['csr_id_A'] != 0]
china_edges = gpd.GeoDataFrame(china_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in china_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
china_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale * china_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
cn_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale * np.sqrt(cn_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.63 * geo_scale * 4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.63 * geo_scale * 6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.63 * geo_scale * 8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.63 * geo_scale * 9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.63 * geo_scale * 11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.63 * geo_scale * 14)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           borderpad=1.1,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.25, 
         s='Total Generating Capacity: \n' + f"{int(round(cn_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')




#%% china: plot edges and capacities

# select the edges that are only in China
china_ids = list(china_shp['csr_id'])
china_edges = edges[(edges['csr_id_A'].isin(china_ids)) & (edges['csr_id_B'].isin(china_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
china_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.2 + china_edges.num_of_edge/10,
           zorder = 2)

# capacities
cn_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + cn_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% china: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
cn_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + cn_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_cap.png"), dpi=200, bbox_inches='tight')





#%% china: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
cn_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + cn_capacities.rnw_capacity_mw/40, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% china: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
china_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
cn_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 2 + cn_capacities.ffl_capacity_mw/50,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "china_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% india:  plot aggregate line-capacities

# select the edges that are only in the india
india_ids = list(india_shp['csr_id'])
india_edges = edges[(edges['csr_id_A'].isin(india_ids)) | (edges['csr_id_B'].isin(india_ids))]

# cap capacities at 5000 for bin construction
india_edges_save = india_edges
india_edges['max_capacity_mw'].where(india_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = india_edges.columns
india_edges = pd.concat([india_edges, dummy], axis=0)
india_edges['capacity_bin'] = pd.cut(india_edges['max_capacity_mw'], 10, labels=False)
india_edges['capacity_bin'] = india_edges['capacity_bin'] + 1
india_edges['decile_label'] = india_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(india_edges['max_capacity_mw'])
cmax = np.max(india_edges['max_capacity_mw'])
india_edges = india_edges[india_edges['csr_id_A'] != 0]
india_edges = gpd.GeoDataFrame(india_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in india_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
india_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.45},
              # line width varies with capacity
              linewidth = 0.5 + india_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
india_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 25,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% india: plot line AND centroid capacities

# get geographic dimensions of the map
w = india_shp.total_bounds[2] - india_shp.total_bounds[0]
l = india_shp.total_bounds[3] - india_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
s = (w + l) / 2
geo_scale = 6400 / (s ** 2)


# select the edges that are only in the india
india_ids = list(india_shp['csr_id'])
india_edges = edges[(edges['csr_id_A'].isin(india_ids)) & (edges['csr_id_B'].isin(india_ids))]

# cap capacities at 5000 for bin construction
india_edges_save = india_edges
india_edges['max_capacity_mw'].where(india_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = india_edges.columns
india_edges = pd.concat([india_edges, dummy], axis=0)
india_edges['capacity_bin'] = pd.cut(india_edges['max_capacity_mw'], 10, labels=False)
india_edges['capacity_bin'] = india_edges['capacity_bin'] + 1
india_edges['decile_label'] = india_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(india_edges['max_capacity_mw'])
cmax = np.max(india_edges['max_capacity_mw'])
india_edges = india_edges[india_edges['csr_id_A'] != 0]
india_edges = gpd.GeoDataFrame(india_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in india_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
india_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.2, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + india_edges.max_capacity_mw/1500,
              zorder = 2)

# centroid capacities
in_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(in_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2.5 * 4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2.5 * 6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2.5 * 8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2.5 * 9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=2.5 * 11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=2.5 * 13.2)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=2,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           borderpad=1.5,
           fontsize=13)

plt.text(1.02,
         0.31, 
         s='Total Generating Capacity: \n' + f"{int(round(in_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% india: plot edges

# select the edges that are only in India
india_ids = list(india_shp['csr_id'])
india_edges = edges[(edges['csr_id_A'].isin(india_ids)) | (edges['csr_id_B'].isin(india_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
india_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + india_edges.num_of_edge/8,
           zorder = 2)

# centroids
india_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 50,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_edges.png"), dpi=250, bbox_inches='tight')




#%% india: plot edges and capacities

# select the edges that are only in india
india_ids = list(india_shp['csr_id'])
india_edges = edges[(edges['csr_id_A'].isin(india_ids)) & (edges['csr_id_B'].isin(india_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
india_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.2 + india_edges.num_of_edge/10,
           zorder = 2)

# capacities
in_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + in_capacities.capacity_mw/20,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% india: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
in_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + in_capacities.capacity_mw/20,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_cap.png"), dpi=200, bbox_inches='tight')





#%% india: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
in_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + in_capacities.rnw_capacity_mw/15, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% india: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
india_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
in_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + in_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "india_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% russia: plot aggregate line-capacities

# select the edges that are only in the russia
russia_ids = list(russia_shp['csr_id'])
russia_edges = edges[(edges['csr_id_A'].isin(russia_ids)) | (edges['csr_id_B'].isin(russia_ids))]

# cap capacities at 5000 for bin construction
russia_edges_save = russia_edges
russia_edges['max_capacity_mw'].where(russia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = russia_edges.columns
russia_edges = pd.concat([russia_edges, dummy], axis=0)
russia_edges['capacity_bin'] = pd.cut(russia_edges['max_capacity_mw'], 10, labels=False)
russia_edges['capacity_bin'] = russia_edges['capacity_bin'] + 1
russia_edges['decile_label'] = russia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(russia_edges['max_capacity_mw'])
cmax = np.max(russia_edges['max_capacity_mw'])
russia_edges = russia_edges[russia_edges['csr_id_A'] != 0]
russia_edges = gpd.GeoDataFrame(russia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in russia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
russia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.15},
              # line width varies with capacity
              linewidth = 0.4 + russia_edges.max_capacity_mw/3000,
              zorder = 2)

# centroids
russia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 10,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% russia: plot line AND centroid capacities

# get geographic dimensions of the map
w = russia_shp.total_bounds[2] - russia_shp.total_bounds[0]
l = russia_shp.total_bounds[3] - russia_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
s = (w + l) / 2
geo_scale = 6400 / (s ** 2)

# select the edges that are only in the russia
russia_ids = list(russia_shp['csr_id'])
russia_edges = edges[(edges['csr_id_A'].isin(russia_ids)) & (edges['csr_id_B'].isin(russia_ids))]

# cap capacities at 5000 for bin construction
russia_edges_save = russia_edges
russia_edges['max_capacity_mw'].where(russia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = russia_edges.columns
russia_edges = pd.concat([russia_edges, dummy], axis=0)
russia_edges['capacity_bin'] = pd.cut(russia_edges['max_capacity_mw'], 10, labels=False)
russia_edges['capacity_bin'] = russia_edges['capacity_bin'] + 1
russia_edges['decile_label'] = russia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(russia_edges['max_capacity_mw'])
cmax = np.max(russia_edges['max_capacity_mw'])
russia_edges = russia_edges[russia_edges['csr_id_A'] != 0]
russia_edges = gpd.GeoDataFrame(russia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in russia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
russia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale*russia_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
ru_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(ru_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.7*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.7*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.7*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=0.7*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.7*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=0.7*15)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.155, 
         s='Total Generating Capacity: \n' + f"{int(round(ru_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% russia: plot edges

# select the edges that are only in russia
russia_ids = list(russia_shp['csr_id'])
russia_edges = edges[(edges['csr_id_A'].isin(russia_ids)) | (edges['csr_id_B'].isin(russia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
russia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + russia_edges.num_of_edge/8,
           zorder = 2)

# centroids
russia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 20,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_edges.png"), dpi=250, bbox_inches='tight')



#%% russia: plot edges and capacities

# select the edges that are only in russia
russia_ids = list(russia_shp['csr_id'])
russia_edges = edges[(edges['csr_id_A'].isin(russia_ids)) & (edges['csr_id_B'].isin(russia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
russia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.2 + russia_edges.num_of_edge/10,
           zorder = 2)

# capacities
ru_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + ru_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% russia: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
ru_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + ru_capacities.capacity_mw/50,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_cap.png"), dpi=200, bbox_inches='tight')





#%% russia: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
ru_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + ru_capacities.rnw_capacity_mw/30, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% russia: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
russia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
ru_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + ru_capacities.ffl_capacity_mw/50,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "russia_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% brazil: plot aggregate line-capacities

# select the edges that are only in the brazil
brazil_ids = list(brazil_shp['csr_id'])
brazil_edges = edges[(edges['csr_id_A'].isin(brazil_ids)) | (edges['csr_id_B'].isin(brazil_ids))]

# cap capacities at 5000 for bin construction
brazil_edges_save = brazil_edges
brazil_edges['max_capacity_mw'].where(brazil_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = brazil_edges.columns
brazil_edges = pd.concat([brazil_edges, dummy], axis=0)
brazil_edges['capacity_bin'] = pd.cut(brazil_edges['max_capacity_mw'], 10, labels=False)
brazil_edges['capacity_bin'] = brazil_edges['capacity_bin'] + 1
brazil_edges['decile_label'] = brazil_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(brazil_edges['max_capacity_mw'])
cmax = np.max(brazil_edges['max_capacity_mw'])
brazil_edges = brazil_edges[brazil_edges['csr_id_A'] != 0]
brazil_edges = gpd.GeoDataFrame(brazil_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in brazil_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
brazil_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + brazil_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
brazil_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 25,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_linecap.png"), dpi=250, bbox_inches='tight')




#%% brazil: plot line AND centroid capacities

# get geographic dimensions of the map
w = brazil_shp.total_bounds[2] - brazil_shp.total_bounds[0]
l = brazil_shp.total_bounds[3] - brazil_shp.total_bounds[1]

# take the longest dimension and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(w,l) ** 2)

# select the edges that are only in the brazil
brazil_ids = list(brazil_shp['csr_id'])
brazil_edges = edges[(edges['csr_id_A'].isin(brazil_ids)) & (edges['csr_id_B'].isin(brazil_ids))]

# cap capacities at 5000 for bin construction
brazil_edges_save = brazil_edges
brazil_edges['max_capacity_mw'].where(brazil_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = brazil_edges.columns
brazil_edges = pd.concat([brazil_edges, dummy], axis=0)
brazil_edges['capacity_bin'] = pd.cut(brazil_edges['max_capacity_mw'], 10, labels=False)
brazil_edges['capacity_bin'] = brazil_edges['capacity_bin'] + 1
brazil_edges['decile_label'] = brazil_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(brazil_edges['max_capacity_mw'])
cmax = np.max(brazil_edges['max_capacity_mw'])
brazil_edges = brazil_edges[brazil_edges['csr_id_A'] != 0]
brazil_edges = gpd.GeoDataFrame(brazil_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in brazil_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
brazil_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale*brazil_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
br_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(br_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.3*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.3*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.3*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.3*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.3*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.3*14)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.3, 
         s='Total Generating Capacity: \n' + f"{int(round(br_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')




#%% brazil: plot edges

# select the edges that are only in brazil
brazil_ids = list(brazil_shp['csr_id'])
brazil_edges = edges[(edges['csr_id_A'].isin(brazil_ids)) | (edges['csr_id_B'].isin(brazil_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
brazil_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + brazil_edges.num_of_edge/5,
           zorder = 2)

# centroids
brazil_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_edges.png"), dpi=250, bbox_inches='tight')



#%% brazil: plot edges and capacities

# select the edges that are only in brazil
brazil_ids = list(brazil_shp['csr_id'])
brazil_edges = edges[(edges['csr_id_A'].isin(brazil_ids)) & (edges['csr_id_B'].isin(brazil_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
brazil_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.2 + brazil_edges.num_of_edge/5,
           zorder = 2)

# capacities
br_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + br_capacities.capacity_mw/15,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% brazil: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
br_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + br_capacities.capacity_mw/15,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_cap.png"), dpi=200, bbox_inches='tight')





#%% brazil: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
br_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + br_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% brazil: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
brazil_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
br_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + br_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "brazil_map_fossilcap.png"), dpi=200, bbox_inches='tight')




#%% australia: plot aggregate line-capacities

# select the edges that are only in the australia
australia_ids = list(australia_shp['csr_id'])
australia_edges = edges[(edges['csr_id_A'].isin(australia_ids)) | (edges['csr_id_B'].isin(australia_ids))]

# cap capacities at 5000 for bin construction
australia_edges_save = australia_edges
australia_edges['max_capacity_mw'].where(australia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = australia_edges.columns
australia_edges = pd.concat([australia_edges, dummy], axis=0)
australia_edges['capacity_bin'] = pd.cut(australia_edges['max_capacity_mw'], 10, labels=False)
australia_edges['capacity_bin'] = australia_edges['capacity_bin'] + 1
australia_edges['decile_label'] = australia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(australia_edges['max_capacity_mw'])
cmax = np.max(australia_edges['max_capacity_mw'])
australia_edges = australia_edges[australia_edges['csr_id_A'] != 0]
australia_edges = gpd.GeoDataFrame(australia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in australia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
australia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + australia_edges.max_capacity_mw/1500,
              zorder = 2)

# centroids
australia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 30,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_linecap.png"), dpi=250, bbox_inches='tight')




#%% australia: plot line AND centroid capacities

# get geographic dimensions of the map
w = australia_shp.total_bounds[2] - australia_shp.total_bounds[0]
l = australia_shp.total_bounds[3] - australia_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)

# select the edges that are only in the australia
australia_ids = list(australia_shp['csr_id'])
australia_edges = edges[(edges['csr_id_A'].isin(australia_ids)) & (edges['csr_id_B'].isin(australia_ids))]

# cap capacities at 5000 for bin construction
australia_edges_save = australia_edges
australia_edges['max_capacity_mw'].where(australia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = australia_edges.columns
australia_edges = pd.concat([australia_edges, dummy], axis=0)
australia_edges['capacity_bin'] = pd.cut(australia_edges['max_capacity_mw'], 10, labels=False)
australia_edges['capacity_bin'] = australia_edges['capacity_bin'] + 1
australia_edges['decile_label'] = australia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(australia_edges['max_capacity_mw'])
cmax = np.max(australia_edges['max_capacity_mw'])
australia_edges = australia_edges[australia_edges['csr_id_A'] != 0]
australia_edges = gpd.GeoDataFrame(australia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in australia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
australia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale*australia_edges.max_capacity_mw/2500,
              zorder = 2)

# centroid capacities
au_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(au_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.5*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.5*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.5*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=1.5*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.5*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=1.5*14.2)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.5,
           title_fontsize=15,
           borderpad=1.05,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.3, 
         s='Total Generating Capacity: \n' + f"{int(round(au_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% australia: plot edges

# select the edges that are only in australia
australia_ids = list(australia_shp['csr_id'])
australia_edges = edges[(edges['csr_id_A'].isin(australia_ids)) | (edges['csr_id_B'].isin(australia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
australia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + australia_edges.num_of_edge/5,
           zorder = 2)

# centroids
australia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_edges.png"), dpi=250, bbox_inches='tight')



#%% australia: plot edges and capacities

# select the edges that are only in australia
australia_ids = list(australia_shp['csr_id'])
australia_edges = edges[(edges['csr_id_A'].isin(australia_ids)) & (edges['csr_id_B'].isin(australia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
australia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + australia_edges.num_of_edge/3,
           zorder = 2)

# capacities
au_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + au_capacities.capacity_mw/10,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% australia: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
au_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + au_capacities.capacity_mw/15,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_cap.png"), dpi=200, bbox_inches='tight')





#%% australia: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
au_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + au_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% australia: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
australia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
au_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + au_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "australia_map_fossilcap.png"), dpi=200, bbox_inches='tight')





#%% southkorea: plot aggregate line-capacities

# select the edges that are only in the southkorea
southkorea_ids = list(southkorea_shp['csr_id'])
southkorea_edges = edges[(edges['csr_id_A'].isin(southkorea_ids)) | (edges['csr_id_B'].isin(southkorea_ids))]

# cap capacities at 5000 for bin construction
southkorea_edges_save = southkorea_edges
southkorea_edges['max_capacity_mw'].where(southkorea_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = southkorea_edges.columns
southkorea_edges = pd.concat([southkorea_edges, dummy], axis=0)
southkorea_edges['capacity_bin'] = pd.cut(southkorea_edges['max_capacity_mw'], 10, labels=False)
southkorea_edges['capacity_bin'] = southkorea_edges['capacity_bin'] + 1
southkorea_edges['decile_label'] = southkorea_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(southkorea_edges['max_capacity_mw'])
cmax = np.max(southkorea_edges['max_capacity_mw'])
southkorea_edges = southkorea_edges[southkorea_edges['csr_id_A'] != 0]
southkorea_edges = gpd.GeoDataFrame(southkorea_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in southkorea_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
southkorea_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 1 + southkorea_edges.max_capacity_mw/1500,
              zorder = 2)

# centroids
southkorea_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 100,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% south korea: plot line AND centroid capacities

# get geographic dimensions of the map
w = southkorea_shp.total_bounds[2] - southkorea_shp.total_bounds[0]
l = southkorea_shp.total_bounds[3] - southkorea_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)


# select the edges that are only in the south korea
southkorea_ids = list(southkorea_shp['csr_id'])
southkorea_edges = edges[(edges['csr_id_A'].isin(southkorea_ids)) & (edges['csr_id_B'].isin(southkorea_ids))]

# cap capacities at 5000 for bin construction
southkorea_edges_save = southkorea_edges
southkorea_edges['max_capacity_mw'].where(southkorea_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = southkorea_edges.columns
southkorea_edges = pd.concat([southkorea_edges, dummy], axis=0)
southkorea_edges['capacity_bin'] = pd.cut(southkorea_edges['max_capacity_mw'], 10, labels=False)
southkorea_edges['capacity_bin'] = southkorea_edges['capacity_bin'] + 1
southkorea_edges['decile_label'] = southkorea_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(southkorea_edges['max_capacity_mw'])
cmax = np.max(southkorea_edges['max_capacity_mw'])
southkorea_edges = southkorea_edges[southkorea_edges['csr_id_A'] != 0]
southkorea_edges = gpd.GeoDataFrame(southkorea_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in southkorea_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
southkorea_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.57, 'aspect': 17, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + southkorea_edges.max_capacity_mw/1000,
              zorder = 2)

# centroid capacities
kr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 0.7*geo_scale*np.sqrt(kr_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=9*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=9*11.5)
# =============================================================================
# patch20k = Line2D([0], [0], 
#                   linestyle='None', 
#                   marker='o', 
#                   color='black', 
#                   label='20000 MW', 
#                   markerfacecolor='palegoldenrod', 
#                   markeredgewidth=0.4, 
#                   markersize=9*14)
# =============================================================================
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=9,
           title_fontsize=15,
           borderpad=1.08,
           bbox_to_anchor=(1.04, 0.5),
           handletextpad=3,
           fontsize=13)

plt.text(1.02,
         0.08, 
         s='Total Generating Capacity: \n' + f"{int(round(kr_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% southkorea: plot edges

# select the edges that are only in southkorea
southkorea_ids = list(southkorea_shp['csr_id'])
southkorea_edges = edges[(edges['csr_id_A'].isin(southkorea_ids)) | (edges['csr_id_B'].isin(southkorea_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
southkorea_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + southkorea_edges.num_of_edge/5,
           zorder = 2)

# centroids
southkorea_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_edges.png"), dpi=250, bbox_inches='tight')



#%% southkorea: plot edges and capacities

# select the edges that are only in southkorea
southkorea_ids = list(southkorea_shp['csr_id'])
southkorea_edges = edges[(edges['csr_id_A'].isin(southkorea_ids)) & (edges['csr_id_B'].isin(southkorea_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
southkorea_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + southkorea_edges.num_of_edge/3,
           zorder = 2)

# capacities
kr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + kr_capacities.capacity_mw/10,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% southkorea: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
kr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + kr_capacities.capacity_mw/15,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_cap.png"), dpi=200, bbox_inches='tight')





#%% southkorea: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
kr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + kr_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% southkorea: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
southkorea_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
kr_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + kr_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "southkorea_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% japan: plot aggregate line-capacities

# select the edges that are only in the japan
japan_ids = list(japan_shp['csr_id'])
japan_edges = edges[(edges['csr_id_A'].isin(japan_ids)) | (edges['csr_id_B'].isin(japan_ids))]

# cap capacities at 5000 for bin construction
japan_edges_save = japan_edges
japan_edges['max_capacity_mw'].where(japan_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = japan_edges.columns
japan_edges = pd.concat([japan_edges, dummy], axis=0)
japan_edges['capacity_bin'] = pd.cut(japan_edges['max_capacity_mw'], 10, labels=False)
japan_edges['capacity_bin'] = japan_edges['capacity_bin'] + 1
japan_edges['decile_label'] = japan_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(japan_edges['max_capacity_mw'])
cmax = np.max(japan_edges['max_capacity_mw'])
japan_edges = japan_edges[japan_edges['csr_id_A'] != 0]
japan_edges = gpd.GeoDataFrame(japan_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in japan_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
japan_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + japan_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
japan_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 25,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% japan: plot line AND centroid capacities

# get geographic dimensions of the map
w = japan_shp.total_bounds[2] - japan_shp.total_bounds[0]
l = japan_shp.total_bounds[3] - japan_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)

# select the edges that are only in the japan
japan_ids = list(japan_shp['csr_id'])
japan_edges = edges[(edges['csr_id_A'].isin(japan_ids)) & (edges['csr_id_B'].isin(japan_ids))]

# cap capacities at 5000 for bin construction
japan_edges_save = japan_edges
japan_edges['max_capacity_mw'].where(japan_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = japan_edges.columns
japan_edges = pd.concat([japan_edges, dummy], axis=0)
japan_edges['capacity_bin'] = pd.cut(japan_edges['max_capacity_mw'], 10, labels=False)
japan_edges['capacity_bin'] = japan_edges['capacity_bin'] + 1
japan_edges['decile_label'] = japan_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(japan_edges['max_capacity_mw'])
cmax = np.max(japan_edges['max_capacity_mw'])
japan_edges = japan_edges[japan_edges['csr_id_A'] != 0]
japan_edges = gpd.GeoDataFrame(japan_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in japan_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
japan_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.165, 'aspect': 13, 'anchor': (0, 0.48), 'pad': -0.19},
              #line width varies with capacity
              linewidth = 0.5 + geo_scale*japan_edges.max_capacity_mw/4500,
              zorder = 2)

# centroid capacities
jp_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale * np.sqrt(jp_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)

# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='5000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=2*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=2*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='20000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=2*14)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=1.6,
           title_fontsize=15,
           borderpad=1.05,
           bbox_to_anchor=(0.8, 0.5),
           fontsize=13)

plt.text(0.8,
         0.33, 
         s='Total Generating Capacity: \n' + f"{int(round(jp_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%% japan: plot edges

# select the edges that are only in japan
japan_ids = list(japan_shp['csr_id'])
japan_edges = edges[(edges['csr_id_A'].isin(japan_ids)) | (edges['csr_id_B'].isin(japan_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
japan_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + japan_edges.num_of_edge/5,
           zorder = 2)

# centroids
japan_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_edges.png"), dpi=250, bbox_inches='tight')



#%% japan: plot edges and capacities

# select the edges that are only in japan
japan_ids = list(japan_shp['csr_id'])
japan_edges = edges[(edges['csr_id_A'].isin(japan_ids)) & (edges['csr_id_B'].isin(japan_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
japan_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.2 + japan_edges.num_of_edge/5,
           zorder = 2)

# capacities
jp_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + jp_capacities.capacity_mw/25,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% japan: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
jp_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + jp_capacities.capacity_mw/25,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_cap.png"), dpi=200, bbox_inches='tight')





#%% japan: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
jp_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + jp_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% japan: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
japan_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
jp_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + jp_capacities.ffl_capacity_mw/25,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "japan_map_fossilcap.png"), dpi=200, bbox_inches='tight')





#%% colombia: plot aggregate line-capacities

# select the edges that are only in the colombia
colombia_ids = list(colombia_shp['csr_id'])
colombia_edges = edges[(edges['csr_id_A'].isin(colombia_ids)) | (edges['csr_id_B'].isin(colombia_ids))]

# cap capacities at 5000 for bin construction
colombia_edges_save = colombia_edges
colombia_edges['max_capacity_mw'].where(colombia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = colombia_edges.columns
colombia_edges = pd.concat([colombia_edges, dummy], axis=0)
colombia_edges['capacity_bin'] = pd.cut(colombia_edges['max_capacity_mw'], 10, labels=False)
colombia_edges['capacity_bin'] = colombia_edges['capacity_bin'] + 1
colombia_edges['decile_label'] = colombia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(colombia_edges['max_capacity_mw'])
cmax = np.max(colombia_edges['max_capacity_mw'])
colombia_edges = colombia_edges[colombia_edges['csr_id_A'] != 0]
colombia_edges = gpd.GeoDataFrame(colombia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in colombia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
colombia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 1 + colombia_edges.max_capacity_mw/1500,
              zorder = 2)

# centroids
colombia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 50,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% colombia: plot line AND centroid capacities

# get geographic dimensions of the map
w = colombia_shp.total_bounds[2] - colombia_shp.total_bounds[0]
l = colombia_shp.total_bounds[3] - colombia_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)

# select the edges that are only in the colombia
colombia_ids = list(colombia_shp['csr_id'])
colombia_edges = edges[(edges['csr_id_A'].isin(colombia_ids)) & (edges['csr_id_B'].isin(colombia_ids))]

# cap capacities at 5000 for bin construction
colombia_edges_save = colombia_edges
colombia_edges['max_capacity_mw'].where(colombia_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = colombia_edges.columns
colombia_edges = pd.concat([colombia_edges, dummy], axis=0)
colombia_edges['capacity_bin'] = pd.cut(colombia_edges['max_capacity_mw'], 10, labels=False)
colombia_edges['capacity_bin'] = colombia_edges['capacity_bin'] + 1
colombia_edges['decile_label'] = colombia_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(colombia_edges['max_capacity_mw'])
cmax = np.max(colombia_edges['max_capacity_mw'])
colombia_edges = colombia_edges[colombia_edges['csr_id_A'] != 0]
colombia_edges = gpd.GeoDataFrame(colombia_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in colombia_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
colombia_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.305, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + colombia_edges.max_capacity_mw/500,
              zorder = 2)

# centroid capacities
co_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale * np.sqrt(co_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3.8*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3.8*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3.8*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3.8*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='2500 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=3.8*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='3000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=3.8*14)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k, patch20k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=3.5,
           title_fontsize=15,
           handletextpad=2,
           bbox_to_anchor=(1.04, 0.5),
           borderpad=2,
           fontsize=13)

plt.text(1.02,
         0.27, 
         s='Total Generating Capacity: \n' + f"{int(round(co_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_region_edge_cap.png"), dpi=250, bbox_inches='tight')



#%%

# =============================================================================
# co_cap_save = co_capacities
# 
# co_capacities = co_cap_save 
# co_capacities = co_capacities.iloc[[(np.abs(co_capacities['capacity_mw'] - 1500)).argmin(),]]
# 
# =============================================================================

#%% colombia: plot edges

# select the edges that are only in colombia
colombia_ids = list(colombia_shp['csr_id'])
colombia_edges = edges[(edges['csr_id_A'].isin(colombia_ids)) | (edges['csr_id_B'].isin(colombia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
colombia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + colombia_edges.num_of_edge/5,
           zorder = 2)

# centroids
colombia_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 40,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_edges.png"), dpi=250, bbox_inches='tight')



#%% colombia: plot edges and capacities

# select the edges that are only in colombia
colombia_ids = list(colombia_shp['csr_id'])
colombia_edges = edges[(edges['csr_id_A'].isin(colombia_ids)) & (edges['csr_id_B'].isin(colombia_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
colombia_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + colombia_edges.num_of_edge/2,
           zorder = 2)

# capacities
co_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + co_capacities.capacity_mw/5,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% colombia: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
co_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + co_capacities.capacity_mw/15,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_cap.png"), dpi=200, bbox_inches='tight')





#%% colombia: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
co_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + co_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_renewcap.png"), dpi=200, bbox_inches='tight')




#%% colombia: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
colombia_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
co_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + co_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "colombia_map_fossilcap.png"), dpi=200, bbox_inches='tight')



#%% turkey: plot aggregate line-capacities

# select the edges that are only in the turkey
turkey_ids = list(turkey_shp['csr_id'])
turkey_edges = edges[(edges['csr_id_A'].isin(turkey_ids)) & (edges['csr_id_B'].isin(turkey_ids))]

# cap capacities at 5000 for bin construction
turkey_edges_save = turkey_edges
turkey_edges['max_capacity_mw'].where(turkey_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = turkey_edges.columns
turkey_edges = pd.concat([turkey_edges, dummy], axis=0)
turkey_edges['capacity_bin'] = pd.cut(turkey_edges['max_capacity_mw'], 10, labels=False)
turkey_edges['capacity_bin'] = turkey_edges['capacity_bin'] + 1
turkey_edges['decile_label'] = turkey_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(turkey_edges['max_capacity_mw'])
cmax = np.max(turkey_edges['max_capacity_mw'])
turkey_edges = turkey_edges[turkey_edges['csr_id_A'] != 0]
turkey_edges = gpd.GeoDataFrame(turkey_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in turkey_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
turkey_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.5},
              # line width varies with capacity
              linewidth = 0.4 + turkey_edges.max_capacity_mw/2500,
              zorder = 2)

# centroids
turkey_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "black", 
                   edgecolor = "black",
                   markersize = 25,
                   linewidth = 0.8,
                   zorder = 3)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=15)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_linecap.png"), dpi=250, bbox_inches='tight')



#%% turkey: plot line AND centroid capacities

# get geographic dimensions of the map
w = turkey_shp.total_bounds[2] - turkey_shp.total_bounds[0]
l = turkey_shp.total_bounds[3] - turkey_shp.total_bounds[1]

# take the mean of dimensions and use it as a scale for the marker size of bubbles
geo_scale = 6400 / (np.maximum(l,w) ** 2)

# select the edges that are only in the turkey
turkey_ids = list(turkey_shp['csr_id'])
turkey_edges = edges[(edges['csr_id_A'].isin(turkey_ids)) & (edges['csr_id_B'].isin(turkey_ids))]

# cap capacities at 5000 for bin construction
turkey_edges_save = turkey_edges
turkey_edges['max_capacity_mw'].where(turkey_edges['max_capacity_mw'] < 5000, 5000, inplace=True)

# dummy to set the lowest value at 0
dummy = pd.DataFrame(np.zeros(9)).transpose()
dummy.columns = turkey_edges.columns
turkey_edges = pd.concat([turkey_edges, dummy], axis=0)
turkey_edges['capacity_bin'] = pd.cut(turkey_edges['max_capacity_mw'], 10, labels=False)
turkey_edges['capacity_bin'] = turkey_edges['capacity_bin'] + 1
turkey_edges['decile_label'] = turkey_edges['capacity_bin'] * 500

# define min and max values for colormap, delete dummy
cmin = np.min(turkey_edges['max_capacity_mw'])
cmax = np.max(turkey_edges['max_capacity_mw'])
turkey_edges = turkey_edges[turkey_edges['csr_id_A'] != 0]
turkey_edges = gpd.GeoDataFrame(turkey_edges, geometry = 'line')

# create color map
norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax, clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.gist_heat)
color_list = [mapper.to_rgba(x)[0] for x in turkey_edges['max_capacity_mw']]


# truncate the color map so we can use just the red part (no white)
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = plt.get_cmap('gist_heat')
new_cmap = truncate_colormap(cmap, 0, 0.6)


# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
turkey_edges.plot(ax=ax, 
              column='max_capacity_mw',
              legend=True,
              cmap = new_cmap,
              figsize = (20, 20),
              legend_kwds = {'shrink': 0.18, 'aspect': 13, 'anchor': (0, 0.48)},
              #line width varies with capacity
              linewidth = 0.5 + turkey_edges.max_capacity_mw/750,
              zorder = 2)

# centroid capacities
tr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = geo_scale*np.sqrt(tr_capacities.capacity_mw),
                               linewidth = 0.4,
                               zorder = 2)

# edit legend colorbar
cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_yticklabels(['0','1000 MW','2000 MW','3000 MW','4000 MW','>5000 MW'])
cbar.set_title('Line Capacity', loc='left', size=16, pad=21)


# add legend for bubbles
patch5h = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', color='black', 
                 label='200 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3*4.5)
patch1k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='500 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3*6)
patch2k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='1000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3*8)
patch5k = Line2D([0], [0], 
                 linestyle='None', 
                 marker='o', 
                 color='black', 
                 label='2000 MW', 
                 markerfacecolor='palegoldenrod', 
                 markeredgewidth=0.4, 
                 markersize=3*9.3)
patch10k = Line2D([0], [0], 
                 linestyle='None',  
                  marker='o', 
                  color='black', 
                  label='5000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=3*11.5)
patch20k = Line2D([0], [0], 
                  linestyle='None', 
                  marker='o', 
                  color='black', 
                  label='10000 MW', 
                  markerfacecolor='palegoldenrod', 
                  markeredgewidth=0.4, 
                  markersize=3*14)
plt.legend(handles=(patch5h, patch1k, patch2k, patch5k, patch10k),
           loc='center right',
           title='Regional Capacity',
           labelspacing=2.25,
           title_fontsize=15,
           borderpad=1.1,
           bbox_to_anchor=(1.04, 0.5),
           fontsize=13)

plt.text(1.02,
         0.08, 
         s='Total Generating Capacity: \n' + f"{int(round(tr_capacities['capacity_mw'].sum(),0)):,d}" + ' MW', 
         horizontalalignment='center',
         verticalalignment='center',
         fontsize=12, 
         transform=ax.transAxes)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_region_edge_cap.png") , dpi=250, bbox_inches='tight')




#%% turkey: plot edges

# select the edges that are only in turkey
turkey_ids = list(turkey_shp['csr_id'])
turkey_edges = edges[(edges['csr_id_A'].isin(turkey_ids)) & (edges['csr_id_B'].isin(turkey_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
turkey_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.5 + turkey_edges.num_of_edge/5,
           zorder = 2)

# centroids
turkey_shp['centroid'].plot(ax=ax,
                   figsize = (20, 20), 
                   color = "yellow", 
                   edgecolor = "black",
                   markersize = 60,
                   linewidth = 0.75,
                   zorder = 3)


# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_edges.png"), dpi=250, bbox_inches='tight')



#%% turkey: plot edges and capacities

# select the edges that are only in turkey
turkey_ids = list(turkey_shp['csr_id'])
turkey_edges = edges[(edges['csr_id_A'].isin(turkey_ids)) & (edges['csr_id_B'].isin(turkey_ids))]

# plot
fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.5,
            zorder = 1)

# centroid to centroid edges
turkey_edges.plot(ax=ax, 
           color='black', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.3 + turkey_edges.num_of_edge/5,
           zorder = 2)

# capacities
tr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 2 + tr_capacities.capacity_mw/10,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_edges_cap.png"), dpi=250, bbox_inches='tight')





#%% turkey: plot regional capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
                   figsize = (20, 20), 
                   color = "lightskyblue", 
                   edgecolor = "white", 
                   linewidth = 0.5,
                   zorder = 1)

# capacities
tr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "palegoldenrod", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 3 + tr_capacities.capacity_mw/10,
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_cap.png") , dpi=200, bbox_inches='tight')





#%% turkey: plot region renewable capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
                figsize = (20, 20), 
                color = "lightskyblue", 
                edgecolor = "white", 
                linewidth = 0.5,
                zorder = 1)

# capacities
tr_capacities['centroid'].plot(ax=ax,
                               figsize = (20, 20), 
                               color = "yellow", 
                               edgecolor = "black",
                               # marker size varies with capacity
                               markersize = 5 + tr_capacities.rnw_capacity_mw/10, #divided by 50 above
                               linewidth = 0.4,
                               zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_renewcap.png") , dpi=200, bbox_inches='tight')




#%% turkey: plot region fossil capacity

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
turkey_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# capacities
tr_capacities['centroid'].plot(ax=ax,
                            figsize = (20, 20), 
                            color = "orangered", 
                            edgecolor = "black",
                            # marker size varies with capacity
                            markersize = 5 + tr_capacities.ffl_capacity_mw/15,
                            linewidth = 0.4,
                            zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "turkey_map_fossilcap.png"), dpi=200, bbox_inches='tight')






#%% us, plot all lines

# find lines in the us
us_lines =  gpd.sjoin(lines, us_shp[['csr_id', 'geometry']], how='left', predicate='intersects')
us_lines = us_lines[us_lines['csr_id'].notnull()]

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
us_shp.plot(ax=ax,
            figsize = (20, 20), 
            color = "lightskyblue", 
            edgecolor = "white", 
            linewidth = 0.35,
            zorder = 1)

# centroid to centroid edges
us_lines.plot(ax=ax, 
           color='red', 
           figsize = (20, 20),
           # line width varies with number of edges
           linewidth = 0.3,
           zorder = 2)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

#plt.show()
fig.savefig(os.path.join(output_path, "us_map_alllines.png"), dpi=150, bbox_inches='tight')


# reset warnings
warnings.filterwarnings('default')



