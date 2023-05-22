# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:12:23 2022

@author: ky297
"""

import subprocess
import sys
import os
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

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

# GeoPandas 0.12.2
try:
    import geopandas as gpd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopandas'])
finally:
    import geopandas as gpd

# Matplotlib 3.7.1
try:
    import matplotlib.pyplot as plt
    from matplotlib import rc
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'matplotlib'])
finally:
    import matplotlib.pyplot as plt
    from matplotlib import rc

# Shapely 2.0.1
try:
    from shapely.geometry import Polygon
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'shapely'])
finally:
    from shapely.geometry import Polygon



#%% directory


# 1. root directory
directory_path = os.path.realpath(__file__)[:-33]
os.chdir(directory_path)

# 2. data path (where model outputs go)
data_path = os.path.join(directory_path, "data", "mod_output")

# 3. Shapefile path
shape_path = os.path.join(directory_path, "data", "shapefile", "selected_regions")

# 4. Image output path
output_path = os.path.join(directory_path, "figures")



#%% override option

# if running this script from the master, override the directory path
try:
   directory_path = master_directory_path
   os.chdir(master_directory_path)
   data_path = os.path.join(master_directory_path, "data", "mod_output")
   shape_path = os.path.join(master_directory_path, "data", "shapefile", "selected_regions")
   output_path = os.path.join(master_directory_path, "figures")
except NameError:
    pass


#%% imports


# FIXME: shapefile needs to be stored in DropBox

# import shapefile data
csr_path = os.path.join(shape_path, "selected_regions.shp")
csr_shp = gpd.read_file(csr_path)
csr_shp['ctry_code'] = csr_shp['reg_id'].str[:2]
csr_shp = csr_shp.to_crs("EPSG:4326")

# import price change data
wage_path = os.path.join(data_path, "wagechange.csv")
df_wage_change = pd.read_csv(wage_path, names=['wage_ch', 'csr_id'])

# import wage change data
price_path = os.path.join(data_path, "priceresults.csv")
df_price_change = pd.read_csv(price_path, names=['price_ch', 'csr_id'])


# set font to Latex Palatino
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# =============================================================================
# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
# =============================================================================



#%% process data


# country codes
us_codes = ['US']

china_codes = ['CN']

europe_codes = ['AT', # austria
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


# get region shape files
us_shp = csr_shp[csr_shp['ctry_code'].isin(us_codes)]
us_shp = gpd.clip(us_shp, Polygon([[-130, 10], [50, 10], [50, 52], [-130, 52]]))

europe_shp = csr_shp[csr_shp['ctry_code'].isin(europe_codes)]
europe_shp = gpd.clip(europe_shp, Polygon([[-34, 35], [50, 35], [50, 75], [-34, 75]]))

china_shp = csr_shp[csr_shp['ctry_code'].isin(china_codes)]


# merge price changes with shapefiles
df_us = us_shp.merge(df_wage_change, how='left', on=['csr_id'])
df_us = df_us.merge(df_price_change, how='left', on=['csr_id'])

df_europe = europe_shp.merge(df_wage_change, how='left', on=['csr_id'])
df_europe = df_europe.merge(df_price_change, how='left', on=['csr_id'])

df_china = china_shp.merge(df_wage_change, how='left', on=['csr_id'])
df_china = df_china.merge(df_price_change, how='left', on=['csr_id'])



#%% check for outliers and clip

# us
plt.boxplot(df_us['price_ch'])
plt.show()
plt.boxplot(df_us['wage_ch'])
plt.show()

# europe
plt.boxplot(df_europe['price_ch'])
plt.show()
plt.boxplot(df_europe['wage_ch'])
plt.show()

# china
plt.boxplot(df_china['price_ch'])
plt.show()
plt.boxplot(df_china['wage_ch'])
plt.show()

# function for finding outlier boundaries (we define outliers conservatively at twice the IQR)
def find_outliers(data):
    q1 = np.nanquantile(data, 0.25)
    q3 = np.nanquantile(data, 0.75)
    iq_range = q3-q1
    upper = q3 + (1.5 * iq_range)
    lower = q1 - (1.5 * iq_range)
    return upper, lower


# clip values to highlight differences, label outliers

# us - wages
df_us['w_outlier'] = np.where(df_us['wage_ch'] > find_outliers(df_us['wage_ch'])[0], 1, 0)
df_us['w_outlier'] = np.where(df_us['wage_ch'] < find_outliers(df_us['wage_ch'])[1], -1, df_us['w_outlier'])

df_us['clip_wage'] = np.where(df_us['wage_ch'] > find_outliers(df_us['wage_ch'])[0], find_outliers(df_us['wage_ch'])[0], df_us['wage_ch'])
df_us['clip_wage'] = np.where(df_us['wage_ch'] < find_outliers(df_us['wage_ch'])[1], find_outliers(df_us['wage_ch'])[1], df_us['clip_wage'])

# us - prices
df_us['p_outlier'] = np.where(df_us['price_ch'] > find_outliers(df_us['price_ch'])[0], 1, 0)
df_us['p_outlier'] = np.where(df_us['price_ch'] < find_outliers(df_us['price_ch'])[1], -1, df_us['p_outlier'])

df_us['clip_price'] = np.where(df_us['price_ch'] > find_outliers(df_us['price_ch'])[0], find_outliers(df_us['price_ch'])[0], df_us['price_ch'])
df_us['clip_price'] = np.where(df_us['price_ch'] < find_outliers(df_us['price_ch'])[1], find_outliers(df_us['price_ch'])[1], df_us['clip_price'])


# europe - wages
df_europe['w_outlier'] = np.where(df_europe['wage_ch'] > find_outliers(df_europe['wage_ch'])[0], 1, 0)
df_europe['w_outlier'] = np.where(df_europe['wage_ch'] < find_outliers(df_europe['wage_ch'])[1], -1, df_europe['w_outlier'])

df_europe['clip_wage'] = np.where(df_europe['wage_ch'] > find_outliers(df_europe['wage_ch'])[0], find_outliers(df_europe['wage_ch'])[0], df_europe['wage_ch'])
df_europe['clip_wage'] = np.where(df_europe['wage_ch'] < find_outliers(df_europe['wage_ch'])[1], find_outliers(df_europe['wage_ch'])[1], df_europe['clip_wage'])

# europe - prices
df_europe['p_outlier'] = np.where(df_europe['price_ch'] > find_outliers(df_europe['price_ch'])[0], 1, 0)
df_europe['p_outlier'] = np.where(df_europe['price_ch'] < find_outliers(df_europe['price_ch'])[1], -1, df_europe['p_outlier'])

df_europe['clip_price'] = np.where(df_europe['price_ch'] > find_outliers(df_europe['price_ch'])[0], find_outliers(df_europe['price_ch'])[0], df_europe['price_ch'])
df_europe['clip_price'] = np.where(df_europe['price_ch'] < find_outliers(df_europe['price_ch'])[1], find_outliers(df_europe['price_ch'])[1], df_europe['clip_price'])


# china - wages
df_china['w_outlier'] = np.where(df_china['wage_ch'] > find_outliers(df_china['wage_ch'])[0], 1, 0)
df_china['w_outlier'] = np.where(df_china['wage_ch'] < find_outliers(df_china['wage_ch'])[1], -1, df_china['w_outlier'])

df_china['clip_wage'] = np.where(df_china['wage_ch'] > find_outliers(df_china['wage_ch'])[0], find_outliers(df_china['wage_ch'])[0], df_china['wage_ch'])
df_china['clip_wage'] = np.where(df_china['wage_ch'] < find_outliers(df_china['wage_ch'])[1], find_outliers(df_china['wage_ch'])[1], df_china['clip_wage'])


# china - prices
df_china['p_outlier'] = np.where(df_china['price_ch'] > find_outliers(df_china['price_ch'])[0], 1, 0)
df_china['p_outlier'] = np.where(df_china['price_ch'] < find_outliers(df_china['price_ch'])[1], -1, df_china['p_outlier'])

df_china['clip_price'] = np.where(df_china['price_ch'] > find_outliers(df_china['price_ch'])[0], find_outliers(df_china['price_ch'])[0], df_china['price_ch'])
df_china['clip_price'] = np.where(df_china['price_ch'] < find_outliers(df_china['price_ch'])[1], find_outliers(df_china['price_ch'])[1], df_china['clip_price'])




#%% automate tick ranges

# count zeros after decimal point, if no decimal then count digits
def count_zeros(value):
    if np.abs(value) < 1:
        zeros = len(str(int(np.round(1 / np.abs(value), 0)))) - 1
        return zeros
    if np.abs(value) > 1:
        digits_plus = len(str(int(np.abs(value)))) * (-1) + 1
        return digits_plus


# define 5 ticks between the high and low numbers
def construct_ticks(high,low):
    
    val_range = high - low
    power = count_zeros(val_range) + 2
    tens = 10 ** power
    interval = np.floor( (val_range * tens) / 4) / tens
    print('Interval: ' + str(interval))
    
    t1 = np.ceil(low / interval) * interval
    t2 = np.round(t1 + interval, count_zeros(interval) + 1) 
    t3 = np.round(t2 + interval, count_zeros(interval) + 1) 
    t4 = np.round(t3 + interval, count_zeros(interval) + 1) 
    t5 = np.round(t4 + interval, count_zeros(interval) + 1) 
   
    return t1,t2,t3,t4,t5


# normalize price to the lowest outlier
df_us['relative_price'] = df_us['clip_price'] / find_outliers(df_us['price_ch'])[1]
df_europe['relative_price'] = df_europe['clip_price'] / find_outliers(df_europe['price_ch'])[1]
df_china['relative_price'] = df_china['clip_price'] / find_outliers(df_china['price_ch'])[1]

df_us['mrel_price'] = df_us['clip_price'] / np.mean(df_us['clip_price'])
df_europe['mrel_price'] = df_europe['clip_price'] / np.mean(df_europe['clip_price'])
df_china['mrel_price'] = df_china['clip_price'] / np.mean(df_china['clip_price'])


# multiply wage changes by 100 to get percent changes
df_us['clip_wage_per'] = df_us['clip_wage'] * 100
df_europe['clip_wage_per'] = df_europe['clip_wage'] * 100
df_china['clip_wage_per'] = df_china['clip_wage'] * 100


# define tick ranges
us_ticks_w = construct_ticks(find_outliers(df_us['wage_ch'])[0] * 100, find_outliers(df_us['wage_ch'])[1]  * 100)
#top = find_outliers(df_us['price_ch'])[0] / find_outliers(df_us['price_ch'])[1]
us_ticks_p = construct_ticks(np.max(df_us['mrel_price']),np.min(df_us['mrel_price']))

europe_ticks_w = construct_ticks(find_outliers(df_europe['wage_ch'])[0]  * 100, find_outliers(df_europe['wage_ch'])[1]  * 100)
#top = find_outliers(df_europe['price_ch'])[0] / find_outliers(df_europe['price_ch'])[1]
europe_ticks_p = construct_ticks(np.max(df_us['mrel_price']),np.min(df_us['mrel_price']))

china_ticks_w = construct_ticks(find_outliers(df_china['wage_ch'])[0]  * 100, find_outliers(df_china['wage_ch'])[1]  * 100)
#top = find_outliers(df_china['price_ch'])[0] / find_outliers(df_china['price_ch'])[1]
china_ticks_p = construct_ticks(np.max(df_us['mrel_price']),np.min(df_us['mrel_price']))




#%% us - wage changes

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_us.plot(column='clip_wage_per',
           ax=ax,
           figsize = (20, 20), 
           cmap = 'jet', 
           edgecolor = "black", 
           linewidth = 0.15,
           legend = True,
           legend_kwds={'shrink': 0.43, 
                        'ticks': [us_ticks_w[0], 
                                  us_ticks_w[1], 
                                  us_ticks_w[2], 
                                  us_ticks_w[3], 
                                  us_ticks_w[4]],
                        'pad':-0.01})


cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=18)
cbar.set_title('Wage Change (\%)', size=18, pad=21)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.show()
fig.savefig(os.path.join(output_path, "us_wagechange_map.png"), dpi=250, bbox_inches='tight')





#%% us - price changes


fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_us.plot(column='mrel_price',
           ax=ax,
           figsize = (20, 20), 
           cmap = 'jet', 
           edgecolor = "black", 
           linewidth = 0.15,
           legend = True,
           legend_kwds={'shrink': 1, 'aspect':4.5, 'ticks': [us_ticks_p[0], us_ticks_p[1], us_ticks_p[2], us_ticks_p[3], us_ticks_p[4]], 'pad':-0.01})

cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=12)
cbar.set_title('Relative Price', size=12, pad=12)
cbar.set_position([0.153,0.325,1,0.04])

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.show()
fig.savefig(os.path.join(output_path, "us_relativeprice_map.png") , dpi=250, bbox_inches='tight')




#%% europe - wage changes

fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_europe.plot(column='clip_wage_per',
               ax=ax,
               figsize = (20, 20), 
               cmap = 'jet', 
               edgecolor = "black", 
               linewidth = 0.15,
               legend = True,
               legend_kwds={'shrink': 0.43, 'ticks': [europe_ticks_w[0], europe_ticks_w[1], europe_ticks_w[2], europe_ticks_w[3], europe_ticks_w[4]], 'pad':-0.01})

cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=18)
cbar.set_title('Wage Change (\%)', size=18, pad=21)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.show()
fig.savefig(os.path.join(output_path, "europe_wagechange_map.png") , dpi=250, bbox_inches='tight')





#%% china - wage changes


fig, ax = plt.subplots(figsize=(20,20), frameon=False)

# zone shapes
df_china.plot(column='clip_wage_per',
               ax=ax,
               figsize = (20, 20), 
               cmap = 'jet', 
               edgecolor = "black", 
               linewidth = 0.15,
               legend = True,
               legend_kwds={'shrink': 0.43, 'ticks': [china_ticks_w[0], china_ticks_w[1], china_ticks_w[2], china_ticks_w[3], china_ticks_w[4]], 'pad':-0.01})

cbar = ax.get_figure().get_axes()[1]
cbar.tick_params(labelsize=18)
cbar.set_title('Wage Change (\%)', size=18, pad=21)

# get rid of border
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# delete axis ticks
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.show()
fig.savefig(os.path.join(output_path, "china_wagechange_map.png") , dpi=250, bbox_inches='tight')




