# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:54:47 2022

Power Line Capacity Estimation

@author: Kevin Yin
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

# GeoPy 2.3.0
try:
    from geopy.distance import distance
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'geopy'])
finally:
    from geopy.distance import distance




#%% parameters

# root directory
directory_path = os.path.realpath(__file__)[:-40]
os.chdir(directory_path)

# data output path
data_path = os.path.join(directory_path, "data", "sql_output")

# surge impedance
z0 = 400



#%% override option

# if running this script from the master, override the directory path
try:
   os.chdir(master_directory_path)
   data_path = os.path.join(master_directory_path, "data", "sql_output")
except NameError:
    pass



#%% clean data

# import data
df_lines = pd.read_csv(os.path.join(data_path, "links.csv"))

# estimate distances 
df_lines['tupleA'] = list(zip(df_lines.latitude_1, df_lines.longitude_1))
df_lines['tupleB'] = list(zip(df_lines.latitude_2, df_lines.longitude_2))
df_lines['distance_km'] = df_lines[['tupleA', 'tupleB']].apply(lambda x: distance(x['tupleA'], x['tupleB']).km, axis=1)
df_lines.drop(columns=['tupleA','tupleB'], inplace=True)

# de-string voltages
def destring_list(str_list):
    float_list = []
    for i in str_list:
        try:
            # some voltages are recorded in kV, convert to V
            if float(i) < 1000:
                float_list.append(1000*float(i))
            else:
                float_list.append(float(i))
        except ValueError:
            float_list.append(np.nan)
    return float_list

# apply the above function
df_lines['v_data'] = 1
df_lines.loc[df_lines['voltage'].isnull(), 'v_data'] = 0
df_lines.loc[df_lines['voltage'].isnull(), 'voltage'] = '0'

# convert strings of voltages into a list of strings (some strings are split by semi-colons)
df_lines['voltage'] = df_lines['voltage'].str.split(',')

# apply the destring function
df_lines['voltage'] = df_lines['voltage'].apply(lambda x: destring_list(x))
df_lines['voltage_avg'] = df_lines['voltage'].apply(lambda x: np.nanmean(x))

# make voltages into kilovolts
df_lines['voltage_kv'] = df_lines['voltage_avg']/1000



#%% impute voltages

# construct distance bins
df_lines_no_0 = df_lines[df_lines['voltage_kv'] != 0]
bin_edges = np.arange(0, 2501, 25).tolist()
df_lines_no_0['dist_bin'] = pd.cut(df_lines_no_0['distance_km'], bin_edges)
voltage_bins = df_lines_no_0.groupby(by='dist_bin').voltage_kv.mean()
voltage_bins = voltage_bins.reset_index(level=0)

# identify missing voltages, match to distance bin, and impute missing values
mask_1 = df_lines['voltage_kv'] == 0

for b in voltage_bins.dist_bin:
    mask_2 = df_lines['distance_km'] > b.left
    mask_3 = df_lines['distance_km'] <= b.right
    df_lines.loc[mask_1 & mask_2 & mask_3,'voltage_kv'] = voltage_bins.loc[voltage_bins['dist_bin'] == b,'voltage_kv'].values[0]


# test interval for descriptive statistics
b = pd.Interval(0,25,closed='right')
# how many lines are less than 25 kilometers?
df_lines[(df_lines['distance_km'] > b.left) & (df_lines['distance_km'] <= b.right)]['voltage_kv'].count()
# how many lines are less than 25 kilometers and have missing voltage?
df_lines[(df_lines['distance_km'] > b.left) & (df_lines['distance_km'] <= b.right) & mask_1]['voltage_kv'].count()
# how many missing voltages total?
df_lines[mask_1]['voltage_kv'].count()



#%% estimate capacity

# function to estimate capacity in MW using St. Clair curve as per Dunlop et. al. (curve was fitted manually using miles)
def st_clair(km,v,z0):
    miles = km*0.621371
    pu_sil = 0.1021681 + (11.43091 - 0.1021681)/(1 + (miles/12.49187) ** 0.7700907)
    capacity = pu_sil * np.square(v) / z0
    return capacity


# estimate capacities
df_lines['max_capacity_mw'] = st_clair(df_lines['distance_km'], df_lines['voltage_kv'], z0)


# drop processing columns
df_lines.drop(columns=['voltage_avg','voltage','v_data'], inplace=True)


# export
df_lines.to_csv(os.path.join(data_path, "links_cap.csv"), index=False)


# reset warnings
warnings.filterwarnings('default')




