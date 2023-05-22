"""

Master Script for "Clean Growth", Arkolakis & Walsh (2023)
@author: Kevin Yin

Note: For trouble shooting or smaller tasks, each of the called scripts can be run individually. 

"""


#%% user parameters

import sys
import os

# variable to determine whether to run from scratch or from intermediates, NO is default and recommended
# If you have an explicit reason to abstract OSM data from scratch, set to YES
# running from scratch can take several (up to 5) days
run_from_scratch = "NO"



#%% run scripts

# root directory (changes the directory automatically in submodules, no need to change directories in the called scripts)
master_directory_path = os.path.realpath(__file__)[:-23]

# code directory
code_path = os.path.join(master_directory_path, "code", "Python")

# include code folder as part of module search
sys.path.append(code_path)


#%%

# execute 
print("Running scripts.")

if run_from_scratch == "NO":
    import line_capacity_estimation
    print("STEP 1: Line capacities estimation finished.")
    import aggregate_grid
    print("STEP 2: Grid aggregation finished.")
    import plot_figures
    print("STEP 3: Plotting figures finished.")
    import plot_model_output
    print("STEP 4: Plotting model outputs finished.")

if run_from_scratch  == "YES":
    import osm_abstraction_algorithm
    print("STEP 0: Abstraction algorithm finished.")
    import line_capacity_estimation
    print("STEP 1: Line capacities estimation finished.")
    import aggregate_grid
    print("STEP 2: Grid aggregation finished.")
    import plot_figures
    print("STEP 3: Plotting figures finished.")
    import plot_model_output
    print("STEP 4: Plotting model outputs finished.")


print("Run completed.")