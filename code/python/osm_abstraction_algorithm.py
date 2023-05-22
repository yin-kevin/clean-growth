"""
Kevin Yin 
Jan 18, 2022

This script applies a SQL algorithm to process OSM power grid data and makes it computationally 
feasible by looping over rectangular partitions instead of considering the whole data set at
once. It connects to a Postgres database containing the data and divides it into equal sized boxes
and applies the algorithm to each box, concatenating the resulting data at the end. 

The user can set parameters at the beginning that determine the parent directory, the
output folder, the name of the Postgres database, the bounding box of the country, and the number of 
box partitions. 
"""

#%% imports

import os
import numpy as np
import time
import psycopg2
import warnings

from psycopg2 import OperationalError

# suppress warnings
warnings.filterwarnings('ignore')



#%% user parameters

# 1. root directory
directory_path = os.path.realpath(__file__)[:-41]
os.chdir(directory_path)

# 2. specify paths
# where output .csv files will be written to
output_path = os.path.join(directory_path, "data", "sql_output")
# where the scripts for the SQL algorithm are stored
sql_path = os.path.join(directory_path, "code", "sql")

# 3. database name
db_name = "world_db"

# 4. long\lat coordinates of minimum bounding box for the entire country
x_west = -180
x_east = 180
y_north = 90
y_south = -90

# 5. number of segments dividing each axis (determines the number of boxes) 
x_numseg = 120
y_numseg = 70

# 6. How many boxes to test and the ID of the first box (0 is the lowest)
num_of_box = 8400
init_box_id = 0




#%% override option

# if running this script from the master, override the directory path
try:
   directory_path = master_directory_path
   os.chdir(master_directory_path)
   sql_path = os.path.join(master_directory_path, "code", "sql")
   output_path = os.path.join(master_directory_path, "data", "sql_output")
   db_name = master_db
except NameError:
    pass




#%% create boxes

# define increment sizes
x_increment = np.abs(x_west-x_east)/x_numseg
y_increment = np.abs(y_north-y_south)/y_numseg

# record coordinates
x_coords = [x_west + n*x_increment for n in range(x_numseg + 1)]
y_coords = [y_south + n*y_increment for n in range(y_numseg + 1)]

# create boxes
# NOTE: the adding and subtracting 0.1 makes the boxes overlap so that stations on the border get clustered
# boxes are numbered increasing from south to north, west to east
box_list = []

for i in range(x_numseg):
    for j in range(y_numseg):
        box_list.append([x_west + (i)*x_increment - 0.1, 
                         y_south + (j)*y_increment - 0.1, 
                         x_west + (i+1)*x_increment + 0.1, 
                         y_south + (j+1)*y_increment + 0.1]
                        )

# Set the test set list of boxes we'll operate over
test_set = list(range(init_box_id, init_box_id + num_of_box))




#%% functions

# function for connecting to Postgres database
def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection


# function to convert time to hours, minutes, seconds
def time_convert(start, end):
    total_seconds = end - start
    # calculate hours
    hours = int(total_seconds/3600)
    minutes = int((total_seconds - hours * 3600)/60)
    seconds = round(total_seconds - hours * 3600 - minutes * 60, 3)
    # only return non-zero units
    if hours > 0:
        return "{} hrs, {} min, {} sec.".format(hours, minutes, seconds)
    if hours == 0 and minutes > 0:
        return "{} min, {} sec.".format(minutes, seconds)
    if hours == 0 and minutes == 0:
        return "{} sec.".format(seconds)

# in case of error, rollback
#with conn.cursor() as cur:
#    cur.execute("rollback;")




#%% SQL: create tables


# track computation time
start_time = time.time()

# connect
conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")

# run un-looped portion of the algorithm, creates power_lines_orig and power_stations_orig tables
with open(os.path.join(sql_path, "OSM_algorithm_1_setup.txt")) as algo1_file:
    algo1 = algo1_file.read()

with conn.cursor() as cur:
    cur.execute(algo1)
    conn.commit()
    cur.close()

conn.close()
print("Setup complete, tables created.")

print("Cumulative execution time:", time_convert(start_time, time.time()))



#%% SQL: line and station processing

# ESTIMATED RUNTIME: 42 HRS
start_time = time.time()

# for box in box_list:
counter = 0
for i in test_set:
    
    box = box_list[i]
    counter += 1
    
    print("starting box {}".format(i))
    
    # create the bounding box in the database using the box coordinates
    box_lines = """
                CREATE TABLE power_lines_box AS
                SELECT * FROM power_lines_orig
                WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way);
                """.format(box[0], box[1], box[2], box[3])
    
    box_stations = """
                   CREATE TABLE power_stations_box AS
                   SELECT * FROM power_stations_orig
                   WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way); 
                   """.format(box[0], box[1], box[2], box[3])

    # remove these 'objects' from power_'object'_orig to keep track of unedited lines/stations
    delete_processed = """
                       DELETE FROM power_lines_orig X WHERE EXISTS
                           (SELECT line_id FROM power_lines_box Y WHERE X.line_id = Y.line_id);

                       DELETE FROM power_stations_orig X WHERE EXISTS 
                           (SELECT station_id FROM power_stations_box Y WHERE X.station_id = Y.station_id);
                       """
                                 
    # the SQL script takes in tables called "power_lines" and "power_stations" so we rename for compatibility
    rename_tables = """
                    ALTER TABLE power_lines_box
                    RENAME TO power_lines;
                    ALTER TABLE power_stations_box
                    RENAME TO power_stations;
                    """
    
    # create the box as an object we can relate other tables to
    create_box = """ 
                 CREATE TABLE bounding_box AS
                 SELECT ST_MakeEnvelope({}, {}, {}, {}, 4326) AS polygon;
                 """.format(box[0], box[1], box[2], box[3])
    
    # run remaining algorithm on that box, split so we can vacuum between steps
    # NOTE: the algorithm names are only approximate summaries of their purpose, each performs many kinds of operations
    with open(os.path.join(sql_path, "OSM_algorithm_2_split.txt")) as algo2_file:
        algo2 = algo2_file.read()
    with open(os.path.join(sql_path, "OSM_algorithm_3_merge.txt")) as algo3_file:
        algo3 = algo3_file.read()
    with open(os.path.join(sql_path, "OSM_algorithm_4_trueT.txt")) as algo4_file:
        algo4 = algo4_file.read()
    with open(os.path.join(sql_path, "OSM_algorithm_5_nearT.txt")) as algo5_file:
        algo5 = algo5_file.read()
    
    # put new data back into main table
    insert_new = """
                 INSERT INTO power_lines_orig (line_id, power, cables, voltage, wires, way)
                 SELECT line_id, power, cables, voltage, wires, way FROM power_lines;

                 INSERT INTO power_stations_orig (station_id, power, voltage, way)
                 SELECT station_id, power, voltage, way FROM power_stations;
                 
                 CREATE TABLE temp AS
                 SELECT ROW_NUMBER() OVER() AS new_line_id, * FROM power_lines_orig;
                 
                 ALTER TABLE temp
                 DROP COLUMN line_id;
                 ALTER TABLE temp
                 RENAME COLUMN new_line_id TO line_id;
                 
                 DROP TABLE power_lines_orig;
                 ALTER TABLE temp
                 RENAME TO power_lines_orig;
                 
                 CREATE TABLE temp AS
                 SELECT ROW_NUMBER() OVER() AS new_station_id, * FROM power_stations_orig;
                 
                 ALTER TABLE temp
                 DROP COLUMN station_id;
                 ALTER TABLE temp
                 RENAME COLUMN new_station_id TO station_id;
                 
                 DROP TABLE power_stations_orig;
                 ALTER TABLE temp
                 RENAME TO power_stations_orig;
                 """
    
    # reset database tables
    reset_data = """
                 DROP TABLE power_lines;
                 DROP TABLE power_stations;
                 DROP TABLE power_lines_init;
                 DROP TABLE power_stations_init;
                 DROP TABLE bounding_box;
                 """
                 
    conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")
    
    # execute setup
    with conn.cursor() as cur:
        cur.execute(box_lines)
        conn.commit()
        cur.execute(box_stations)
        conn.commit()
        cur.execute(delete_processed)
        conn.commit()
        cur.execute(rename_tables)
        conn.commit()
        cur.execute(create_box)
        conn.commit()
    
    # algorithm
    with conn.cursor() as cur:
        cur.execute(algo2)
        conn.commit()
    print("algo 2 completed")
    with conn.cursor() as cur:
        cur.execute(algo3)
        conn.commit()
    print("algo 3 completed")
    with conn.cursor() as cur:
        cur.execute(algo4)
        conn.commit()
    print("algo 4 completed")
    with conn.cursor() as cur:
        cur.execute(algo5)
        conn.commit()
    print("algo 5 completed")
    
    # insert updates and reset tables
    with conn.cursor() as cur:
        cur.execute(insert_new)
        conn.commit()
        cur.execute(reset_data)
        conn.commit()
    
    conn.close()
    
    # show number of boxes completed (not the same as the box index)
    print("Number of boxes processed:", counter)
    
# conn.close()
print("Cumulative execution time:", time_convert(start_time, time.time()))




#%% SQL: line straightening

conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")

# redefine the boxes so they don't overlap
box_list = []

for i in range(x_numseg):
    for j in range(y_numseg):
        box_list.append([x_west + (i)*x_increment, 
                         y_south + (j)*y_increment, 
                         x_west + (i+1)*x_increment, 
                         y_south + (j+1)*y_increment]
                        )


# create a column that indicates whether or not the line was run through the abstraction
abstracted_indicate = """
                       ALTER TABLE power_lines_orig
                       ADD abstracted int; 
                       """
with conn.cursor() as cur:                       
    cur.execute(abstracted_indicate)
    conn.commit()


# loop over non-overlapping boxes to abstract and clean lines
counter = 0
for i in test_set:
    
    box = box_list[i]
    counter += 1

    # create the bounding box in the database using the box coordinates, this time only lines fully contained
    box_lines = """
                CREATE TABLE power_lines_box AS
                SELECT * FROM power_lines_orig
                WHERE ST_Contains(ST_MakeEnvelope({}, {}, {}, {}, 4326), way);
                """.format(box[0], box[1], box[2], box[3])
    
    delete_processed = """
                       DELETE FROM power_lines_orig X WHERE EXISTS
                           (SELECT line_id FROM power_lines_box Y WHERE X.line_id = Y.line_id);
                       """
    
    # define the abstraction algorithm which will mark each line that has been abstracted
    with open(os.path.join(sql_path, "OSM_algorithm_6_straight.txt")) as algo6_file:
        algo6 = algo6_file.read()
    
    # FIXME
    delete_dup = """
                 DELETE FROM power_lines_clean A
                 WHERE EXISTS (SELECT FROM power_lines_clean B
                               WHERE A.line_id > B.line_id
                               AND ST_EQUALS(A.way, B.way))
                 """
    
    # after abstraction, put the lines back into power_lines_orig
    insert_new = """
                 INSERT INTO power_lines_orig (line_id, power, cables, voltage, wires, way, abstracted)
                 SELECT line_id, power, cables, voltage, wires, way, abstracted FROM power_lines_clean;
                 
                 CREATE TABLE temp AS
                 SELECT ROW_NUMBER() OVER() AS new_line_id, * FROM power_lines_orig;
                 
                 ALTER TABLE temp
                 DROP COLUMN line_id;
                 ALTER TABLE temp
                 RENAME COLUMN new_line_id TO line_id;
                 
                 DROP TABLE power_lines_orig;
                 ALTER TABLE temp
                 RENAME TO power_lines_orig;
                 """
    
    # reset database tables
    reset_data = """
                 DROP TABLE power_lines_box;
                 DROP TABLE power_lines_clean;
                 """
    
    # execute the statements defined above
    with conn.cursor() as cur:
        cur.execute(box_lines)
        conn.commit()
        cur.execute(delete_processed)
        conn.commit()
    
    # run main body of algorithm
    with conn.cursor() as cur:
        cur.execute(algo6)
        conn.commit()
    
    # insert updates and reset tables
    with conn.cursor() as cur:
        cur.execute(delete_dup)
        conn.commit()
        cur.execute(insert_new)
        conn.commit()
        cur.execute(reset_data)
        conn.commit()
    
    # show completed boxes
    print("Number of boxes straightened:", counter)


conn.close()
print("Cumulative execution time:", time_convert(start_time, time.time()))




#%% SQL: leftover lines

# create box that includes all used boxes
left = []
right = []
top = []
bottom = []

for i in test_set:
    left.append(box_list[i][0])
    right.append(box_list[i][2])
    top.append(box_list[i][1])
    bottom.append(box_list[i][3])
    
bbox = [min(left), min(bottom), max(right), max(top)]


straighten_leftover = """
                      CREATE TABLE power_lines_box AS
                      SELECT * FROM power_lines_orig
                      WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way)
                      AND abstracted IS NULL;
                      
                      CREATE TABLE power_lines_final AS
                      SELECT line_id, 
                             power, 
                    	     cables, 
                    	     voltage, 
                    	     wires, 
                    	     ST_MakeLine(ST_StartPoint(way), ST_EndPoint(way)) AS way, 
                    	     1 AS abstracted
                      FROM power_lines_box;
                      
                      INSERT INTO power_lines_final
                      SELECT line_id, power, cables, voltage, wires, way, abstracted
                      FROM power_lines_orig 
                      WHERE abstracted = 1;
                      
                      DROP TABLE power_lines_box;
                      
                      DELETE FROM power_lines_final WHERE way IS NULL;
                      """.format(bbox[0], bbox[1], bbox[2], bbox[3])

conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")
with conn.cursor() as cur:
    cur.execute(straighten_leftover)
    conn.commit()
conn.close()


print("Leftover lines straightened.")
print("Cumulative execution time:", time_convert(start_time, time.time()))




#%% SQL: construct final tables

conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")

# create final output tables, intersections is empty because they still need to be calculated
create_tables = """
                CREATE TABLE links AS
                SELECT line_id,
	                   ST_X( ST_Transform(ST_StartPoint(way), 4326) ) AS longitude_1,
                       ST_Y( ST_Transform(ST_StartPoint(way), 4326) ) AS latitude_1,
                       ST_X( ST_Transform(ST_EndPoint(way), 4326) ) AS longitude_2,
                       ST_Y( ST_Transform(ST_EndPoint(way), 4326) ) AS latitude_2,
                       power,
                       cables,
                       voltage,
                       wires
                FROM power_lines_final;
                
                CREATE TABLE vertices AS
                SELECT station_id,
                       ST_X( ST_Transform(ST_Centroid(way), 4326) ) AS longitude,
                       ST_Y( ST_Transform(ST_Centroid(way), 4326) ) AS latitude,
                       power,
                       voltage
                FROM power_stations_orig
                WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way);
                
                CREATE TABLE intersections (line_id int, station_id int);
                """.format(bbox[0], bbox[1], bbox[2], bbox[3])


# connect and run the above query
with conn.cursor() as cur:
    cur.execute(create_tables)
    conn.commit()    


# calculate intersections
counter = 0
for i in test_set:
    
    box = box_list[i]
    counter += 1
    
    # create the bounding box in the database using the box coordinates
    box_lines = """
                CREATE TABLE power_lines_box AS
                SELECT * FROM power_lines_final
                WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way);
                """.format(box[0], box[1], box[2], box[3])
    
    box_stations = """
                   CREATE TABLE power_stations_box AS
                   SELECT * FROM power_stations_orig
                   WHERE ST_Intersects(ST_MakeEnvelope({}, {}, {}, {}, 4326), way); 
                   """.format(box[0], box[1], box[2], box[3])
    
    # put new data back into main table
    insert_new = """
                 INSERT INTO intersections (line_id, station_id)
                 SELECT L.line_id, 
                        S.station_id
                 FROM power_lines_box L, power_stations_box S
                 WHERE ST_Intersects(ST_Endpoint(L.way), S.way)
                 OR ST_Intersects(ST_Startpoint(L.way), S.way);
                 """
    
    # reset database tables
    reset_data = """
                 DROP TABLE power_lines_box;
                 DROP TABLE power_stations_box;
                 """
    
    # execute the statements defined above
    with conn.cursor() as cur:
        cur.execute(box_lines)
        conn.commit()
        cur.execute(box_stations)
        conn.commit()
        cur.execute(insert_new)
        conn.commit()
        cur.execute(reset_data)
        conn.commit()
    
    # show completed boxes
    print("Number of boxes of intersections calculated:", counter)



#%% SQL: export

conn = create_connection(db_name, "postgres", "", "127.0.0.1", "5432")

# export final tables
export_tables = r"""
                    COPY links TO '{}\links.csv' DELIMITER ',' CSV HEADER;
                    COPY vertices TO '{}\vertices.csv' DELIMITER ',' CSV HEADER;
                    COPY intersections TO '{}\intersections.csv' DELIMITER ',' CSV HEADER;
                    """.format(output_path, output_path, output_path)

with conn.cursor() as cur:
    cur.execute(export_tables)
    conn.commit()


# reset database
reset_database = """
                 DROP TABLE intersections;
                 DROP TABLE links;
                 DROP TABLE vertices;
                 """
with conn.cursor() as cur:
    cur.execute(reset_database)
    conn.commit()


print("Algorithm complete.")
print("Cumulative execution time:", time_convert(start_time, time.time()))


# reset warnings
warnings.filterwarnings('default')


