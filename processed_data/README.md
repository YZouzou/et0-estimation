# Processed Data

### et_data.csv
This file contains the meteorological measurements along with the variables computed using the FAO 56 PM equations [link](https://www.fao.org/3/x0490e/x0490e06.htm#chapter%202%20%20%20fao%20penman%20monteith%20equation) for ET0 calculation. This file is created using the **process_data.py** script found in the *data_preparation/* directory.

### station_definitions.csv
This file contains the station properties, their regions as assigned in **region_definition.ipynb**, and their datasets (train or test) as assigned in **train_test_split.ipynb**. Both notebooks, **region_definition.ipynb** and **train_test_split.ipynb** are found in the *data_preparation/* directory.

### map_data
This directory contains the geojson file used to draw the region boundaries.


