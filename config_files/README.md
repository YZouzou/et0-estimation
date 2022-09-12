# Configuration Files

## variable_abbreviations.yaml
This file contains the definitions of variable abbreviations to ensure a consistent use of these abbreviations in all parts of the project.

## data_cleaning_config.yaml
This configuration file is used in the **clean_raw_data.py** script. It includes the following definintions:
* **PATHS**: The read path which contains the raw data and the write path where the cleaned data is to be saved.
* **VAR_FILES**: The variable/measurement name contained in each file as in *file_name: variable_name*. (Data was provided as one file for each measured variable).
* **ST_DEF_FILE**: The name of the file containing the station properties (IDs, locations, elevations, etc.)

## data_processing_config.yaml
This configuration file is used in the **process_data.py** script. It includes the following definintions:
* **DATA_DIR_PATH**: The path containing the cleaned data to be loaded by the script.
* **SAVE_PATH**: The path (and file name) of the processed data file to be saved.
* **USE_MIN_MAX_HUMIDITY**: If USE_MIN_MAX_HUMIDITY=True, the actual vapor pressure (ea) is computed using (RHmax, Tmin, RHmin, Tmax) Else, the actual vapor pressure (ea) is computed using (RHmax, Tmin) only
* **VARIABLES**: Variables to load from cleaned data dir and include in the merged dataset

## input_combinations.yaml
This configuration file includes the definition of input combinations used to create models for ET0 estimation.