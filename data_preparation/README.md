# Data preparation
This section includes the code used for cleaning and preparing the raw weather data received from the meteorological services.

## File description:
* **data_examination.ipynb**: In this notebook, the raw data was examined to determine the required cleaning procedure.

* **clean_raw_data.py**: This script applies the cleaning procedure on the raw data found in *raw_data/* directory and saves the cleaned data files in *cleaned_data/* directory. The script uses variables defined in the configuration files **data_cleaning_config.yaml** and **variable_abbreviations.yaml** found in *config_files/* directory.

* **ET0_calculation_details.ipynb**: A description of the ET0 calculation procedure based on the [FAO Paper 56](https://www.fao.org/3/x0490e/x0490e07.htm#TopOfPage).

* **process_data.py**: This script merges the measured variable files in the *cleaned_data/* directory into one dataset, drops missing values, and computes reference evapotranspiration (ET0) as described in **ET0_calculation_details.ipynb**. The processed dataset (**et_data.csv**) is saved in *processed_data/* directory. The script uses variables defined in the configuration files **data_processing_config.yaml** and **variable_abbreviations.yaml** found in *config_files/* directory.
   * **Note:** Only the variables required for ET0 computation will have their missing values dropped, if any other variables are included in the dataset the code required to impute/drop the missing values of the additional variables should be added to this script.

* **region_definition.ipynb**: In this notebook, each station is assigned to one of Turkey's seven regions, i.e., Mediterranean Sea, Aegean Sea, Marmara, Black Sea, Central Anatolia, Eastern Anatolia, and Southeastern Anatolia. The **station_definitions.csv** file is then updated with the region definitions and saved in *processed_data/* directory.

* **train_test_split.ipynb**: In this notebook, stations are split into test and training stations. The **station_definitions.csv** and **et_data.csv** files are then updated with the station assignments and saved in *processed_data/* directory.
