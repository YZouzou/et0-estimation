import pandas as pd
import numpy as np
from pathlib import Path
import os
import yaml

# Changing to main directory
main_dir_path = Path.cwd().parents[0]
os.chdir(main_dir_path)

# Importing ET0 functions and DataProcessor class
from ETProject.ET0_functions import *
from ETProject.DataProcessor import DataProcessor


# Running data processing code
config_file_path = 'config_files/data_processing_config.yaml'
if __name__ == '__main__':

    with open(config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    data_processor = DataProcessor(
        cleaned_data_dir_path=config_data['DATA_DIR_PATH'],
        var_abbr_config_path='config_files/variable_abbreviations.yaml',
        variables=config_data['VARIABLES'],
        use_sun_duration=config_data['USE_SUN_DURATION'],
        use_min_max_humidity=config_data['USE_MIN_MAX_HUMIDITY'],
        ws_height=config_data['WS_HEIGHT'])

    df = data_processor.process_data()

    # Loading variable abbreviation config file
    with open('config_files/variable_abbreviations.yaml', 'r') as f:
        var_abbr = yaml.safe_load(f)['VAR_NAMES']

    # Filling missing average humidity data
    avg_hum = var_abbr['average_relative_humidity']
    max_hum = var_abbr['maximum_relative_humidity']
    min_hum = var_abbr['minimum_relative_humidity']

    df[avg_hum] = df[avg_hum].fillna((df[min_hum] + df[max_hum]) / 2.)

    # Saving dataframe
    df.to_csv(config_data['SAVE_PATH'], index=False)