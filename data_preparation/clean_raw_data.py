import pandas as pd
import re
from pathlib import Path
import os
import yaml


def clean_station_definition_data(read_path, write_path):
    """
    Reads the station definition file from read_path, applies the cleaning procedure,
    and writes the cleaned dataset to write_path.

    Parameters:
    -----------

    read_path, write_path: str
        Paths to read and write data
    """

    df = pd.read_csv(read_path, sep='|')

    # Renaming column names from Turkish to English
    rename_dict = {
        '?stasyon Numaras?': 'st_num',
        '?stasyon Ad?': 'st_name',
        '?l': 'province',
        '?l?e': 'district',
        'Enlem': 'latitude',
        'Boylam': 'longitude',
        'Rak?m': 'elevation'
    }

    df = df.rename(columns=rename_dict)

    # Renaming province names with missing letters
    rename_dict = {
        'D?zce': 'Düzce',
        'Bart?n': 'Bartın',
        '?stanbul': 'İstanbul',
        'K?rklareli': 'Kırklareli',
        'Karab?k': 'Karabük',
        '?ank?r?': 'Çankırı',
        '?orum': 'Çorum',
        'G?m??hane': 'Gümüşhane',
        'A?r?': 'Ağrı',
        'I?d?r': 'Iğdır',
        '?anakkale': 'Çanakkale',
        'Bal?kesir': 'Balıkesir',
        'Eski?ehir': 'Eskişehir',
        'K?r?kkale': 'Kırıkkale',
        'K?tahya': 'Kütahya',
        'K?r?ehir': 'Kırşehir',
        '?zmir': 'İzmir',
        'Nev?ehir': 'Nevşehir',
        'Elaz??': 'Elazığ',
        'Bing?l': 'Bingöl',
        'Mu?': 'Muş',
        'Ayd?n': 'Aydın',
        'Ni?de': 'Niğde',
        'Kahramanmara?': 'Kahramanmaraş',
        'Ad?yaman': 'Adıyaman',
        'Diyarbak?r': 'Diyarbakır',
        'Mu?la': 'Muğla',
        '?anl?urfa': 'Şanlıurfa',
        'Tekirda?': 'Tekirdağ',
        '??rnak': 'Şırnak',
        'U?ak': 'Uşak'
    }

    df['province'] = df['province'].str.strip()
    df['province'] = df['province'].replace(rename_dict)

    # Removing unnecessary columns
    df = df.drop(columns=['st_name', 'district'])
    df.to_csv(write_path, index=False)

def clean_variable_data(read_path, write_path, var_name):
    """
    Reads the variable dataset from read_path, applies the cleaning procedure,
    and writes the cleaned dataset to the defined write_path.

    Parameters:
    -----------

    read_path, write_path: str
        Paths to read and write data

    var_name: str
        Name of the measured variable. This will be used as the column name of this variable.
    """

    df = pd.read_csv(read_path, sep='|')

    rename_dict = {'Istasyon_No': 'st_num', 'Istasyon_Adi': 'st_name', 'YIL': 'year', 'AY': 'month',
                   df.columns[-1]: var_name}

    # Renaming columns
    df = df.rename(columns=rename_dict)

    # Removing unnecessary columns
    df = df.drop(columns=['st_name'])

    # Dropping month 13 data if available
    cond = df['month'] == 13
    idx_to_drop = df.loc[cond].index
    df = df.drop(index=idx_to_drop)

    # Sorting data
    df = df.sort_values(by=['st_num', 'year', 'month']).reset_index(drop=True)

    # Writing cleaned dataset
    df = df.to_csv(write_path, index=False)


def clean_data(data_cleaning_config_path, var_abbr_config_path):
    """
    Cleans the raw data based on the file names defined in the data_cleaning_config.yaml
    configuration file and using the abbreviations defined in variable_abbreviations.yaml.
    """

    # Loading variable name abbreviations configuration file
    with open(var_abbr_config_path) as f:
        var_abbr = yaml.safe_load(f)['VAR_NAMES']

    # Loading data cleaning configuration file
    with open(data_cleaning_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    read_dir = config_data['PATHS']['read_dir']
    write_dir = config_data['PATHS']['write_dir']

    # Cleaning station_definitions file
    file_name = config_data['ST_DEF_FILE']['name']
    if file_name:
        read_path = os.path.join(read_dir, file_name)
        write_path = os.path.join(write_dir, re.sub('\.txt$', '.csv', file_name))
        clean_station_definition_data(read_path, write_path)

    # Cleaning measured variable files
    for file_name, var_name in config_data['VAR_FILES'].items():
        read_path = os.path.join(read_dir, file_name)
        write_path = os.path.join(write_dir, var_name + '.csv')
        clean_variable_data(read_path, write_path, var_abbr[var_name])


# Running the cleaning code based on the variables defined in the configuration file data_cleaning_config.yaml
data_cleaning_config_path = 'config_files/data_cleaning_config.yaml'
var_abbr_config_path = 'config_files/variable_abbreviations.yaml'

if __name__ == '__main__':

    # Navigating to the main directory
    main_dir_path = Path.cwd().parents[0]
    os.chdir(main_dir_path)

    assert os.path.exists(data_cleaning_config_path),\
        f"The data cleaning configuration file was not found at {data_cleaning_config_path}"

    assert os.path.exists(var_abbr_config_path), \
        f"The variable abbreviations configuration file was not found at {var_abbr_config_path}"

    clean_data(data_cleaning_config_path, var_abbr_config_path)
