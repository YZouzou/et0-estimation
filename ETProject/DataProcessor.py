import pandas as pd
import numpy as np
import os
import yaml
from ET0_functions import *

# DataProcessor class
#####################


class DataProcessor:
    """
    Loads the cleaned raw data files, merges them, removes missing values, and computes ET0.
    Years with missing month measurements are dropped.

    Parameters:
    -----------

    cleaned_data_dir_path: str
        Path to the directory containing the cleaned data

    var_abbr_config_path: str
        Path to the configuration file containing the user-defined variable abbreviations

    use_sun_duration: bool
        If True, uses sun duration to compute incoming radiation.
        Else, uses the measured incoming radiation.

    use_min_max_humidity: bool
        If True, the actual vapor pressure (ea) is computed using (RHmax, Tmin, RHmin, Tmax)
        Else, the actual vapor pressure (ea) is computed using (RHmax, Tmin) only

    ws_height: float
        Height at which the wind speed was measured.

    variables: list
        Variables to load from cleaned data dir and include in the merged dataset

    cols_to_keep: list
        List of the computed variables/parameters to include in the result dataframe.
        Possible values:
            ['psych_const', 'es', 'ea', 'sat_slope', 'Ra', 'Rso', 'N', 'inc_rad', 'Rns', 'Rnl', 'Rn', 'ET0']
    """

    def __init__(self,
                 cleaned_data_dir_path,
                 var_abbr_config_path,
                 variables,
                 use_sun_duration=False,
                 use_min_max_humidity=False,
                 ws_height=10,
                 cols_to_keep=['Ra', 'Rso', 'Rn', 'ET0']):

        self.cleaned_data_dir_path = cleaned_data_dir_path

        # Loading variable name abbreviations configuration file
        with open(var_abbr_config_path) as f:
            self.var_abbr = yaml.safe_load(f)['VAR_NAMES']  # Dictionary of variable_names: abbreviations

        self.use_sun_duration = use_sun_duration
        self.cols_to_keep = cols_to_keep
        self.use_min_max_humidity = use_min_max_humidity
        self.ws_height = ws_height
        self.variables = variables

    def load_data(self):

        var_file_paths = list(map(lambda x: os.path.join(self.cleaned_data_dir_path, x + '.csv'), self.variables))

        # Loading station definitions file
        st_def = pd.read_csv(os.path.join(self.cleaned_data_dir_path, 'station_definitions.csv')).drop(
            columns=['province'])
        merged_df = None

        # Merging datasets
        for path in var_file_paths:
            if merged_df is None:
                merged_df = pd.read_csv(path)
                continue

            df = pd.read_csv(path)
            merged_df = merged_df.merge(df, on=['st_num', 'year', 'month'], how='outer')

        # Merging station definition df with the merged dataset
        df = st_def.merge(merged_df, on='st_num', how='right')
        df = df.sort_values(by=['st_num', 'year', 'month'])

        return df

    def add_ET0(self, data):

        df = data.copy()

        # Converting the wind speed measured at h=10m
        # to the equivalent wind speed at height 2m
        if self.ws_height != 2:
            df['avg_ws'] = windspeed_2m(h=self.ws_height, data=df)

        # Converting solar radiation units
        # from cal/cm2 to MJ/m2
        if not self.use_sun_duration:
            df[self.var_abbr['average_solar_radiation']] = df[self.var_abbr[
                'average_solar_radiation']] * 4.184 * 10 ** (-2)

        # Replacing maximum relative humidity values greater than 100 with 100
        cond = df[self.var_abbr['maximum_relative_humidity']] > 100
        idx = df[cond].index
        df.loc[idx, self.var_abbr['maximum_relative_humidity']] = 100

        df = compute_ET0(
            min_temp=self.var_abbr['minimum_temperature'],
            max_temp=self.var_abbr['maximum_temperature'],
            avg_temp=self.var_abbr['average_temperature'],
            RHmin=self.var_abbr['minimum_relative_humidity'],
            RHmax=self.var_abbr['maximum_relative_humidity'],
            elev='elevation',
            latitude='latitude',
            month='month',
            inc_rad=self.var_abbr['average_solar_radiation'],
            sun_duration=self.var_abbr['average_sun_duration'],
            avg_ws=self.var_abbr['average_wind_speed'],
            st_num='st_num',
            year='year',
            G_method=1,
            maxmin=self.use_min_max_humidity,
            compute_inc_rad=self.use_sun_duration,
            cols_to_keep=self.cols_to_keep,
            data=df)

        return df

    def process_data(self, verbose=True):

        df = self.load_data()

        # Variables required for ET0 computation
        variables = ['maximum_temperature', 'minimum_temperature', 'maximum_relative_humidity', 'average_wind_speed']

        if self.use_min_max_humidity:
            variables += ['minimum_relative_humidity']

        if self.use_sun_duration:
            assert_variables = variables + ['average_sun_duration']
        else:
            assert_variables = variables + ['average_solar_radiation']

        # Number of datapoints before dropping missing values
        s1 = df.shape[0]

        txt = 'Cleaning dataset:\n'

        cond = np.isin(assert_variables, self.variables)

        assert all(cond), f"{np.array(assert_variables)[~cond]} variables should be included for ET0 computation"

        variables = ['elevation'] + [self.var_abbr[x] for x in variables]

        for variable in variables:
            txt += '    Dropped {} missing {} values\n'.format(df[variable].isna().sum(), variable)

        if self.use_sun_duration:
            var_name = self.var_abbr['average_sun_duration']
            # Dropping NaN and zero sun duration values
            cond = (df[var_name].isna()) | (df[var_name] == 0)
            dropped_txt = var_name

        else:
            var_name = self.var_abbr['average_solar_radiation']
            # Dropping instances with missing radiation
            cond = df[var_name].isna()
            dropped_txt = var_name

        df = df.drop(index=df[cond].index).reset_index(drop=True)

        df = df.dropna(subset=variables)

        txt += '    Dropped {} missing {} values\n'.format(cond.sum(), dropped_txt)

        # Dropping station-year combos with less than a full year readings
        # Grouping by station number and year
        st_year_group = df.groupby(['st_num', 'year']).count()

        # Finding the indexes of st-year combos with less than 12 measurements
        cond = st_year_group['month'] < 12
        index = st_year_group[cond].index

        # MI_df has a MultipleIndex of st_num and year
        MI_df = df.set_index(keys=['st_num', 'year'])
        txt += '    Dropped {} values from st-year combos with less than 12 vals/year\n' \
            .format(MI_df.loc[index, :].shape[0])

        MI_df = MI_df.drop(index=index)
        df = MI_df.reset_index()

        s2 = df.shape[0]
        txt += '----------------------------------\n'
        txt += 'Total dropped values: {}\n'.format(s1 - s2)
        txt += 'Dataset size: {}'.format(df.shape[0])
        if verbose == True:
            print(txt)

        # Sorting data
        df = df.sort_values(by=['st_num', 'year', 'month']).reset_index(drop=True)

        # Adding average temperature column
        # Average temperature used in ET0 computation should be the average
        # of the maximum and minimum temperatures rather than the monthly average
        col_loc = df.columns.get_loc(self.var_abbr['minimum_temperature']) + 1
        df.insert(
            col_loc,
            self.var_abbr['average_temperature'],
            (df[self.var_abbr['minimum_temperature']] + df[self.var_abbr['maximum_temperature']]) / 2.)

        # Adding ET0 and other computed variables
        df = self.add_ET0(df)

        return df