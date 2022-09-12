import pandas as pd
import numpy as np


# Reference evapotranspiration functions
########################################

# Psychrometric Constant
def psych_constant(elev='elevation', data=None):
    """
    Computes the psychrometric constant in [kPa.°C^(-1)]

    Parameters:
    ----------

    elev:
        Elevation above sea level in [m]

    data: pandas.DataFrame
        Dataframe containing station elevation
    """

    if data is not None:
        cols = [elev]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        elev = data[elev]

    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26

    return 0.665e-3 * P


# Saturation vapour pressure
def sat_vap_pressure(T='avg_temp', data=None):
    """
    Computes the saturation vapour pressure in [kPa]

    Parameters:
    ----------

    T:
        Air temperature in [°C]

    data: pandas.DataFrame
        Dataframe containing air temperature in [°C]
    """

    if data is not None:
        cols = [T]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        T = data[T]

    return 0.6108 * np.exp((17.27 * T) / (T + 237.3))


# Slope of Saturation Vapour Pressure Curve
def saturation_slope(T='avg_temp', data=None):
    """
    Computes the slope of the saturation vapour pressure curve
    at the given temperature in [kPa.°C^(-1)]

    Parameters:
    ----------

    T:
        Air temperature in [°C]

    data: pandas.DataFrame
        Dataframe containing air temperature in [°C]

    """

    if data is not None:
        cols = [T]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        T = data[T]

    return (4098 * sat_vap_pressure(T)) / (T + 237.3) ** 2


# Actual vapour pressure
def actual_vap_pressure(Tmin='min_temp',
                        RHmax='max_hum',
                        Tmax='max_temp',
                        RHmin='min_hum',
                        maxmin=True,
                        data=None):
    """
    Computes the actual vapour pressure in [kPa] using two methods
    where the parameter maxmin determines the method to be used

    1. maxmin = True
        Actual vapour pressure is computed using both maximum and
        minimum relative humidities

    2. maxmin = False
        Actual vapour pressure is calculated using only
        the maximum relative humidity

    Parameters:
    ----------

    Tmax, Tmin:
        Maximum and minimum temperatures in [°C]

    RHmax, RHmin:
        Maximum and minimum relative humidities as a percentage (Ex. 75)

    maxmin: bool
        Variable to determine which method to use.

    data: pandas.DataFrame
        Dataframe containing all the variables

    """

    if data is not None:
        cols = [Tmin, RHmax]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
        Tmin = data[Tmin]
        RHmax = data[RHmax]

        if maxmin == True:
            cols = [Tmax, RHmin]
            assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

            Tmax = data[Tmax]
            RHmin = data[RHmin]

    if maxmin == True:
        return (sat_vap_pressure(Tmin) * RHmax * 0.01 + sat_vap_pressure(Tmax) * RHmin * 0.01) / 2
    else:
        return sat_vap_pressure(Tmin) * RHmax * 0.01


# Extraterrestrial radiation
def extra_terr_rad(latitude='latitude', month='month', day=None, data=None):
    """
    Computes extraterrestrial radiation in [MJ.m^(-2).day^(-1)]
    given the latitude and day or month of the year.

    The equation for extraterrestrial radiation requires day of the year to be defined.
    When monthly values are used, the middle-day of the month is used as stated
    in Annex 2 Table 2.5 of the FAO ET0 calculation handbook.

    If "day" is defined "month" is ignored.

    Returns a series.

    Parameters:
    ----------

    latitude:
        Latitude in degrees.
        Latitude should be negative for southern hemisphere and positive for northern hemisphere.

    month:
        Month of the year.

    day:
        Day of the year.
        If given, month is ignored.

    data: pandas.DataFrame
        Dataframe containing latitude and month or day of the year values.

    """
    if data is not None:

        if day is not None:
            cols = [day, latitude]
            assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
            day = data[day]

        else:
            cols = [month, latitude]
            assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
            month = data[month]

        latitude = data[latitude]

    # Equation from Annex 2 Table 2.5 of the
    # FAO reference evapotranspiration calculation manual
    if day is not None:
        J = day
    else:
        try:
            # Check Annex 2 Table 2.5
            J = (30.4 * month - 15).astype(np.int32)
        except:
            J = int(30.4 * month - 15)

    # Convert latitude to radians
    phi = latitude * np.pi / 180

    # Inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)

    # Solar Decimation
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)

    # Sunset hour angle
    ws = np.arccos(-np.tan(phi) * np.tan(delta))

    # Solar constant
    Gsc = 0.082

    return (24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


# Clear-sky solar radiation
def clear_sky_rad(elev='elevation', ex_rad='Ra', data=None):
    """
    Computes the clear-sky solar radiation in
    [MJ.m^(-2).day^(-1)] given the station elevation and extraterrestrial radiation.

    Parameters:
    ----------

    elev:
        Station elevation in meters.

    ex_rad:
        Extraterrestrial radiation in [MJ.m^(-2).day^(-1)]

    data: pandas.DataFrame
        Dataframe containing station elevation and extraterrestrial radiation.

    """

    if data is not None:
        cols = [elev, ex_rad]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
        elev = data[elev]
        ex_rad = data[ex_rad]

    return (0.75 + 2e-5 * elev) * ex_rad


def daylight_hours(month='month', latitude='latitude', day=None, data=None):
    '''
    Compute daylight hours based on month/day of the year and latitude.
    Returns a series of daylight hours.

    Parameters:
    -----------

    month/day:
        Month/Day of the year.

    latitude:
        Latitude in degrees.
        Latitude should be negative for southern hemisphere and positive for northern hemisphere.

    data: pandas.DataFrame
        Dataframe containing month and latitude values


    '''

    if data is not None:
        if day is not None:
            cols = [day, latitude]
            assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
            day = data[day]
        else:
            cols = [month, latitude]
            assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"
            month = data[month]

        latitude = data[latitude]

    if day is not None:
        J = day
    else:
        try:
            # Check Annex 2 Table 2.5
            J = (30.4 * month - 15).astype(np.int32)
        except:
            J = int(30.4 * month - 15)

    # Convert latitude to radians
    phi = latitude * np.pi / 180

    # Solar Decimation
    delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)

    # Sunset hour angle
    ws = np.arccos(-np.tan(phi) * np.tan(delta))

    return 24 * ws / np.pi


def solar_rad(n='sun_duration', N='N', ex_rad='Ra', data=None):
    '''
    Get solar radiation from extraterrestrial radiation and sunshine duration.

    Parameters:
    -----------

    n:
        Measured sunlight duration in hours

    N:
        Daylight hours

    ex_rad:
        Extraterrestrial radiation in [MJ.m^(-2).day^(-1)]

    data: pandas.DataFrame
        Dataframe containing sunlight duration, daylight hours, and extraterrestrial radiation

    '''

    if data is not None:
        cols = [n, N, ex_rad]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        n = data[n]
        N = data[N]
        ex_rad = data[ex_rad]

    # Angstrom values as and bs
    a_angs = 0.25
    b_angs = 0.5

    return (a_angs + b_angs * (n / N)) * ex_rad


# Net shortwave (solar) radiation
def net_short_rad(Rs='inc_rad', data=None):
    """
    Computes the net shortwave radiation in [MJ.m^(-2).day^(-1)]
    using the ET0 reference crop albedo given the measured solar
    radiation

    Parameters:
    ----------

    Rs:
        Incoming solar radiation in [MJ.m^(-2).day^(-1)]

    data: pandas.DataFrame
        Dataframe containing all the variables

    """

    albedo = 0.23

    if data is not None:
        cols = [Rs]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        Rs = data[Rs]

    return (1 - albedo) * Rs


# Net longwave radiation
def net_long_rad(Tmax='max_temp',
                 Tmin='min_temp',
                 ea='ea',
                 Rs='inc_rad',
                 Rso='cs_rad',
                 data=None):
    """
    Computes the net longwave radiation in [MJ.m^(-2).day^(-1)]
    given maximum and minimum temperatures,
    actual vapour pressure, measured solar radiation,
    and clear-sky solar radiation

    Parameters:
    ----------

    Tmax, Tmin:
        Maximum and minimum temperatures in [°C]

    ea:
        Actual vapour pressure in [kPa]

    Rs:
        Incoming solar radiation in [MJ.m^(-2).day^(-1)]

    Rso:
        Clear-sky solar radiation in [MJ.m^(-2).day^(-1)]

    data: pandas.DataFrame
        Dataframe containing all the variables
    """

    if data is not None:
        cols = [Tmax, Tmin, ea, Rs, Rso]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        Tmax = data[Tmax]
        Tmin = data[Tmin]
        ea = data[ea]
        Rs = data[Rs]
        Rso = data[Rso]

    # Convert temperatures to kelvin
    Tmax_k = Tmax + 273.16
    Tmin_k = Tmin + 273.16

    # Stefan-Boltzmann constant
    sig = 4.903e-9

    return sig * ((Tmax_k ** 4 + Tmin_k ** 4) / 2) * (0.34 - 0.14 * np.sqrt(ea)) * (1.35 * Rs / Rso - 0.35)


# Wind speed measured at height 2m
def windspeed_2m(h, uz='avg_ws', data=None):
    """
    Computes the equivalent wind speed at a height 2m
    above the ground given the wind speed at a height h

    Parameters:
    ----------

    h: float
        Height of wind speed measurement device in [m]

    uz:
        Wind speed measured at a height h above the ground in [m/s]

    data: pandas.DataFrame
        Dataframe containing all the variables


    """

    if data is not None:
        cols = [uz]
        assert all(np.isin(cols, data.columns)), "Columns not found in the given dataframe"

        uz = data[uz]

    return uz * 4.87 / np.log(67.8 * h - 5.42)


def soil_heat_flux(method=1,
                   avg_temp='avg_temp',
                   year='year',
                   month='month',
                   st_num='st_num',
                   data=None):
    '''
    Compute soil heat flux (G) using both equations:
    * Method 1 (Eq. 1):
        G_i = 0.07 * (T_(i+1) - T_(i-1))

    * Method 2 (Eq. 2):
        G_i = 0.14 * (T_i - T_(i-1))

    There are three cases when computing G, they are defined by the following labels in the function:

        Label 0:
       - Measurement having the following and the previous measurements
         both available --> Method 1 & 2 apply.
    Possible cases:
           * Last year December measurement in a certain station.
           * December measurement of a year where the following year is not available (gap in time-series).

    Label 1:
       - Measurement having the previous measurement available but
         the following one not available --> Method 2 applies.

    Label 2:
       - Measurement missing the previous measurement --> Both equations do not apply.
         Solution: Use December measurement of the same year as the previous measurement.
         Possible cases:
            * First year January measurement of a certain station.
            * January measurement of a year where the previous year is not available (gap in time-series).


    Parameters:
    -----------

    data: pandas.DataFrame
        Dataframe containing average temperature, station numbers, years, and months

    method: int (1 or 2)
        Determines the equation to be used

    avg_temp/year/month/st_num: str
        Names of the columns containing the corresponding variables
        or series of the corresponding variables

    '''

    if data is not None:
        cols = np.array([avg_temp, year, month, st_num])
        cond = np.isin(cols, data.columns)

        assert all(cond), "Columns {} are missing or wrongly defined.".format(cols[cond])

        df = data[cols].copy()
    else:
        cols = [st_num, year, month, avg_temp]
        cond = list(map(lambda x: type(x) == pd.core.series.Series, cols))
        assert all(cond), "Enter data and column names or pandas.Series for each variable"
        df = pd.concat(cols, axis=1)

        # Getting column names
        st_num, year, month, avg_temp = df.columns

    df['day'] = 15
    df.insert(1, 'date', pd.to_datetime(df[[year, month, 'day']]))
    df = df.drop(columns=['day'])
    df = df.sort_values(by=[st_num, 'date']).reset_index(drop=True)

    if method == 1:
        df['G'] = df[avg_temp].diff(2).shift(-1) * 0.07

    elif method == 2:
        df['G'] = df[avg_temp].diff() * 0.14

    # Label 1
    # This case need only be checked when using method 1
    if method == 1:
        # Computing difference between each measurement and the following one
        df['date_diff'] = -1 * df['date'].diff(periods=-1)
        df['st_diff'] = df[st_num].diff(periods=-1)

        # Condition 1:
        #     - Same station as the following station
        #     - Date difference between the value and the following value exceeds 1 month
        cond1 = (df['date_diff'] > pd.Timedelta(days=32)) & (df['st_diff'] == 0)
        # Condition 2:
        #     - Different station from the following station
        cond2 = df['st_diff'] != 0

        cond = cond1 | cond2

        # Index of label 1 cases
        idx1 = df[cond].index
        idx2 = idx1 - 1
        idx = np.concatenate([idx1, idx2])
        idx = np.sort(idx)
        label_1 = df.loc[idx]
        label_1['G'] = label_1[avg_temp].diff() * 0.14
        label_1 = label_1.loc[idx1]
        df.loc[idx1, 'G'] = label_1['G']

    # Label 2
    # Computing difference between each measurement and the previous one
    df['date_diff'] = df['date'].diff()
    df['st_diff'] = df[st_num].diff()

    # Condition 1:
    #     - Same station as the previous station
    #     - Date difference between the value and the previous value exceeds 1 month
    cond1 = (df['date_diff'] > pd.Timedelta(days=32)) & (df['st_diff'] == 0)

    # Condition 2:
    #     - Different station from the previous station
    cond2 = df['st_diff'] != 0

    cond = cond1 | cond2

    # Index of label 2 cases
    idx1 = df[cond].index
    idx2 = idx1 + 11
    idx = np.concatenate([idx1, idx2])
    idx = np.sort(idx)
    label_2 = df.loc[idx]
    label_2['G'] = label_2[avg_temp].diff(-1) * 0.14
    label_2 = label_2.loc[idx1]
    df.loc[idx1, 'G'] = label_2['G']

    return df['G']


def compute_ET0(min_temp='min_temp',
                max_temp='max_temp',
                avg_temp='avg_temp',
                RHmin='min_hum',
                RHmax='max_hum',
                elev='elevation',
                latitude='latitude',
                month='month',
                inc_rad='inc_rad',
                sun_duration='sun_duration',
                G='G',
                avg_ws='avg_ws',
                st_num='st_num',
                year='year',
                G_method=1,
                maxmin=True,
                compute_inc_rad=True,
                day=None,
                cols_to_keep=['Ra', 'Rso', 'Rn', 'ET0'],
                data=None):
    '''
    Compute reference evapotranspiration from the given dataframe or numeric values.

    If a dataframe is given in "data" parameter, column names of the corresponding variables
    should be defined in the input. (Ex: max_temp = 'max_temp')

    Returns a dictionary of computed parameters/variables if the input is numeric.
    Returns a dataframe of the selected columns in "cols_to_keep" appended to the input dataframe
    if the input is a dataframe.

    Parameters:
    -----------

    min_temp/max_temp/avg_temp:
        Minimum, maximum, and average temperatures in Celsius.
        Average temperature should be computed as the average between the maximum and minimum
        temperatures of the studied period.

    RHmin/RHmax:
        Minimum and maximum relative humidities as percentages (Ex: 75).

    elev:
        Elevation of the studied area/station in meters above sea level.

    latitude:
        Latitude of the studied area/station in degrees.
        Positive for locations in the northern hemisphere and negative
        for the southern hemisphere.

    month:
        Month of the year.
        If day of the year is given, month is ignored.

    inc_rad:
        Measured incoming solar radiation.
        Ignored if compute_inc_rad is True.

    sun_duration:
        Sunshine duration in hours.
        Not required if the incoming radiation is given and compute_inc_rad is False.

    G:
        Soil heat flux (G).
        If given, G_method should be set to 0.

    avg_ws:
        Average wind speed at 2m above the ground in m/s.

    st_num:
        Station numbers.
        Only required if G is to be computed (G_method = 1 or 2)

    year:
        Year of the measurement.
        Only required if G is to be computed (G_method = 1 or 2)

    maxmin: bool
        If True, the actual vapor pressure (ea) is computed using (RHmax, Tmin, RHmin, Tmax)
        Else, the actual vapor pressure (ea) is computed using (RHmax, Tmin) only

    compute_inc_rad: bool
        Determines whether to compute incoming radiation from sunshine hours or not.
        If False, inc_rad should be given.

    day:
        Day of the year.
        If given, onth of the year is ignored.

    cols_to_keep:
        List of the computed variables/parameters to include in the result dataframe.
        Possible values:
            ['psych_const', 'es', 'ea', 'sat_slope', 'Ra', 'Rso', 'N', 'inc_rad', 'Rns', 'Rnl', 'Rn', 'ET0']

    data: pandas.DataFrame
        Dataframe containing the required variables.


    '''

    # Assertions
    #####################

    if data is not None:
        cols = np.array([min_temp, max_temp, avg_temp, RHmax, elev, latitude, month, avg_ws])
        cond = np.isin(cols, data.columns)
        assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])

        min_temp = data[min_temp]
        max_temp = data[max_temp]
        avg_temp = data[avg_temp]
        RHmax = data[RHmax]
        elev = data[elev]
        latitude = data[latitude]
        month = data[month]
        avg_ws = data[avg_ws]

        # Adding day if not None
        if day is not None:
            cols = [day]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            day = data[day]

        # Adding RHmin to assertion if maxmin is True
        if maxmin == True:
            cols = [RHmin]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            RHmin = data[RHmin]

        # Adding G, year, and st_num to assertion if G_method is 0
        if G_method != 0:
            cols = [st_num, year]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            year = data[year]
            st_num = data[st_num]

            cols_to_keep = ['G'] + cols_to_keep

        else:
            cols = [G]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            G = data[G]

        # Adding inc_rad to assertion if compute_inc_rad is False
        # Else, add sun_duration to assertion
        if compute_inc_rad == False:
            cols = [inc_rad]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            inc_rad = data[inc_rad]

        else:
            cols = [sun_duration]
            cond = np.isin(cols, data.columns)
            assert all(cond), "{} columns not found in the given dataframe".format(cols[~cond])
            sun_duration = data[sun_duration]

            cols_to_keep = ['N', 'inc_rad'] + cols_to_keep



    else:
        vals = [min_temp, max_temp, avg_temp, RHmax, elev, latitude, avg_ws]

        # Adding day if not None
        if day is not None:
            vals += [day]

        else:
            vals += [month]

        # Adding RHmin to assertion if maxmin is True
        if maxmin == True:
            vals += [RHmin]

        # Asserting G_method is not 0
        # G can't be computed without data
        assert G_method == 0, "Enter G value or data to compute G"

        # Adding inc_rad to assertion if compute_inc_rad is False
        # Else, add sun_duration to assertion
        if compute_inc_rad == False:
            vals += [inc_rad]
        else:
            vals += [sun_duration]

        cond = np.array(list(map(lambda x: (type(x) == int) or (type(x) == float), vals)))
        vals = np.array(vals)
        assert all(cond), '{} values are non-numerical. Enter "data" and column names or numerical values.'.format(
            vals[~cond])

    # Computing ET0
    ##########################
    result_dict = dict()

    # Psychrometric Constant
    psych_const = psych_constant(elev)
    result_dict['psych_const'] = psych_const

    # Mean Saturation Vapour Pressure
    es = (sat_vap_pressure(max_temp) + sat_vap_pressure(min_temp)) / 2.
    result_dict['es'] = es

    # Actual vapour pressure
    ea = actual_vap_pressure(Tmin=min_temp,
                             RHmax=RHmax,
                             Tmax=max_temp,
                             RHmin=RHmin,
                             maxmin=maxmin)
    result_dict['ea'] = ea

    # Slope of Saturation Vapour Pressure Curve
    sat_slope = saturation_slope(avg_temp)
    result_dict['sat_slope'] = sat_slope

    # Extraterrestrial radiation
    ex_rad = extra_terr_rad(latitude=latitude, month=month, day=day)
    result_dict['Ra'] = ex_rad

    # Clear-sky solar radiation
    cs_rad = clear_sky_rad(elev=elev, ex_rad=ex_rad)
    result_dict['Rso'] = cs_rad

    if compute_inc_rad == True:
        N = daylight_hours(month=month, day=day, latitude=latitude)
        result_dict['N'] = N

        # Filtering measured sun durations that are greater than N
        if data is not None:
            cond = sun_duration > N
            sun_duration.loc[cond] = N.loc[cond]
        else:
            if sun_duration > N:
                sun_duration = N

        inc_rad = solar_rad(n=sun_duration, N=N, ex_rad=ex_rad)
        result_dict['inc_rad'] = inc_rad

    # Filtering inc_rad not to exceed Rso (Rs/Rso < 1)
    if data is not None:
        cond = inc_rad > cs_rad
        inc_rad.loc[cond] = cs_rad.loc[cond].copy()
    else:
        if inc_rad > cs_rad:
            inc_rad = cs_rad

    # Net shortwave (solar) radiation
    ns_rad = net_short_rad(inc_rad)
    result_dict['Rns'] = ns_rad

    # Net longwave radiation
    nl_rad = net_long_rad(Tmax=max_temp,
                          Tmin=min_temp,
                          ea=ea,
                          Rs=inc_rad,
                          Rso=cs_rad)
    result_dict['Rnl'] = nl_rad

    # Net radiation Rn
    net_rad = ns_rad - nl_rad
    result_dict['Rn'] = net_rad

    # Computing G
    if G_method != 0:
        G = soil_heat_flux(method=G_method,
                           avg_temp=avg_temp,
                           year=year,
                           month=month,
                           st_num=st_num)
        result_dict['G'] = G

    # Reference Evapotranspiration ET0
    ET0 = \
        (0.408 * sat_slope * (net_rad - G) + psych_const * 900 * avg_ws * (es - ea) / (avg_temp + 273)) \
        / \
        (sat_slope + psych_const * (1 + 0.34 * avg_ws))

    result_dict['ET0'] = ET0

    if data is not None:
        df = pd.DataFrame(result_dict)

        return pd.concat([data, df[cols_to_keep]], axis=1)

    return result_dict

