# Raw Data from Turkish Meteorological Service

**Read files using:** `pd.read_csv(path, delimiter = '|')`


## Station definitions
**File name**: *station_definitions*

* Contains station numbers, names, cities, provinces, longitiude, and latitude.
* There is a total of 1298 stations
* Station numbers are unique
* There are 4 negative elevation values
* Names have ? signs in places of Turkish letters


## Temperature

* Temperatures are in Celsius
* Month 13 is the year average

#### File names:
* *monthly_max_temperature_(C)*
* *monthly_min_temperature_(C)*
* *monthly_avg_temperature_(C)*


## Wind speed

* Values in $m/s$

#### File names:
* *monthly_avg_wind_speed*

## Relative humidity

#### File names:
* *monthly_max_relative_humidity*
* *monthly_min_relative_humidity*
* *monthly_avg_relative_humidity*


## Global solar radiation monthly total

* Values in $kWh/m^2$
* Dataset contains only 15772 values
* Measurements start from 2005

#### File names:
* *monthly_total_global_solar_radiation*



## Global solar radiation monthly maximum

* Values in $W/m^2$
* Dataset contains only 15667 values
* Measurements start from 2005

#### File names:
* *monthly_max_global_solar_radiation*



## Solar radiation monthly average

* Values in $cal/cm^2$
* Dataset has only 50968 values
* Measurements date back to 1965

#### File names:
* *monthly_avg_solar_radiation*



## Sunlight duration

* Values in hours

#### File names:
* *monthly_total_sun_duration*
* *monthly_avg_sun_duration*