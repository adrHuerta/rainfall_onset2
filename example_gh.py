import pandas as pd
import numpy as np
import xarray as xr

exec(open("./python/getOnsetCessation.py").read())

### gridded calculation ----------------------------------------------------------------

chirps = xr.open_dataset("./data/chirps_gh.nc")

# setting hydrological year [here is not needed to set the hydrological year as the rainy season is within a year] and removing "02-29" dates
chirps = chirps.sel(time=slice("1981-01-01", "2016-12-31"))
chirps = chirps.isel(time=~chirps.time.dt.strftime('%m-%d').isin("02-29"))

# getting mean Onset Cessation dates (climate normal)
meanOC_gridded = xr.apply_ufunc(getMeanOnsetCessation,
                                chirps.precip,
                                input_core_dims=[['time']],
                                output_core_dims=[["time2"]],
                                vectorize=True,
                                output_dtypes=['float64'])

meanOC_gridded.isel(time2=0).plot() # mean numeric onset (since 1st January)
meanOC_gridded.isel(time2=1).plot() # mean numeric cessation (since 1st January)
meanOC_gridded.isel(time2=2).plot() # mean daily rainfall
meanOC_gridded.isel(time2=[0,1]).plot(x="longitude", y="latitude",col="time2")

# getting Onset Cessation dates
OC_gridded = xr.apply_ufunc(getOnsetCessation,
                            chirps.precip,
                            input_core_dims=[['time']],
                            output_core_dims=[["time2"]],
                            vectorize=True,
                            output_dtypes=['float64'])

# in OC_gridded the onset/cessation date is presented as numeric value (since 1st January 1981)
# time2 = 0 -> onset of 1981 / time2 = 1 -> cessation of 1981
# time2 = 2 -> onset of 1982 / time2 = 3 -> cessation of 1982 and so on... til 2016
OC_gridded.isel(time2=0).plot()
OC_gridded.isel(time2=1).plot()
OC_gridded.isel(time2=2).plot()
OC_gridded.isel(time2=3).plot()

# as OC_gridded have the onset/cessation in a continuous numeric value (since 1st January 1981)
# here I converted to a numeric value for each year (since 1st January for each year)

def date2num_func(z):
    date2num = np.tile(np.arange(0, 365, dtype="float"), 2016-1981+1)
    if np.isnan(z).any():
        res = z
    else:
        res = np.take(date2num, z.astype(int))
    return res


OC_gridded_date2num = xr.apply_ufunc(date2num_func,
                                     OC_gridded,
                                     input_core_dims=[['time2']],
                                     output_core_dims=[['time2']],
                                     vectorize=True,
                                     output_dtypes=['float64'])

OC_gridded_date2num.isel(time2=0).plot()
OC_gridded_date2num.isel(time2=1).plot()
OC_gridded_date2num.isel(time2=2).plot() # note the difference between OC_gridded
OC_gridded_date2num.isel(time2=3).plot()

OC_gridded_date2num.isel(time2=[0,1]).plot(x="longitude", y="latitude",col="time2")


# getting onset and cessation numeric values in different objects
OC_gridded_onset = OC_gridded_date2num.sel(time2 = OC_gridded_date2num.time2[range(0, (2016-1981+1)*2, 2)]) # odd numbers
OC_gridded_onset["time2"] = range(1981,2017)
OC_gridded_cessation = OC_gridded_date2num.sel(time2 = OC_gridded_date2num.time2[range(1, (2016-1981+1)*2, 2)]) # even numbers
OC_gridded_cessation["time2"] = range(1981,2017)

# example of time series for a specific grid
# one major season
exp0 = (OC_gridded_onset).sel(longitude=-1, latitude=10, method='nearest').to_dataframe().drop(['longitude', 'latitude'], axis=1)
exp1 = (OC_gridded_cessation).sel(longitude=-1, latitude=10, method='nearest').to_dataframe().drop(['longitude', 'latitude'], axis=1)
pd.merge(exp0, exp1, left_index=True, right_index=True).rename({'precip_x': 'onset', 'precip_y': 'cessation'}, axis=1).plot(xlabel="Year", ylabel="Numeric value since 1st January (0-364)")

# two seasons? (based on the figure of https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JD025428)
exp0 = (OC_gridded_onset).sel(longitude=-1, latitude=6, method='nearest').to_dataframe().drop(['longitude', 'latitude'], axis=1)
exp1 = (OC_gridded_cessation).sel(longitude=-1, latitude=6, method='nearest').to_dataframe().drop(['longitude', 'latitude'], axis=1)
pd.merge(exp0, exp1, left_index=True, right_index=True).rename({'precip_x': 'onset', 'precip_y': 'cessation'}, axis=1).plot(xlabel="Year", ylabel="Numeric value since 1st January (0-364)")


# length of wet season by OC (lengthOC)

def lengthOC_func(z):
    map_in_x = z[np.arange(0, len(z), 2)]
    map_in_y = z[np.arange(1, len(z), 2)]
    # some years can not be computed because:
    # 1) data is bad
    # 2) the year does not follow the seasonality
    # 3) an atypical year
    lengthOC = [np.where(y - x < 0, np.nan, y - x) for x, y in zip(map_in_x, map_in_y)]

    return np.array(lengthOC)

lengthOC_gridded = xr.apply_ufunc(lengthOC_func,
                                  OC_gridded,
                                  vectorize = True,
                                  input_core_dims=[["time2"]],
                                  output_core_dims=[["time3"]],
                                  output_dtypes=['float64'])

# spatial variability of the lengthOC in the year 1
lengthOC_gridded.isel(time3=0).plot()

# total precipitation by OC (PRCPTOT)

def PRCPTOT_func(z, total_p):
    map_in_x = z[np.arange(0, len(z), 2)]
    map_in_y = z[np.arange(1, len(z), 2)]
    # some years can not be computed because:
    # 1) data is bad
    # 2) the year does not follow the seasonality
    # 3) an atypical year
    PRCPTOT = [np.where(y - x < 0, np.nan, np.sum(total_p[int(x):int(y)])) for x, y in zip(map_in_x, map_in_y)]

    return np.array(PRCPTOT)

PRCPTOT_gridded = xr.apply_ufunc(PRCPTOT_func,
                                 OC_gridded,
                                 chirps.precip,
                                 vectorize = True,
                                 input_core_dims=[["time2"], ["time"]],
                                 output_core_dims=[["time3"]],
                                 output_dtypes=['float64'])

# spatial variability of the PRCPTOT in the year 1
PRCPTOT_gridded.isel(time3=0).plot()