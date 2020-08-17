import pandas as pd
import numpy as np
import xarray as xr

exec(open("./python/getOnsetCessation.py").read())

### point-station calculation ----------------------------------------------------------------

pr_data = pd.read_csv("./data/example_zoo.csv", index_col = 0)
pr_data.index = pd.to_datetime(pr_data.index)

# setting hydrological year and removing "02-29" dates
est = pr_data.est3.loc["1981-08-01":"2016-07-31"]
est = est[~est.index.strftime('%m-%d').isin(["02-29"])]

# getting Onset Cessation dates
OC = getOnsetCessation(x=est)
OC_dates = est.index[OC]

to_map_x = OC[np.arange(0, len(OC), 2)]
to_map_y = OC[np.arange(1, len(OC), 2)]

# length of wet season by OC (lengthOC)
lengthOC = [y - x for x, y in zip(to_map_x, to_map_y)]
pd.Series(lengthOC).plot()

# total precipitation by OC (PRCPTOT)
PRCPTOT = [np.sum(est[x:y]) for x, y in zip(to_map_x, to_map_y)]
pd.Series(PRCPTOT).plot()

# 1-day maximum precipitation by OC (Rx1day)
Rx1day = [np.max(est[x:y]) for x, y in zip(to_map_x, to_map_y)]
pd.Series(Rx1day).plot()


### gridded calculation ----------------------------------------------------------------

piscop = xr.open_dataset("./data/piscop_sample.nc")

# setting hydrological year and removing "02-29" dates
piscop = piscop.sel(z=slice("1981-08-01", "2016-07-31"))
piscop = piscop.isel(z=~piscop.z.dt.strftime('%m-%d').isin("02-29"))

# getting Onset Cessation dates
OC_gridded = xr.apply_ufunc(getOnsetCessation,
                            piscop.p,
                            input_core_dims=[['z']],
                            output_core_dims=[["z2"]],
                            vectorize=True,
                            output_dtypes=['float64'])

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
                                  input_core_dims=[["z2"]],
                                  output_core_dims=[["z3"]],
                                  output_dtypes=['float64'])

# spatial variability of the lengthOC in the year 1
lengthOC_gridded.isel(z3=0).plot()


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
                                 piscop.p,
                                 vectorize = True,
                                 input_core_dims=[["z2"], ["z"]],
                                 output_core_dims=[["z3"]],
                                 output_dtypes=['float64'])

# spatial variability of the PRCPTOT in the year 1
PRCPTOT_gridded.isel(z3=0).plot()