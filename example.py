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


piscop = xr.open_dataset("./data/piscop_sample.nc") # , chunks={"latitude": 10, "longitude": 10}

# deleting "02-29" dates, and daily climatology
piscop = piscop.isel(z = ~piscop.z.dt.strftime('%m-%d').isin("02-29"))


# hy_year = list(range(243, 365)) + list(range(0, 243)) # 1st 09
hy_year = list(range(243-31, 365)) + list(range(0, 243-31)) #1st 08
hy_year_init = "08-01"
daily_clim = xr.open_dataset(cdo.ydaymean(input = piscop))
daily_clim = daily_clim.sel(z = daily_clim.z[hy_year])

"""
daily_clim.isel(z = 0).p.plot()
exp = daily_clim.sel(longitude=-72, latitude=-6, method='nearest').p.values.flatten()
pd.Series(np.cumsum(exp - np.mean(exp))).plot()
"""


# wet onset and cessation climatology
def wSeason(obj):
    anom_cumsum = np.cumsum(obj - np.mean(obj))


return np.array([np.argmin(anom_cumsum), np.argmax(anom_cumsum), np.mean(obj)])

piscop_wseason = xr.apply_ufunc(wSeason,
                                daily_clim,
                                vectorize=True,
                                input_core_dims=[['z']],
                                output_core_dims=[["z2"]])

"""
# dates since 1st 09
piscop_wseason.p.isel(z2=1).plot() # i_wet
piscop_wseason.p.isel(z2=0).plot() # f_wet
# R_ave
piscop_wseason.p.isel(z2=2).plot()
"""


def onSet(zoo_ts, i_wet_gr, f_wet_gr, R_ave_gr):
    f_wet = daily_clim.z[int(i_wet_gr)] + np.timedelta64(45, 'D') + np.timedelta64(365, 'D')


i_wet = daily_clim.z[int(f_wet_gr)] - np.timedelta64(45, 'D')
#   return np.array(int(f_wet)/int(i_wet))

n_days = int(((f_wet - i_wet) / np.timedelta64(1, 'D')) + 1)

i_wet = str(i_wet.dt.strftime('%m-%d').values)
f_wet = str(f_wet.dt.strftime('%m-%d').values)

zoo_ts = pd.DataFrame.from_records([[i] for i in zoo_ts])
zoo_ts.columns = ["value"]
zoo_ts['date'] = pd.Series(np.datetime_as_string(piscop.z.values))
zoo_ts['date'] = pd.Series(np.datetime_as_string(piscop.z.values)).apply(lambda x: x[0:10])
zoo_ts['year'] = pd.Series(np.datetime_as_string(piscop.z.values)).apply(lambda x: int(x[0:4]))
zoo_ts['md'] = pd.Series(np.datetime_as_string(piscop.z.values)).apply(lambda x: x[5:10])

ini = str(zoo_ts["year"].min()) + "-" + i_wet
fin = str(zoo_ts["year"].max()) + "-" + f_wet

ini = zoo_ts.loc[zoo_ts["date"] == ini].index[0]
fin = zoo_ts.loc[zoo_ts["date"] == fin].index[0] + 1

zoo_ts = zoo_ts.iloc[ini:fin]
zoo_ts = zoo_ts.reset_index()
# zoo_ts = zoo_ts.loc[zoo_ts["date"].strftime('%m-%d').isin(zoo_ts.index[0:n_days].strftime('%m-%d'))]
# zoo_ts = np.split(zoo_ts, len(zoo_ts)/n_days)
return np.array(n_days)
# res = [getDateCessation(i, R_ave=float(R_ave_gr)) for i in zoo_ts]


#
#


"""
   #zoo_ts.index = piscop.z.values

   zoo_ts = zoo_ts.loc[str(zoo_ts.index.year.min())+"-"+i_wet:str(zoo_ts.index.year.max())+"-"+f_wet]
   zoo_ts = zoo_ts.loc[zoo_ts.index.strftime('%m-%d').isin(zoo_ts.index[0:n_days].strftime('%m-%d'))]
   zoo_ts = np.split(zoo_ts, 35)
   res = [getDateCessation(i, R_ave = R_ave) for i in zoo_ts]
   return np.array(res)
"""

"""
zoo_ts = piscop.sel(longitude=-72, latitude=-6, method='nearest').p.values.flatten()
i_wet_gr = piscop_wseason.isel(z2=1).sel(longitude=-72, latitude=-6, method='nearest').p.values
f_wet_gr = piscop_wseason.isel(z2=0).sel(longitude=-72, latitude=-6, method='nearest').p.values
R_ave_gr = piscop_wseason.isel(z2=2).sel(longitude=-72, latitude=-6, method='nearest').p.values
"""


def getDateCessation(x, R_ave):
    date_res = x.index[np.cumsum(x["value"] - R_ave).argmax()]


hy_year = np.datetime64(date_res.strftime("%Y") + "-" + hy_year_init) - np.timedelta64(365, 'D')
date_res = np.datetime64(date_res)
return int((date_res - hy_year) / np.timedelta64(1, 'D'))


def getDateOnset(x, R_ave):
    date_res = x.index[np.cumsum(x["value"] - R_ave).argmin()]


hy_year = np.datetime64(date_res.strftime("%Y") + "-" + hy_year_init)
date_res = np.datetime64(date_res)
return int((date_res - hy_year) / np.timedelta64(1, 'D'))


def getTotalPrecipitation(x, R_ave):
    date_res_fin = np.cumsum(x["value"] - R_ave).argmax()


date_res_ini = np.cumsum(x["value"] - R_ave).argmin()
return x[date_res_ini:date_res_fin].sum()

res_res = xr.apply_ufunc(onSet,
                         piscop.p,
                         piscop_wseason.isel(z2=0),
                         piscop_wseason.isel(z2=1),
                         piscop_wseason.isel(z2=2),
                         input_core_dims=[['z'], [], [], []],
                         vectorize=True,
                         output_dtypes=['float64'],
                         dask="parallelized")
res_res.p.plot()

np.arange(i_wet.values,
          f_wet.values,
          np.timedelta64(1, 'D'),
          dtype='datetime64')

np.array()
t = datetime.now()
daily_clim.z.values[0].astype(datetime)
[1472731200000000000].to_datetime
datetime.utcfromtimestamp(1472731200000000000)
time.mktime(t1)
xr.apply_ufunc(special_mean,
               daily_clim,
               vectorize=True,
               input_core_dims=[['time']],
               output_core_dims=[["time2"]],
               output_sizes={"time2": 365},
               output_dtypes=[np.float],
               dask='parallelized')

import multiprocessing


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
      Like numpy.apply_along_axis(), but takes advantage of multiple
      cores.
      """


# Effective axis where apply_along_axis() will be applied by each
# worker (any non-zero axis number would work, so as to allow the use
# of `np.array_split()`, which is only done on axis 0):
effective_axis = 1 if axis == 0 else axis
if effective_axis != axis:
    arr = arr.swapaxes(axis, effective_axis)

# Chunks for the mapping (only a few chunks):
chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
          for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

pool = multiprocessing.Pool()
individual_results = pool.map(unpacking_apply_along_axis, chunks)
# Freeing the workers:
pool.close()
pool.join()

return np.concatenate(individual_results)

piscop = piscop.isel(time=~piscop.time.dt.strftime('%m-%d').isin("02-29"))
res = (('time', 'latitude', 'longitude'), np.apply_along_axis(special_mean, 0, piscop["precp"]))
xr.DataArray(res).isel(time=0)
res2 = (('time', 'latitude', 'longitude'), parallel_apply_along_axis(special_mean, 0, piscop["precp"].values))


def magnitude(a, b):
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)


return xr.apply_ufunc(func, a, b)


def special_mean(x):
    cars = {'Date': piscop.time.dt.strftime('%Y-%m-%d').values.tolist(),
            'Value': x
            }


df = pd.DataFrame(cars, columns=['Date', 'Value'])
df["md"] = df.apply(lambda x: x["Date"][5:10], axis=1)

return df.groupby("md").mean().values.flatten()

df.groupby("md").mean().values.flatten()
np.arange(1, 10).shape

res = xr.apply_ufunc(special_mean, piscop, vectorize=True)

xr.apply_ufunc(special_mean, piscop.precp, input_core_dims=[["time"]])

piscop.precp.mean(axis=0)

ny, nx = 100, 100
nt = 44
data = xr.DataArray(np.random.randn(nt, ny, nx),
                    dims=['time', 'y', 'x'],
                    name='blue reflectance')

rmin, rmax, nbins = -4, 4, 50
nbins
bins = np.linspace(rmin, rmax, nbins)

data_digitized = xr.apply_ufunc(np.digitize, data, kwargs={'bins': bins})


def special_mean(x):
    return np.arange(1, 10)


special_mean(x=10)

res = xr.apply_ufunc(special_mean,
                     piscop.p,
                     vectorize=True,
                     input_core_dims=[['time']],
                     output_core_dims=[["time2"]],
                     output_sizes={"time2": 365},
                     output_dtypes=[np.float],
                     dask='parallelized')

res.to_netcdf("adrian.nc")

import xarray as xr
import pandas as pd
import numpy as np
from cdo import *

cdo = Cdo()

piscop = xr.open_dataset("./data/piscop_sample.nc")  # , chunks={"latitude": 10, "longitude": 10}

# deleting "02-29" dates, and daily climatology
piscop = piscop.isel(z=~piscop.z.dt.strftime('%m-%d').isin("02-29"))

import pandas as pd
import numpy as np

pr_data = pd.read_csv("./data/example_zoo.csv", index_col=0)
pr_data.index = pd.to_datetime(pr_data.index)

est = pr_data.est1.loc["1981-08-01":"2016-07-31"]  # setting hydrological year
est = est[~est.index.strftime('%m-%d').isin(["02-29"])]  # deleting "02-29" dates

est = est.values
est.shape = (int(len(est) / 365), 365)
np.apply_along_axis(np.mean, 1, est)

pd.Series(np.apply_along_axis(np.mean, 0, est)).plot()