import xarray as xr

piscop = xr.open_dataset("/home/adrian/Documents/wa_budyko_datasets/netcdf/P/PPISCOpd.nc")
piscop.isel(z = 0).p.plot()

min_lon = -72.5
min_lat = -17
max_lon = -70
max_lat = -15

mask_lon = (piscop.longitude >= min_lon) & (piscop.longitude <= max_lon)
mask_lat = (piscop.latitude >= min_lat) & (piscop.latitude <= max_lat)

piscop_sample = piscop.where(mask_lon & mask_lat, drop = True)
piscop_sample.isel(z=0).p.plot()
piscop_sample.to_netcdf("./data/piscop_sample.nc")