rm(list = ls())

source("./R/getOnsetCessation.R")

### point-station calculation ----------------------------------------------------------------

pr_data <- zoo::read.zoo("./data/example_zoo.csv", 
                         header = T, 
                         format = "%Y-%m-%d", 
                         sep = ",")
est <- pr_data[, 3]

# setting hydrological year and removing "02-29" dates
est <- window(est, start = "1981-08-01", end = "2016-07-31")
est <- est[!(format(time(est), "%m-%d") %in% c("02-29"))]

# getting Onset Cessation dates
OC <- getOnsetCessation(zoo::coredata(est))
OC_dates <- zoo::index(est)[OC]


# length of wet season by OC (lengthOC)
lengthOC <- mapply(function(x, y){ y - x },
                  x = OC[seq(1, length(OC), 2)],
                  y = OC[seq(2, length(OC), 2)])
plot(lengthOC, type = "l")

# total precipitation by OC (PRCPTOT)
PRCPTOT <- mapply(function(x, y){ sum(est[x:y]) },
                     x = OC[seq(1, length(OC), 2)],
                     y = OC[seq(2, length(OC), 2)])
plot(PRCPTOT, type = "l")

# 1-day maximum precipitation by OC (Rx1day)
Rx1day <- mapply(function(x, y){ max(est[x:y]) },
                     x = OC[seq(1, length(OC), 2)],
                     y = OC[seq(2, length(OC), 2)])
plot(Rx1day, type = "l")



### gridded calculation ----------------------------------------------------------------

piscop <- raster::brick("./data/piscop_sample.nc")

# setting hydrological year and removing "02-29" dates
seq_ts = seq(as.Date("1981-01-01"), as.Date("2016-12-31"), by = "day")
piscop_ts = zoo::zoo(1:length(seq_ts), seq_ts)
piscop_ts = window(piscop_ts, start = "1981-08-01", end = "2016-07-31")
piscop_ts = piscop_ts[!(format(time(piscop_ts), "%m-%d") %in% c("02-29"))]

piscop = piscop[[zoo::coredata(piscop_ts)]] # this process is really slow!
piscop = piscop + 0                         # to get numeric data (bug)

# getting Onset Cessation dates
OC_gridded <- raster::calc(piscop, fun = getOnsetCessation)


# length of wet season by OC (lengthOC)
lengthOC_gridded <- raster::calc(OC_gridded, 
                                 fun = function(z){

                                   mapply(function(x, y){ 
                                     # some years can not be computed because:
                                     # 1) data is bad
                                     # 2) the year does not follow the seasonality
                                     # 3) an atypical year
                                     ifelse(y - x < 0, NA, y - x) },
                                          x = z[seq(1, length(z), 2)],
                                          y = z[seq(2, length(z), 2)])
                                   
                                 })

# spatial variability of the lengthOC in the year 1
sp::spplot(lengthOC_gridded[[1]])
