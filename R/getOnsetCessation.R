getOnsetCessation <- function(x, wInd = 45)
  {

  ## daily climatology
  daily_clim <- matrix(x, byrow = T, ncol = 365)
  daily_clim <- apply(daily_clim, 2, mean)

  ## climatology onset and cessation dates
  R_ave <- mean(daily_clim)
  d <- cumsum(daily_clim - R_ave)
  i_wet_clim <- match(min(d), d)
  f_wet_clim <- match(max(d), d)

  ## getting onset and cessation dates by hydrological year
  do.call(c,
          lapply(split(x, ceiling(seq_along(x)/365)),
                 function(y){
                   i_wet = ifelse(i_wet_clim - wInd < 0, 1, i_wet_clim - wInd) # wInd can be too much
                   f_wet = ifelse(f_wet_clim + wInd > 365, 365, f_wet_clim + wInd) # in some pixels
                   d_year <- cumsum(y[i_wet:f_wet] - R_ave)
                   onset_year <- match(min(d_year), d_year)
                   cessation_year <- match(max(d_year), d_year)
                   c(onset_year, cessation_year)
                   })
          ) -> wet_periods
  
  as.numeric(wet_periods) + rep(seq(0, length(x)-1, 365), each = 2)
  
  }