def dailyClim(x):

    daily_clim = np.copy(x)
    daily_clim.shape = (int(len(x) / 365), 365)
    daily_clim = np.apply_along_axis(np.mean, 0, daily_clim)

    return daily_clim

def getDate(x, i_wet_i, f_wet_i, wInd, R_ave):

    i_wet = int(np.where(i_wet_i - wInd < 0, 0, i_wet_i - wInd))
    f_wet = int(np.where(f_wet_i + wInd > 365, 365, f_wet_i + wInd))
    d_year = np.cumsum(x[i_wet:f_wet] - R_ave)
    onset_year = np.argmin(d_year)
    cessation_year = np.argmax(d_year)

    return [onset_year, cessation_year]

def getOnsetCessation(x, wInd=45):

    ## daily climatology
    daily_clim = dailyClim(x)

    ## climatology onset and cessation dates
    R_ave = np.mean(daily_clim)
    d = np.cumsum(daily_clim - R_ave)
    i_wet_clim = np.argmin(d)
    f_wet_clim = np.argmax(d)

    ## getting onset and cessation dates by hydrological year
    wet_periods = np.split(x, len(x)/365)
    wet_periods = list(map(lambda y: getDate(y, i_wet_i=i_wet_clim,
                                             f_wet_i=f_wet_clim,
                                             wInd=wInd,
                                             R_ave=R_ave),
                           wet_periods))

    return np.concatenate(wet_periods, axis=0) + np.repeat(np.arange(0, len(x), 365), 2)