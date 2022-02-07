#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def expo_ma(x, n):
    if not hasattr(x, 'iloc'):
        x = pd.Series(x)
        idx = None
    else:
        idx = x.index

    result = [x.iloc[0]]
    for _, i in x.iteritems():
        last_value = result[-1]
        current_value = (2 * i + (n - 1) * last_value) / (n + 1)
        result.append(current_value)
    return pd.Series(result[1:], index=idx)

def log_return(price):
    log_price = np.log(price)
    return log_price.shift(-1) - log_price

if __name__ == "__main__":
    from factors import macd
    from factor_check import cumsum
    #  dta = pd.read_hdf("./000725_snap.h5")
    #  t = pd.to_datetime(dta['exchange_time_stamp'])
    #  dta['time'] = t.apply(lambda x: (x.hour + 8) * 10000 + x.minute * 100 + x.second)
    #  dta['mid_price'] = (dta['bid_prices1'] + dta['bid_prices2']) / 2

    dta = pd.read_pickle("./data/20210726.pk")
    dta = dta[dta['wind_code'] == '688399.SH']
    #  dta = dta[dta['wind_code'] == '688005.SH']
    print(dta)
    dta['time'] = dta['c_time'] / 1000
    del dta['c_time']
    dta['mid_price'] = dta['c_close'] / 1000
    dta = dta[(dta['mid_price'] > 0.01) & (dta['time'] > 93100) & (dta['time'] < 145700)]
    dta['log_return'] = log_return(dta['mid_price'])

    dta['ma6'] = expo_ma(dta['mid_price'], 6)
    dta['macd'] = macd(dta['mid_price'], 12, 26, 9)

    dta.dropna(axis=0, subset=['log_return', 'macd'], inplace=True)
    cumsum(dta['log_return'], dta['macd'], plot=True)
    print(dta.head())

    #  plt.plot(dta['mid_price'])
    #  plt.plot(dta['ma6'])
    #  plt.show()
    #  plt.plot(dta['macd'])
    #  plt.show()
