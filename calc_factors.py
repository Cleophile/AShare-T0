#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
from tool import expo_ma, log_return
from factor_check import cumsum, daily_ic, prediction_power
import matplotlib.pyplot as plt

def macd(x, short, long, mid):
    fast = expo_ma(x, short)
    slow = expo_ma(x, long)
    dif = fast - slow
    dea = expo_ma(dif, mid)
    return (dif - dea) * 2

def fake_factor(x, ic, df=4):
    # factor = ic * x + y
    #  x = np.array(x)

    var_x = np.std(x, ddof=1) ** 2
    var_y = (1 - ic * ic) * var_x

    #  original_var = df / (df - 2)
    y0 = np.random.standard_t(df, size=len(x))

    cov_xy0 = np.cov(x, y0)[1][0]
    a = - cov_xy0 / var_x
    y0 = y0 + a * x

    original_var = np.var(y0)
    y = y0 / np.sqrt(original_var) * np.sqrt(var_y)

    signal = ic * x + y

    return signal

if __name__ == "__main__":
    dta = pd.read_pickle("./data/20210726.pk")
    dta = dta[dta['wind_code'] == '688399.SH']
    dta['log_return'] = log_return(dta['c_close'])

    #  dta['signal'] = fake_factor(dta['log_return'].dropna(), 0.05)
    l = []
    for i in range(1000):
        dta['signal'] = fake_factor(dta['log_return'].dropna(), 0.15)
        dta.dropna(axis=0, subset=['log_return', 'signal'], inplace=True)

        #  print(dta.head())
        #  print(daily_ic(dta['log_return'], dta['signal']))
        pp = prediction_power(dta['log_return'], dta['signal'])
        cs = cumsum(dta['log_return'], dta['signal'])
        l.append(pp)
        #  plt.plot(cs)
        #  plt.show()
    print(np.mean(l))
