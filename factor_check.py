#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize

def daily_ic(ret, factor):
    return np.corrcoef(ret, factor)[1][0]

def prediction_power(ret, factor, limit=0.2):
    df = pd.DataFrame(dict(ret=ret, factor=factor))
    low = df['factor'].quantile(limit / 2)
    high = df['factor'].quantile(1 - limit / 2)
    df = df[(df['factor'] >= high) | (df['factor'] <= low)]
    return np.corrcoef(df['ret'], df['factor'])[1][0]

def cumsum(ret, factor, plot=None):
    result = np.cumsum(ret * factor)

    if plot is True:
        plt.plot(result)
        plt.show()
    else:
        if plot is not None:
            plt.plot(result)
            plt.savefig(plot)

    return result

