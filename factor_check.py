#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def daily_ic(ret, factor):
    np.corrcoef(ret, factor)[1][0]

def prediction_power(ret, factor):
    # extreme = 
    pass

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

