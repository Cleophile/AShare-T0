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
