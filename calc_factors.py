#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

from tool import expo_ma

def macd(x, short, long, mid):
    fast = expo_ma(x, short)
    slow = expo_ma(x, long)
    dif = fast - slow
    dea = expo_ma(dif, mid)
    return (dif - dea) * 2

