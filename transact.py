#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

# market: time
# 时分秒

import numpy as np

Transaction_Cost = 0.00164

class Order():
    '''
    Handle the status of Order
    '''
    def __init__(self, direction, number) -> None:
        self.number = number
        # 0: Buy 1: Sell
        self.direction = direction
        # 0 : Order Submitted
        # 1 : Order Filled
        #-1 : Order Cancelled
        self.status = 0

    def fill(self, price):
        self.status = 1
        self.price = price

class Account():
    '''
    Handle daily trade log and pnl
    '''
    def __init__(self) -> None:
        self.pnl = 0
        self.current_pos = 0
        self.total_pos = 0
        self.order_book = []
        self.order_active = []

class Trade():
    '''
    Functions when trading
    '''
    def __init__(self) -> None:
        self.account = Account()

        self.fee = np.log(Transaction_Cost + 1)
        self.threshold = 0.0

        self.poslimit = 400
        self.per_order = 100

        self.daily_limit = None

    def can_send(self, signal):
        current_pos = self.account.current_pos
        max_up = min(self.per_order, self.poslimit - current_pos)
        if max_up < 0:
            max_up = 0

        max_down = max(-self.per_order, -1 * self.poslimit - current_pos)
        if max_down > 0:
            max_down = 0

        if self.daily_limit is not None:
            remain_limit = self.daily_limit - self.account.total_pos
            if max_up > remain_limit:
                max_up = remain_limit
            if max_down < -remain_limit:
                max_down = -remain_limit

        if signal > (self.fee + self.threshold):
            return Order(number=max_up, direction=1)
        elif signal < -(self.fee + self.threshold):
            return Order(number=-max_down, direction=-1)

    # Judge by the market
    def can_fill(self, market, order):
        order.status = 1

    # handle if buy and sell at the same time
    def will_exaust(self, market, order):
        for order in self.account.order_active:

def gen_signal(factors, weights):
    return 0


if __name__ == "__main__":
    pass


