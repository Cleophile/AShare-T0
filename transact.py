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
    def __init__(self, direction, number, price) -> None:
        self.position_cost = None
        self.original_number = number
        self.original_price = price

        self.current_position = 0

        # 0: Buy 1: Sell
        self.direction = direction

        # 0 : Order Submitted
        # 1 : Order Filled
        # 2 : Reverse Submitted
        # 3 : Reverse Filled
        self.status = 0

        self.pnl = 0

    def fill(self, price):
        self.status = 1
        self.price = price

class Account():
    '''
    Functions when trading
    '''
    def __init__(self, per_order=200, poslimit=600) -> None:
        assert per_order >= 200

        self.current_pos = 0
        self.total_pos = 0

        self.order_book = []
        self.order_active = []

        self.fee = np.log(Transaction_Cost + 1)
        self.threshold = 0.0

        self.poslimit = poslimit
        self.per_order = per_order

        self.daily_limit = None

    def can_send(self, signal, bid1, ask1):
        current_pos = self.current_pos
        max_up = min(self.per_order, self.poslimit - current_pos)
        if max_up < 0:
            max_up = 0

        max_down = max(-self.per_order, -1 * self.poslimit - current_pos)
        if max_down > 0:
            max_down = 0

        if self.daily_limit is not None:
            remain_limit = self.daily_limit - self.total_pos
            if max_up > remain_limit:
                max_up = remain_limit
            if max_down < -remain_limit:
                max_down = -remain_limit

        if signal > (self.fee + self.threshold):
            return Order(number=max_up, direction=1, price=ask1)
        elif signal < -(self.fee + self.threshold):
            return Order(number=-max_down, direction=-1, price=bid1)

    def update_pnl(self, price):
        pass

    # handle if buy and sell at the same time
    def will_exaust(self, signal, order):
        for order in self.order_active:
            pass

class Exchange():
    def __init__(self):
        pass

    # Judge by the market
    def can_fill(self, order : Order, market):
        # 0 : Order Submitted
        # 1 : Order Filled
        # 2 : Reverse Submitted
        # 3 : Reverse Filled
        if order.status == 0:
            if order.direction == 1:
                # now buy
                trade_qty = 0
                trade_cost = 0
                for i in range(1,6):
                    price = "pa{}".format(i)
                    quantity = "qa{}".format(i)

                    if price > order.original_price:
                        break
                    qty_left = min(order.original_number - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += quantity_left
                    trade_cost += quantity_left * price
                order.current_position += trade_qty
                order.pnl -= trade_cost
                if order.current_position != 0:
                    order.status = 1
                else:
                    order.status = -1

            if order.direction == -1:
                # now sell
                trade_qty = 0
                trade_cost = 0
                for i in range(1,6):
                    price = market["pa{}".format(i)]
                    quantity = market["qa{}".format(i)]

                    if price < order.original_price:
                        break
                    qty_left = min(order.original_number - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += quantity_left
                    trade_cost += quantity_left * price
                order.current_position -= trade_qty
                order.pnl += trade_cost
                if order.current_position != 0:
                    order.status = 1
                else:
                    order.status = -1

        if order.status == 2:
            if order.direction == -1:
                # now buy
                trade_qty = 0
                trade_cost = 0
                for i in range(1,6):
                    price = market["pa{}".format(i)]
                    quantity = market["qa{}".format(i)]

                    qty_left = min(-order.current_position - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += quantity_left
                    trade_cost += quantity_left * price
                order.current_position += trade_qty
                order.pnl -= trade_cost
                if order.current_position == 0:
                    order.status = 3

            if order.direction == 1:
                # now sell
                trade_qty = 0
                trade_cost = 0
                for i in range(1,6):
                    price = market["pa{}".format(i)]
                    quantity = market["qa{}".format(i)]

                    qty_left = min(order.current - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += quantity_left
                    trade_cost += quantity_left * price
                order.current_position -= trade_qty
                order.pnl += trade_cost
                if order.current_position == 0:
                    order.status = 3



if __name__ == "__main__":
    pass


