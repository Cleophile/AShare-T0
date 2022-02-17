#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Hikari Software
# Y-Enterprise

# market: time
# 时分秒

import numpy as np
import pandas as pd
from calc_factors import fake_factor
from factor_check import daily_ic

Transaction_Cost = 0.00164
open_time = 93100
morning_close = 113000
afternoon_open = 130000
close_time = 145700

def int_time_to_tuple(s):
    second = s % 100
    s = s // 100
    minute = s % 100
    hour = s // 100
    return [hour, minute, second]

def second_diff(t1, t2):
    h1, m1, s1 = int_time_to_tuple(t1)
    h2, m2, s2 = int_time_to_tuple(t2)
    return (h2 - h1) * 3600 + (m2 - m1) * 60 + (s2 - s1)

class Order():
    '''
    Handle the status of Order
    '''
    def __init__(self, direction, number, price, send_time) -> None:
        self.position_cost = None
        self.original_number = number
        self.original_price = price

        self.current_position = 0

        # 1: Buy -1: Sell
        self.direction = direction

        # 0 : Order Submitted
        # 1 : Order Filled
        # 2 : Reverse Submitted
        # 3 : Reverse Filled
        self.status = 0

        self.pnl = 0

        self.send_time = send_time

    def __str__(self):
        s = []

        s.append("Order sent at: {}".format(self.send_time))
        s.append("Original order: {}, ￥{}".format(self.original_number, self.original_price))
        s.append("Cost: {}".format(self.position_cost))
        s.append("Direction: {}".format({1:'Buy', -1:"Sell"}[self.direction]))
        s.append("Current status:{}".format({0:'Submitted', 1:'Filled', 2:'Reverse Submitted', 3:'Closed', -1:'Cancelled'}[self.status]))
        s.append("Current_position: {}".format(self.current_position))
        s.append("Current_pnl: {}".format(self.pnl))

        return "\n".join(s)

class Account():
    '''
    Functions when trading
    '''
    def __init__(self, per_order=200, poslimit=600, daily_limit=None, threshold=0.0) -> None:
        assert per_order >= 200

        self.current_pos = 0
        self.total_pos = 0

        self.order_book = []

        self.fee = np.log(Transaction_Cost + 1)
        self.threshold = threshold

        self.poslimit = poslimit
        self.per_order = per_order

        self.daily_limit = daily_limit

    def can_send(self, signal, market):
        bid1 = market['pb1']
        ask1 = market['pa1']
        send_time = market['time']

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
            if np.isnan(ask1) or ask1 == 0.0:
                return
            new_order = Order(number=max_up, direction=1, price=ask1, send_time=send_time)
            #  print(new_order)
            self.order_book.append(new_order)
            return
        elif signal < -(self.fee + self.threshold):
            if np.isnan(bid1) or bid1 == 0.0:
                return
            new_order = Order(number=-max_down, direction=-1, price=bid1, send_time=send_time)
            #  print(new_order)
            self.order_book.append(new_order)
            return

    def update_pnl(self, price):
        pass

    # handle if buy and sell at the same time
    def will_exaust(self, signal, market):
        current_time = market['time']
        if current_time >= 145000:
            for order in self.order_book:
                if order.status == 1:
                    order.status = 2
            return

        for order in self.order_book:
            if order.status != 1:
                continue
            passed_time = second_diff(order.send_time, current_time)
            print(passed_time)
            if passed_time < 58:
                continue
            if order.direction == 1:
                #  if signal > self.threshold + self.fee:
                if signal > self.threshold:
                    order.send_time = current_time
                else:
                    order.status = 2
                continue
            if order.direction == -1:
                #  if signal < -self.threshold - self.fee:
                if signal < -self.threshold:
                    order.send_time = current_time
                else:
                    order.status = 2

class Exchange():
    def __init__(self):
        pass

    # Judge by the market
    def can_fill_one(self, order : Order, market):
        # 0 : Order Submitted
        # 1 : Order Filled
        # 2 : Reverse Submitted
        # 3 : Reverse Filled
        if order.status == 0:
            if order.direction == 1:
                # now buy
                trade_qty = 0
                trade_cost = 0
                for i in range(1, 6):
                    price = market["pa{}".format(i)]
                    quantity = market["va{}".format(i)]
                    if price is None or quantity is None:
                        continue
                    if np.isnan(price) or price==0.0 or np.isnan(quantity) or quantity==0.0:
                        continue

                    if price > order.original_price:
                        break
                    qty_left = min(order.original_number - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += qty_left
                    trade_cost += qty_left * price
                order.current_position += trade_qty
                order.pnl -= trade_cost
                order.position_cost = trade_cost
                if order.current_position != 0:
                    order.status = 1
                    return trade_qty
                else:
                    order.status = -1
                    return 0

            if order.direction == -1:
                # now sell
                trade_qty = 0
                trade_cost = 0
                for i in range(1, 6):
                    price = market["pb{}".format(i)]
                    quantity = market["vb{}".format(i)]
                    if price is None or quantity is None:
                        continue
                    if np.isnan(price) or price==0.0 or np.isnan(quantity) or quantity==0.0:
                        continue

                    if price < order.original_price:
                        break
                    qty_left = min(order.original_number - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += qty_left
                    trade_cost += qty_left * price
                order.current_position -= trade_qty
                order.pnl += trade_cost
                order.position_cost = trade_cost
                if order.current_position != 0:
                    order.status = 1
                    return -trade_qty
                else:
                    order.status = -1
        return 0

    def fill_market(self, order : Order, market):
        if order.status == 2:
            if order.direction == -1:
                # now buy
                trade_qty = 0
                trade_cost = 0
                for i in range(1, 6):
                    price = market["pa{}".format(i)]
                    quantity = market["va{}".format(i)]
                    print(price, order.original_price)
                    if price is None or quantity is None:
                        continue
                    if np.isnan(price) or price==0.0 or np.isnan(quantity) or quantity==0.0:
                        continue

                    qty_left = min(-order.current_position - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += qty_left
                    trade_cost += qty_left * price
                order.current_position += trade_qty
                order.pnl -= trade_cost
                if order.current_position == 0:
                    order.status = 3
                return trade_qty

            if order.direction == 1:
                # now sell
                trade_qty = 0
                trade_cost = 0
                for i in range(1, 6):
                    price = market["pb{}".format(i)]
                    quantity = market["vb{}".format(i)]
                    if price is None or quantity is None:
                        continue
                    if np.isnan(price) or price==0.0 or np.isnan(quantity) or quantity==0.0:
                        continue

                    qty_left = min(order.current_position - trade_qty, quantity)
                    if qty_left <= 0:
                        break
                    trade_qty += qty_left
                    trade_cost += qty_left * price
                order.current_position -= trade_qty
                order.pnl += trade_cost
                if order.current_position == 0:
                    order.status = 3
                return -trade_qty
        return 0

    def can_fill(self, account, market):
        for order in account.order_book:
            if order.status == 0:
                trade_qty = self.can_fill_one(order, market)
                account.current_pos += trade_qty
                account.total_pos += np.abs(trade_qty)
            if order.status == 2:
                trade_qty = self.fill_market(order, market)
                account.current_pos += trade_qty

def parse_date(s):
    year = s[:4]
    month = s[5:7]
    day = s[8:10]
    date = year + month + day
    return int(date)

def parse_time(s):
    hour = s[11:13]
    minute = s[14:16]
    second = s[17:19]
    t = hour + minute + second
    return int(t)

def stock_simulation(full_data : pd.DataFrame, day, per_order, poslimit, ic, daily_limit=None, df=5):
    dta = full_data[full_data['date'] == day].copy()
    dta['ret'] = dta['log_price'].shift(-20) - dta['log_price']

    dta = dta[((dta['time'] > open_time) & (dta['time'] < morning_close)) | ((dta['time'] > afternoon_open) & (dta['time'] < close_time))].copy()
    dta = dta[~dta['ret'].isin([np.inf, -np.inf, np.nan])]

    dta['signal'] = fake_factor(dta['ret'], ic, df=df)
    print("IC:", daily_ic(dta['ret'], dta['signal']))
    dta['minute'] = dta['time'] // 100

    print(dta)

    first_second = dta.groupby(['minute']).first()['trade_time'].copy()
    first_second = pd.DataFrame(first_second, columns=['trade_time'])
    first_second['is_first'] = 1
    dta = dta.merge(first_second, on='trade_time', how='left')

    account = Account(per_order=per_order, poslimit=poslimit, daily_limit=daily_limit, threshold=0.00)
    exchange = Exchange()
    for _, market in dta.iterrows():
        if market['is_first'] == 1:
            account.can_send(market['signal'], market)
            exchange.can_fill(account, market)
        account.will_exaust(market['signal'], market)
        exchange.can_fill(account, market)
    print("------------------")
    for order in account.order_book:
        print(order)
        print("------------------")

    pnl_result = [i.pnl for i in account.order_book]
    cost_result = [i.position_cost for i in account.order_book]
    total_pnl = sum(pnl_result)
    print(account.total_pos)
    total_cost = sum(cost_result)
    return {'total_pnl' : total_pnl, 'total_position': account.total_pos, 'money_used': total_cost, 'return_rate': total_pnl / total_cost * 25200}

if __name__ == "__main__":
    full_data = pd.read_csv("../star_tick/202101/688399.csv")
    full_data['date'] = full_data['trade_time'].apply(lambda x : parse_date(x))
    full_data['time'] = full_data['trade_time'].apply(lambda x : parse_time(x))
    full_data['log_price'] = np.log((full_data['pa1'] + full_data['pb1']) / 2)

    all_days = sorted(list(set(full_data['date'])))
    print(all_days)

    for day in all_days:
        print(stock_simulation(full_data, day, 500, 20000, 0.4))
        break
