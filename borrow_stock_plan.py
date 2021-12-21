#!/usr/bin/env python
# coding: utf-8

import qrutils.stock_utils as su
import py_doraemon.trading_days as td
import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import stock_plan_utils as sp
from gen_stock_plan_gp import generate
import sys
import argparse

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 1000)

parser = argparse.ArgumentParser(description='generate borrow stock plan')
parser.add_argument('--plan_date', help='end date of the historical simulation we run', type=int, default=20210301)
parser.add_argument('--fitting_date', help='fitting date of gpmodel we use in this borrow stock plan',
                    type=int, default=[20210113], nargs='+')
parser.add_argument('--percentile', help='percentile', default=25)
parser.add_argument('--log', help='if not None, the program will generate an HTML report file', default=None)


def get_simu_vol(plan_date, symbol, percentile=25, period='all'):
    """
    get certain percentile of trade_quantity from 10 days simulation until plan_date for given stock

    params:
        plan_date  : int YYYYMMDD    # Read uncon_stock_plan of this date to get model_info. Also end_date of the simulation
        symbol     : int             # ticker of stock
        percentile : int
        period     : str             # period to run simulation 'all' or 'second'

    instance:
        get_simu_vol(20201218, 688399, percentile=25)
    """
    print(symbol)
    sys.stdout.flush()

    borrow_stock = pd.read_json(f'http://192.168.11.109:3000/api/tops/loans?date={td.get_next_trading_day(plan_date)}',
                                dtype={"symbol" : str})['symbol']
    borrow_stock = borrow_stock[borrow_stock > '600000']
    borrow_stock = set(borrow_stock)

    df = pd.read_csv(f'/mnt/hdd_storage_2/stock_plan/uncon_stock_plan/{plan_date}/stock_plan.csv',
                     dtype={"symbol" : str})
    df = df.set_index('symbol', drop=False)
    no_plan_stocks = ~df.index.isin(borrow_stock)
    df.loc[no_plan_stocks & (df['poslimit'] > 500), 'poslimit'] = 500
    df.loc[no_plan_stocks & (df['poslimit_ten'] > 500), 'poslimit_ten'] = 500
    df.loc[no_plan_stocks, 'poslimit_second'] = 200

    start_date = td.get_trading_days_before(plan_date, 10)[-1]
    if period == 'all':
        if df.version.iloc[0] == 'gpmodel':
            fitting_date = df[df.symbol == symbol].forecaster_file.values[0][-19:-11]
            poslimit = df[df.symbol == symbol].poslimit.values[0]
            poslimit_ten = df[df.symbol == symbol].poslimit_ten.values[0]
            poslimit_second = df[df.symbol == symbol].poslimit_second.values[0]

            result1 = su.python_simulator(symbol, start_date=start_date, end_date=plan_date, fitting_date=fitting_date,
                                          poslimit=poslimit, period='half_hour', daily=True)[['date','trade_qty']]
            result2 = su.python_simulator(symbol, start_date=start_date, end_date=plan_date, fitting_date=fitting_date,
                                          poslimit=poslimit_ten, period='ten', daily=True)[['date','trade_qty']]
            result3 = su.python_simulator(symbol, start_date=start_date, end_date=plan_date, fitting_date=fitting_date,
                                          poslimit=poslimit_second, period='second', daily=True)[['date','trade_qty']]
            result = pd.concat([result1, result2, result3])
            result = result.groupby('date').sum()
        else:
            mult = df[df.symbol == symbol].mult.values[0]
            edge = df[df.symbol == symbol].edge.values[0]
            model_file = df[df.symbol == symbol].model_file.values[0]
            poslimit = df[df.symbol == symbol].poslimit.values[0]
            version = df[df.symbol == symbol].version.values[0]
            clf = lgb.Booster(model_file=model_file)
            X_and_snapshot = su.get_X_and_snapshot(symbol, start_date=start_date, end_date=plan_date, version=version)
            result = su.python_simulation(X_and_snapshot, clf, edge, mult, poslimit=poslimit, daily=True)
    elif period == 'second':
        fitting_date = df[df.symbol == symbol].forecaster_file.values[0][-19:-11]
        poslimit_second = df[df.symbol == symbol].poslimit_second.values[0]
        result = su.python_simulator(symbol, start_date=start_date, end_date=plan_date, fitting_date=fitting_date,
                                     poslimit=poslimit_second, period='second', daily=True)
    else:
        raise KeyError("period must be {\"all\" or \"second\"}")
    simuvol = np.percentile(result.trade_qty.values, percentile)
    return simuvol


def generate_bol(plan_date, fitting_date, percentile=25, log=None):
    """
    generate borrow_stock_plan from uncon_stock_plan of given plan_date
    uncon_stock_plan will be automatically generated if not yet exists

    params:
        plan_date    : int         # from when we read the uncon_stock_plan
        fitting_date : int         # fitting date of model we use to run stock_plan
        percentile   : int         # we get the certain  percentile of simu_trade_qty and real_trade_qty
    """
    plan_date = int(plan_date)

    # generate_uncon_stock_plan
    if os.path.exists(f'/mnt/hdd_storage_2/stock_plan/uncon_stock_plan/{plan_date}/stock_plan.csv'):
        print('uncon stock plan exists now')
    else:
        run_date = td.get_next_trading_day(plan_date)
        generate(run_date, fit_date_star=fitting_date, plan_type='uncon', version='gpmodel')
    uncon = pd.read_csv(f'/mnt/hdd_storage_2/stock_plan/uncon_stock_plan/{plan_date}/stock_plan.csv',
                        dtype={'symbol': str, 'stockId' : str})

    # get 10 days of real stock info
    days = td.get_trading_days_before(plan_date, 10)
    df_set = []
    for date in days:
        print(date)
        file_name = os.path.join(sp.StockInfoCacheFolder, f"real_simu_shstar_{date}_True.xlsx")
        if os.path.exists(file_name):
            print('real info exists')
            df = pd.read_excel(file_name, dtype={'stockId': str, 'symbol': str})
        else:
            df = sp.get_stock_info(date, plan_type='show', n_days=1, report_simu=True)
        df = df[df.ror_s != 0]
        df['vol_mult'] = df.apply(lambda x: x['vol_r'] / x['vol_s'] if 0 < x['vol_r'] <= x['vol_s'] and
                                  x['vol_r'] / x['vol_s'] < 0.8 else (
                                  0.8 if 0 < x['vol_s'] and x['vol_r'] > 0 else np.nan),
                                  axis=1)
        df['ror_mult'] = df.apply(lambda x: x['ror_r'] / x['ror_s'] if 0 < x['ror_r'] <= x['ror_s']
                                  else (1 if 0 < x['ror_s'] <= x['ror_r'] else np.nan), axis=1)
        df_set.append(df)
    real_info = pd.concat(df_set)
    real_info = real_info.groupby('symbol').mean()

    # get 10 days distribution of simulated volume(simuvol)
    simuvol_file = f'/mnt/hdd_storage_2/stock_plan/borrow_stock_plan/simuvol/{plan_date}_{fitting_date}_simuvol.csv'
    if os.path.exists(simuvol_file):
        print('simu_vol exists')
        simuvol_data = pd.read_csv(simuvol_file, dtype={'symbol': str})
    else:
        result_set = []
        for symbol in uncon['symbol']:
            simuvol = get_simu_vol(plan_date, symbol, percentile)
            result = {'symbol': symbol, 'simuvol25p': simuvol}
            result_set.append(result)
        simuvol_data = pd.DataFrame(result_set)
        simuvol_data.to_csv(simuvol_file, index=False)

    compare = simuvol_data.set_index('symbol').join(uncon.set_index('symbol'))
    compare = compare.join(real_info, rsuffix='_r')

    # Split the stocks into good, bad and new:
    # stocks with ror_r < 0 must be bad
    # "new" is for those that has not appeared in the real trading
    comp = compare[compare.ror_r > 0].copy()
    bad1 = compare[compare.ror_r <= 0].copy()
    new = compare[np.isnan(compare.ror_r)].copy()

    del compare

    comp['ror_mult'] = comp['ror_mult'].fillna(comp['ror_mult'].mean())
    comp['vol_mult'] = comp['vol_mult'].fillna(0.8)
    comp['ror_'] = comp.ror * comp.ror_mult
    comp['ror_second_'] = comp.ror_second * comp.ror_mult
    comp['ror_second_'] = comp['ror_second_'].fillna(0)
    good = comp[(comp.ror_ > 0.2) | (comp.ror_second_ > 0.2)].copy()
    bad2 = comp[(comp.ror_ < 0.2) & (comp.ror_second_ < 0.2)].copy()

    del comp

    good['simu_vol'] = good.apply(lambda x: x['simuvol25p'] if x['ror_second_'] > 0.2 else
                                  x['simuvol25p'] - get_simu_vol(plan_date, x['stockId'], percentile,
                                  period='second'), axis=1)
    good['fr_'] = good.apply(lambda x: x['fr'] if x['fr'] <= 0.8 else 0.8, axis=1)
    good['borrow_vol'] = good.simu_vol * good.fr_
    good['quant'] = good.borrow_vol * 0.5
    good['quant'] = good.apply(lambda x: x['quant'] if x['use_ratio'] >= 0.95 else x['quant']*x['use_ratio'], axis=1)
    good['quant'] = good.apply(lambda x: x['quant'] if x['trade_time'] < 2.6 else min(x['quant'], x['quantity']), axis=1)

    bad = pd.concat([bad1, bad2], sort=True)
    new['borrow_vol'] = new.simuvol25p * 0.7
    new['quant'] = new.borrow_vol * 0.5

    # get market trading volume of stocks
    datelist = td.get_trading_days_before(plan_date, 5)
    market_table = []
    for date in datelist:
        df = pd.read_json(f'http://192.168.11.109:3000/api/config/stock-instruments/{date}/search').T
        market_table.append(df)
    market_table = pd.concat(market_table)
    market_table['vol'] = market_table['vol'].astype(float)
    market_table = market_table.groupby('symbol').mean()

    good = good.join(market_table)
    good['quant'] = good.apply(lambda x: x['quant'] if x['quant']*2 < x['vol']*5 else x['vol']*5*0.5, axis=1)
    good['money'] = good.price*good.quant
    new = new.join(market_table)
    new['quant'] = new.apply(lambda x: x['quant'] if x['quant']*2 < x['vol']*5 else x['vol']*5*0.5, axis=1)
    new['money'] = new.price*new.quant

    final_result = pd.concat([good, new], sort=True)
    final_result.to_csv(f'/mnt/hdd_storage_2/stock_plan/borrow_stock_plan/{plan_date}_borrow_plan.csv')
    final_result.to_csv(f'./{plan_date}_borrow_plan.csv')
    bad.to_csv(f'/mnt/hdd_storage_2/stock_plan/borrow_stock_plan/{plan_date}_bad_stock.csv')
    bad.to_csv(f'./{plan_date}_bad_stock.csv')

    # log must be the last procedure in this function,
    # since data is NOT guaranteed to maintain intact
    if log is not None:
        from report_utils import borrow_summary
        borrow_summary((plan_date, fitting_date, percentile),(good,new,bad,final_result),
                       (None, None,
                        f'/mnt/hdd_storage_2/stock_plan/borrow_stock_plan/{plan_date}_bad_stock.csv',
                        f'/mnt/hdd_storage_2/stock_plan/borrow_stock_plan/{plan_date}_borrow_plan.csv'), log)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'addscript':
        pass
    else:
        print(sys.argv)
        args = parser.parse_args()
        generate_bol(plan_date=args.plan_date, fitting_date=args.fitting_date, log=args.log)
