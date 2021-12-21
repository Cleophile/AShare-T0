#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import qrutils.stock_utils as su
import py_doraemon.trading_days as td
import lightgbm as lgb
import glob
import os
from py_doraemon.dodrio import multi_run
import json
import forecaster_utils as fu
from forecaster_meta import BaseForecaster
import datetime
import sys
import numpy as np


pd.set_option('display.max_columns', None)

StockInfoCacheFolder = "/mnt/hdd_storage_2/stock_plan_cache/compare/real_simu"
PerformanceRequestAddress = "http://192.168.11.104:10073/api/performance"


def simulation_summary(stockId, start_date, end_date, fitting_date, poslimits,
                       periods, max_quantity, latency=False):
    result1 = su.python_simulator(stockId, start_date=start_date, end_date=end_date,
                                  fitting_date=fitting_date, poslimit=poslimits[0],
                                  max_quantity=max_quantity, period=periods[0],
                                  latency=latency)
    result_list = [result1]

    for period, poslimit in zip(periods[1:],poslimits[1:]):
        if result1 == {} or max_quantity <= result1['trade_qty']:
            result_list.append({})
            result1 = {}
            continue
        max_quantity -= result1['trade_qty']
        try:
            result1 = su.python_simulator(stockId, start_date=start_date, end_date=end_date,
                                          fitting_date=fitting_date, poslimit=poslimit,
                                          max_quantity=max_quantity, period=period,
                                          latency=latency)
        except IndexError as e:
            print(f'{stockId}: Error in python simulation:', e)
            result1 = {}
        result_list.append(result1)
    return pd.DataFrame(result_list)

def get_stock_info(run_date, plan_type='show', n_days=5, report_simu=False):
    """
    Extract information of REAL stock trading of the past n days

    params:
        run_date    : int
        plan_type   : str             # type of trade_plan 'new' or 'uncon' or 'show'
        n_days      : int             # number of days we calculate real trading information until run_date - 1
        report_simu : bool            # whether to report the result of simulation

    plan_type ==
        'new'  : returns fill_rate(fr), net_profit(np) and borrow_quantity for stocks we borrowed of sh at run_date
        'uncon': returns fill_rate(fr), net_profit(np) for stocks of 'sh_loan' if plan_type=='uncon'
        'show' : returns and saves fill_rate(fr), net_profit(np),
                 ror, trade_vol, price and borrow_qty for stocks we traded before run_date

    instance:
        get_stock_info(20201224, 'show', n_days=5)
    """
    print(f'whether to report simu : {report_simu}')
    print(f'run_date is {run_date}')
    sys.stdout.flush()

    if plan_type != 'show':
        # get the fill_rate info of stocks for past n_days
        datelist = td.get_trading_days_before(int(run_date), n_days + 1)[1:][::-1]
        start_date = datelist[0]
        end_date = datelist[-1]
        fill_info = []
        for item in datelist:
            print(item)
            df = pd.read_json('http://192.168.11.109:3000/api/tops/accounts?date={}'.format(item),
                          dtype={"symbol": str})
            df = df.groupby('symbol').sum()
            fill_info.append(df)

        fill_info = pd.concat(fill_info).groupby('symbol').mean()
        fill_info['hr'] = 1 - fill_info.pure_cancel_count / fill_info.order_count
        fill_info['fr'] = fill_info.trade_volume / fill_info.submit_volume
        fill_info = fill_info[(fill_info.index >= '600000') & (fill_info.index <= '700000')]

        # get the real performance of stocks for past n_days
        performance = pd.read_json(os.path.join(PerformanceRequestAddress, f"report/{start_date}/{end_date}"),
                                   dtype={"instrument_id": str})
        performance['np'] = performance['gp'] - performance['fee']
        performance = performance.groupby('instrument_id').mean()

        if plan_type == 'new':
            table_name = f'real_stock_info_{start_date}_{end_date}.xlsx'
            # get the real performance of stocks for past three days
            start_date_3 = td.get_trading_days_before(int(run_date), 3+1)[-1]
            end_date_3 = td.get_trading_days_before(int(run_date), 3+1)[1]

            performance_3 = pd.read_json(os.path.join(PerformanceRequestAddress, f"report/{start_date_3}/{end_date_3}"),
                                         dtype={"instrument_id": str})
            performance_3['np'] = performance_3['gp'] - performance_3['fee']
            performance_3 = performance_3.groupby('instrument_id').mean()

            stock_info = performance.join(fill_info, rsuffix='_np')
            stock_info = stock_info.join(performance_3, rsuffix='_3')

            # get the quantity we borrow for stocks
            quantity_info = pd.read_json(f'http://192.168.11.109:3000/api/tops/loans-virtual?date={run_date}',
                                         dtype={'symbol': str}).groupby('symbol').sum()
            quantity_info = quantity_info[quantity_info.quantity > 0]
            quantity_info = quantity_info.drop(columns=['price'])
            stock_info = pd.merge(stock_info, quantity_info, left_index=True, right_index=True, how='right')[
                ['fr', 'hr', 'np', 'quantity', 'amount', 'np_3', 'amount_3', 'price']]
            # tag stocks for ror_3 and ror <= 0.1 as bad
            stock_info['ror'] = stock_info.np * 252 / stock_info.amount
            stock_info['ror_3'] = stock_info.np_3 * 252 / stock_info.amount_3
            stock_info['tag'] = stock_info.apply(lambda x: 'bad' if x.ror <= 0.25 or x.ror_3 <= 0.25 else 'good', axis=1)
            stock_info['stockId'] = stock_info.index
            # stock_info = stock_info[stock_info.stockId > '688000']
            stock_info = stock_info[stock_info.stockId != '688169']
            stock_info = stock_info[['stockId', 'price', 'np', 'ror', 'amount', 'quantity', 'np_3', 'ror_3', 'amount_3',
                                     'fr', 'hr', 'tag']]
            stock_info.to_excel(f'./{table_name}')
            return stock_info

        elif plan_type == 'uncon':
            for symbol in su.get_stock_list('sh_all'):
                if symbol not in fill_info.index and symbol > '600000':
                    fill_info.loc[symbol] = 1
            stock_info = fill_info.join(performance, rsuffix='_np')[['fr', 'hr', 'np']]
            stock_info['tag'] = stock_info.apply(lambda x: 'bad' if x.np <= 0 else 'good', axis=1)
            stock_info['stockId'] = stock_info.index
            # stock_info = stock_info[stock_info.stockId > '688000']
            return stock_info
        else:
            raise KeyError("plan_type must be one of the {'uncon', 'new', 'show'}")

    # plan_type=="show", generate file
    table_name = f'real_simu_shstar_{run_date}_{report_simu}.xlsx'
    if os.path.exists(os.path.join(StockInfoCacheFolder, table_name)):
        stock_info = pd.read_excel(os.path.join(StockInfoCacheFolder, table_name),
                                   dtype={"symbol" : str, "stockId" : str})
        stock_info.set_index("symbol", inplace=True)
        stock_info.to_excel(f'./{table_name}')
        return stock_info

    start_date = run_date
    end_date = run_date

    # calculate 10 days sharpe
    sdate = td.get_trading_days_before(run_date, 10)[-1]
    performance_10 = pd.read_json(os.path.join(PerformanceRequestAddress, f"report/{sdate}/{end_date}"),
                                  dtype={"instrument_id": str})
    performance_10['np'] = performance_10['gp'] - performance_10['fee']
    np_std = performance_10.groupby('instrument_id').agg(np.std, ddof=0)
    np_mean = performance_10.groupby('instrument_id').mean()
    np_std = pd.DataFrame(np_std.np)
    np_mean = pd.DataFrame(np_mean.np)
    ten_sharpe = np_std.join(np_mean, lsuffix='_std', rsuffix='_mean')
    ten_sharpe['tendays_sharpe'] = ten_sharpe.apply(lambda x: x['np_mean']*(252 ** 0.5)/x['np_std'] if
                                   x['np_std'] > 0 else np.nan, axis=1)

    # get n_days performance
    performance = pd.read_json(os.path.join(PerformanceRequestAddress, f"report/{start_date}/{end_date}"),
                               dtype={"instrument_id" : str})
    performance['np'] = performance['gp'] - performance['fee']
    performance['iid'] = performance['instrument_id'] + performance['portfolio']
    performance = performance.groupby('iid').mean()
    performance['iid'] = performance.index
    performance['instrument_id'] = performance.apply(lambda x: x['iid'][0:6], axis=1)
    performance = performance.groupby('instrument_id').sum()
    performance = performance.join(ten_sharpe)

    # get fill rate
    datelist = td.get_trading_day_between(start_date, end_date)
    fill_info = []
    for item in datelist:
        df = pd.read_json('http://192.168.11.109:3000/api/tops/accounts?date={}'.format(item),
                          dtype={"symbol": str})
        df = df.groupby('symbol').sum()
        fill_info.append(df)
    fill_info = pd.concat(fill_info).groupby('symbol').mean()
    fill_info['hr'] = 1 - fill_info.pure_cancel_count / fill_info.order_count
    fill_info['fr'] = fill_info.trade_volume / fill_info.submit_volume
    fill_info = fill_info[(fill_info.index >= '600000') & (fill_info.index <= '700000')]

    # get max_drawdown
    risk = pd.read_json(os.path.join(PerformanceRequestAddress, f"risk/{start_date}"),
                        dtype={"instrument_id": str})
    risk['symbol'] = risk.instrument_id
    risk = risk.set_index('symbol')
    risk['crest_'] = risk.apply(lambda x: round(x['crest'], -9), axis=1)
    risk['draw_start'] = pd.to_datetime(risk["crest_"]) + datetime.timedelta(hours=8)
    risk['trough_'] = risk.apply(lambda x: round(x['trough'], -9), axis=1)
    risk['draw_end'] = pd.to_datetime(risk["trough_"]) + datetime.timedelta(hours=8)

    stock_info = performance.join(fill_info, rsuffix='_np')
    stock_info['ror'] = stock_info.np * 252 / stock_info.amount
    stock_info['stockId'] = stock_info.index
    stock_info = stock_info.join(risk)

    # get poslimit_set of stocks
    stock_plan = pd.read_csv(f'/mnt/hdd_storage_2/stock_plan/new_stock_plan/{td.get_prev_trading_day(end_date)}/stock_plan.csv',
                             dtype={"stockId": str})
    stock_plan['stockId'] = stock_plan.apply(lambda x: x['stockId'][0:6], axis=1)
    stock_plan = stock_plan.dropna(subset=['stockId'])
    if 'poslimit_ten' in stock_plan.columns:
        stock_plan['pos_dict'] = stock_plan.apply(lambda x: {'poslimit': x['poslimit'],
                                                             'poslimit_ten': x['poslimit_ten'],
                                                             'poslimit_second': x['poslimit_second']}, axis=1)
    else:
        stock_plan['pos_dict'] = stock_plan.apply(lambda x: {'poslimit': x['poslimit'],
                                                             'poslimit_second': x['poslimit_second']}, axis=1)
    stock_plan['symbol'] = stock_plan['stockId']
    stock_plan = stock_plan.set_index('symbol', drop=False)
    stock_info = stock_info.join(stock_plan, rsuffix='_sp')

    # get quantity we borrow of stocks
    quantity_info = pd.read_json(f'http://192.168.11.109:3000/api/tops/loans?date={end_date}',
                                 dtype={"symbol": str}).groupby('symbol').sum()
    quantity_info = quantity_info[quantity_info.quantity > 0].drop(columns=['price'])
    stock_info = pd.merge(stock_info, quantity_info, left_index=True, right_index=True, how='right')[
        ['stockId', 'price', 'np', 'avgpnl', 'avgpnl_second', 'ror', 'ror_sp', 'ror_second',
         'pos_dict', 'quantity', 'TotalVolume', 'trade_qty', 'trade_qty_second', 'fr', 'hr', 'trade_time', 'dropdown',
         'draw_start', 'draw_end', 'tendays_sharpe']]
    stock_info['use_ratio'] = stock_info.TotalVolume*0.5/stock_info.quantity

    # get the simulation performance of stocks using same model and poslimit at the same day
    if report_simu is True or report_simu == 'True':
        result_set = []
        for item in stock_plan.index:
            if str(item).startswith('6'):
                latency = False
            else:
                latency = 3.7

            try:
                if 'poslimit_ten' in stock_plan.columns:
                    poslimit = stock_plan[stock_plan.symbol == item].poslimit.values[0]
                    poslimit_ten = stock_plan[stock_plan.symbol == item].poslimit_ten.values[0]
                    poslimit_second = stock_plan[stock_plan.symbol == item].poslimit_second.values[0]
                    forecaster_file = stock_plan[stock_plan.symbol == item].forecaster_file.values[0]
                    fitting_date = forecaster_file[34:42]
                    max_quantity = stock_plan[stock_plan.symbol == item].max_qty.values[0]/1.2

                    result = simulation_summary(item, start_date, end_date, fitting_date, [poslimit, poslimit_ten, poslimit_second],
                                       ['half_hour','ten','second'], max_quantity, latency)
                else:
                    poslimit = stock_plan[stock_plan.symbol == item].poslimit.values[0]
                    poslimit_second = stock_plan[stock_plan.symbol == item].poslimit_second.values[0]
                    forecaster_file = stock_plan[stock_plan.symbol == item].forecaster_file.values[0]
                    fitting_date = forecaster_file[34:42]
                    max_quantity = stock_plan[stock_plan.symbol == item].max_qty.values[0]

                    result = simulation_summary(item, start_date, end_date, fitting_date, [poslimit, poslimit_second],
                                       ['first','second'], max_quantity, latency)
                result = result[result.trade_qty > 0]
                result['money'] = result.avgpnl*252/result.ror
                result['money_all'] = result.money.sum()
                result['np_s'] = result.avgpnl.sum()
                result['ror_s'] = result.np_s*252/result.money_all
                result['vol_s'] = result.trade_qty.sum()
                result_set.append(result)
            except Exception as e:
                print(str(item) + 'notworkkkkkk')
                print(e)
                continue
        result_set = pd.concat(result_set)
        result_set['symbol'] = result_set.stockId
        result_set = result_set.set_index('symbol', drop=False)

        part_set = result_set.groupby('stockId').max()[['np_s', 'ror_s', 'vol_s']]

        stock_info = stock_info.join(part_set, rsuffix='_simu')
        stock_info = stock_info.dropna(subset=['stockId'])
        names = {'np': 'np_r', 'avgpnl': 'np_first', 'avgpnl_second': 'np_second', 'ror': 'ror_r',
                 'ror_sp': 'ror_first', 'TotalVolume': 'vol_r', 'trade_qty': 'vol_first',
                 'trade_qty_second': 'vol_second', 'dropdown': 'drawdown'}
        stock_info.rename(columns=names, inplace=True)
        stock_info = stock_info[['stockId', 'price', 'np_r', 'np_s', 'np_first', 'np_second', 'ror_r', 'ror_s',
                                 'ror_first', 'ror_second', 'pos_dict', 'quantity', 'vol_r', 'vol_s', 'vol_first',
                                 'vol_second', 'use_ratio', 'fr', 'hr', 'drawdown', 'draw_start', 'draw_end',
                                 'trade_time', 'tendays_sharpe']]
    else:
        stock_info = stock_info.dropna(subset=['stockId'])
        names = {'np': 'np_r', 'avgpnl': 'np_first', 'avgpnl_second': 'np_second', 'ror': 'ror_r',
                 'ror_sp': 'ror_first', 'TotalVolume': 'vol_r', 'trade_qty': 'vol_first',
                 'trade_qty_second': 'vol_second', 'dropdown': 'drawdown'}
        stock_info.rename(columns=names, inplace=True)
        stock_info = stock_info[['stockId', 'price', 'np_r', 'np_first', 'np_second', 'ror_r',
                                 'ror_first', 'ror_second', 'pos_dict', 'quantity', 'vol_r', 'vol_first',
                                 'vol_second', 'use_ratio', 'fr', 'hr', 'drawdown', 'draw_start', 'draw_end',
                                 'trade_time', 'tendays_sharpe']]
    stock_info = stock_info.round(2)
    stock_info.to_excel(os.path.join(StockInfoCacheFolder, table_name))
    stock_info.to_excel(f'./{table_name}')
    return stock_info

special_poslimit_first = {
    '688169': [200],
    '688536': [200],
    '688185': [200],
    '688111': [200, 500, 800],
    '688390': [200, 500, 800],
    '688339': [200, 500],
    '688036': [200, 500],
    '688023': [200, 500],
    '688202': [200, 500],
    '688200': [200, 500],
    '688368': [200, 500],
    '688686': [200, 500],
    '688019': [200, 500],
    '688198': [200, 500],
    '688050': [200, 500]
}

special_poslimit_second = {
    "688169" : [200],
    "688536" : [200]
}

def _gen_model_performance(single_stock_info, start_date, end_date, fit_date, version, plan_type, period):
    """
    get stock performance of given time period for given single stock

    params:
        single_stock_info : dict      # dictionary with keys--[stockId, hr(fillrate), np, tag('good' or 'bad), quantity]
        start_date        : int
        end_date          : int
        fit_date          : tuple     # (fit_date_star, fit_date_sh, fit_date_sz)
        version           : int
        plan_type         : str       # type of trade_plan 'new' or 'uncon'
        period            : str       # 'first' or 'second' or 'lgb'

    """
    stockId = str(single_stock_info['stockId'])
    tag = single_stock_info['tag']
    d_latency = 30
    if stockId.startswith('6'):
        latency = False
        if stockId.startswith('68'):
            fit_date = fit_date[0]
        else:
            fit_date = fit_date[1]
    else:
        latency = True
        fit_date = fit_date[2]

    # get poslimit for stocks
    if period == 'first' or period == 'lgb':
        if stockId.startswith('68'):
            if stockId in special_poslimit_first:
                poslimit_set = special_poslimit_first[stockId]
            elif tag=="good":
                poslimit_set = [200, 500, 800, 1100]
            else:
                poslimit_set = [200, 500]
        else:
            poslimit_set = [100]

    elif period == 'second':
        if stockId.startswith('68'):
            if stockId in special_poslimit_second:
                poslimit_set = special_poslimit_second[stockId]
            else:
                poslimit_set = [200, 500]
        else:
            poslimit_set = [100]

    if plan_type == 'new':
        max_quantity = single_stock_info['quantity']
        poslimit_set = [x for x in poslimit_set if x < max_quantity/8]
        if len(poslimit_set) == 0:
            print(f'{stockId} not enough quantity!!!!')
            return

    print(f'stockId: {stockId}, start-end: {(start_date, end_date)}, period: {period}')
    sys.stdout.flush()
    model_list = []
    for fitdate in fit_date:
        single_model = su.get_stock_plan(fitting_date=fitdate, version=version, stockId=stockId)
        if isinstance(single_model, str):
            print(f'{stockId} : could not find model')
            continue
        model_list.append(single_model)
        print(f'{stockId} : we found this model')
    if len(model_list) == 0:
        return None
    models = pd.concat(model_list)
    # models['para'] = models.apply(
    #     lambda x: x['forecaster_file'].split('(')[1].split(')')[0] if len(x['forecaster_file']) > 105 else
    #     0, axis=1)
    # models = models[models.para == 'thresh=0.05,decay=0.8']
    # models = su.get_stock_plan(fitting_date=fit_date, version=version, stockId=stockId)

    # record performance of each model for certain stock
    for poslimit in poslimit_set:
        result_set = []
        for i in range(len(models)):
            model = models.iloc[i]
            model = pd.DataFrame(model).T
            if version == 'gpmodel':
                forecaster = fu.load_from_file(model.forecaster_file[0])
                predictors = [BaseForecaster.parse_source_config(i) for i in forecaster.to_dict()['params']['slots']]
                if plan_type == 'new':

                    X_and_snapshot = su.get_X_and_snapshot(stockId, start_date, end_date,
                                                           predictors_set=predictors, period=period, latency=d_latency)
                    try:
                        result = su.python_simulation(X_and_snapshot=X_and_snapshot,
                                                      clf=json.loads(model.gp_weights[0]),
                                                      edge=0,
                                                      signal_mult=1,
                                                      poslimit=poslimit,
                                                      max_quantity=max_quantity * 2 * 1.2, latency=latency)
                    except IndexError as e:
                        print(f'{stockId}', e)
                    # try:
                    #     result = su.python_simulation(X_and_snapshot=X_and_snapshot,
                    #                                   clf=json.loads(model.gp_weights[0]),
                    #                                   edge=0,
                    #                                   signal_mult=1,
                    #                                   poslimit=poslimit,
                    #                                   max_quantity=max_quantity * 2 * 1.2, latency=latency,
                    #                                   max_loss=result['avgpnl'])
                    # except IndexError as e:
                    #     print(f'{stockId}', e)

                elif plan_type == 'uncon':

                    X_and_snapshot = su.get_X_and_snapshot(model.stockId[0], start_date, end_date,
                                                           predictors_set=predictors, period=period, latency=d_latency)
                    result = su.python_simulation(X_and_snapshot=X_and_snapshot,
                                                  clf=json.loads(model.gp_weights[0]),
                                                  edge=0,
                                                  signal_mult=1,
                                                  poslimit=poslimit, latency=latency)

                if period == 'first':
                    result['zero_cros'] = result['zero_cros']/0.7
                elif period == 'second':
                    result['zero_cros'] = result['zero_cros']/0.3
                result['forecaster_file'] = model.forecaster_file[0]
            elif plan_type == 'new':
                X_and_snapshot = su.get_X_and_snapshot(stockId, start_date, end_date, version)
                result = su.python_simulation(X_and_snapshot=X_and_snapshot,
                                              clf=lgb.Booster(model_file=model.model_file),
                                              edge=model.edge,
                                              signal_mult=model.mult,
                                              poslimit=poslimit,
                                              max_quantity=max_quantity * 2)
            elif plan_type == 'uncon':
                X_and_snapshot = su.get_X_and_snapshot(stockId, start_date, end_date, version)
                result = su.python_simulation(X_and_snapshot=X_and_snapshot,
                                              clf=lgb.Booster(model_file=model.model_file),
                                              edge=model.edge,
                                              signal_mult=model.mult,
                                              poslimit=poslimit)

            result['symbol'] = stockId
            if version != 'gpmodel':
                result['model_file'] = model.model_file
            result['version'] = version
            result['tag'] = 'bright'
            result_set.append(result)
        results = pd.DataFrame(result_set, columns=result.keys())

        # record simulation performance of stocks as csv
        # remind: plan_type=="new" or "uncon"
        if period == 'lgb':
            file_dir = f'/mnt/hdd_storage_2/stock_plan/{plan_type}_stock_plan/{end_date}/poslimit_{poslimit}'
        else:
            file_dir = f'/mnt/hdd_storage_2/stock_plan/{plan_type}_stock_plan/' \
                                                       f'{period}_stock_plan/{end_date}/poslimit_{poslimit}'
        if not os.path.exists(file_dir):
            os.system("mkdir -p {}".format(file_dir))
        file = os.path.join(file_dir, 'stock_performance_{}.csv'.format(stockId))
        results.to_csv(file, index=False)
    return None


def gen_model_performance(start_date, end_date, fit_date, version, stock_info, plan_type, period='lgb'):
    """
    get stock performance of given time period for all stock in stock_info

    params:
        start_date:int
        end_date:int
        fit_date:tuple              #(fit_date_star, fit_date_sh, fit_date_sz)
        version:int
        stock_info:dataframe       #contains stockId, hr_fillrate, np, quantity and tag(generated from get_stock_info())
        plan_type:str              #type of trade_plan 'new' or 'uncon'
        period:str                  #'first' or 'second' or 'lgb'

    instance:
        gen_model_performance(20201211, 20201224, 20201216, 4, stock_info, 'new')

    """
    stock_info_list = [stock_info.iloc[i].to_dict() for i in range(len(stock_info))]

    for i in multi_run(func=_gen_model_performance,
                       params={'single_stock_info': stock_info_list},
                       static_params={'start_date': start_date,
                                      'end_date': end_date,
                                      'fit_date': fit_date,
                                      'version': version,
                                      'plan_type': plan_type,
                                      'period': period},
                       n_jobs=20):
        pass
    return 'finish'


def gen_stock_plan(date, poslimit, thresh, plan_type, period='lgb', alpha=False):
    """
    read the best-model for given poslimit for certain stock

    models created by gen_model_performance,
    thus gen_model_performance must run before this function

    params:
        date      : int             # end_date from which we read the csv files of stock performance
        poslimit  : int
        thresh    : tuple           # 3-d tuple contains thresh for zero_cros, daily_sharpe and ror
        plan_type : str             # type of trade_plan 'new' or 'uncon'
        period    : str             # 'first' or 'second' or 'lgb'
        alpha     : bool            # whether to contain stocks of alpha combined t0

    instance:
        gen_stock_plan(end_date, 200, (10, 6, 0.2), 'new')
    """
    if period == 'lgb':
        file_dir = f'/mnt/hdd_storage_2/stock_plan/{plan_type}_stock_plan/{date}/poslimit_{poslimit}/*.csv'
    else:
        file_dir = f'/mnt/hdd_storage_2/stock_plan/{plan_type}_stock_plan/' \
                   f'{period}_stock_plan/{date}/poslimit_{poslimit}/*.csv'
    if plan_type == 'new':
        util = 'util_r'
    elif plan_type == 'uncon':
        util = 'util'
    stock_plan = pd.DataFrame()
    if not alpha:
        for file in glob.glob(file_dir):
            stock_performance = pd.read_csv(file, dtype={'stockId': str})
            stock_performance = stock_performance[(stock_performance.zero_cros > thresh[0]) &
                                                  (stock_performance.sharpe_5min > thresh[1]) &
                                                  (stock_performance.ror > thresh[2]) &
                                                  (stock_performance.stockId > '688000')]
            if len(stock_performance) != 0:
                stock_performance = stock_performance.sort_values(by=['avgpnl', util], ascending=False).iloc[0]
                stock_plan = stock_plan.append(stock_performance)
    else:
        for file in glob.glob(file_dir):
            stock_performance = pd.read_csv(file, dtype={'stockId': str})
            stock_performance = stock_performance[(stock_performance.zero_cros > thresh[0]) &
                                                  (stock_performance.sharpe_5min > thresh[1]) &
                                                  (stock_performance.ror > thresh[2]) &
                                                  (stock_performance.stockId < '688000')]
            if len(stock_performance) != 0:
                stock_performance = stock_performance.sort_values(by=[util, 'avgpnl'], ascending=False).iloc[0]
                stock_plan = stock_plan.append(stock_performance)
    return stock_plan



