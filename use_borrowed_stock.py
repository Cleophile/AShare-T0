#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import stock_plan_utils as sp
import py_doraemon.trading_days as td
from report_utils import daily_report_log
import os
import argparse
import glob
import sys
sys.stdout.flush()

parser = argparse.ArgumentParser(description='generate daily stock plan, number of stocks in this stock plan, number of'
                                             'stocks to improve and number of stocks we borrowed will be printed on the'
                                             'console, there will be three csv files as outcome--stock_plan, stocks_to'
                                             '_improve and stocks not in this stock plan')
parser.add_argument('--run_date', help='date to use this stock plan', type=int)
parser.add_argument('--fit_date_star', help='fitting date of gpmodel we use in this stock plan for sh_star', type=int,
                    default=[20210224], nargs='+')
parser.add_argument('--plan_type', help="type of stock plan: 'new' or 'uncon'", default='new')
# parser.add_argument('--version', help='version', default='gpmodel')
parser.add_argument('--start_date', help='start date from which we run historical simulation, if not input, start date '
                                         'is the fifth day before run_date, which means the simulation window is 5 days'
                                         , default=None)
parser.add_argument('--end_date', help='end date to which we run historical simulation, if not input, end date is '
                                       'the day before run_date', default=None)
parser.add_argument('--fit_date_sh', help='fitting date of gpmodel we use in this stock plan for sh_main', type=int,
                    default=[20210310], nargs='+')
parser.add_argument('--fit_date_sz', help='fitting date of gpmodel we use in this stock plan for sz', type=int,
                    default=[20210310], nargs='+')
parser.add_argument('--log', help="Generate PDF report to the designated path if not None and plan_type==new", default=None)

def generate(run_date, fit_date_star, plan_type='new', version='gpmodel', start_date=None,
             end_date=None, fit_date_sh=[20210310], fit_date_sz=[20210310], log=None):
    """
    generate stock plan by choosing the best model on simulation
    if plan_type ==
        'new'   : saves file of stock_plan, including the ones need to be improved, and performances of excluded ones
        'uncon' : saves file of stock_plan, including the ones need to be improved

    params:
        run_date      : int
        fit_date_star : int           # fit_date of model we use for star stocks
        plan_type     : str           # type of trade_plan 'new' or 'uncon'
        version       : str           # {"gpmodel"}, version of model used for simulation
        start_date    : int YYYYMMDD  # start date of simulation; if None simulation will run from 5 days ago
        end_date      : int YYYYMMDD  # end date of simulation
        fit_date_sh   : int YYYYMMDD  # fit_date of model we use for sh main stocks
        fit_date_sz   : int YYYYMMDD  # fit_date of model we use for sz stocks
    """
    if start_date is None:
        start_date = td.get_trading_days_before(run_date, 6)
        end_date = start_date[1]
        start_date = start_date[-1]

    # outsource basic stock info for gen_model_performance
    # get trading information of last several days tag with good / bad
    stock_info = sp.get_stock_info(run_date, plan_type=plan_type)

    # generate and save model performance
    fit_date = (fit_date_star, fit_date_sh, fit_date_sz)
    sp.gen_model_performance(start_date, end_date, fit_date, version, stock_info, plan_type, period='first')
    sp.gen_model_performance(start_date, end_date, fit_date, version, stock_info, plan_type, period='second')

    # get the best-model for all poslimit
    if plan_type == 'new':
        ror_bar = 0.15
        util = 'util_r'
    elif plan_type == 'uncon':
        ror_bar = 0.2
        util = 'util'
    else:
        raise KeyError("plan_type must be \"new\" or \"uncon\"")

    stock_plan_1100 = sp.gen_stock_plan(end_date, 1100, (10, 10, ror_bar), plan_type, period='first')
    stock_plan_800 = sp.gen_stock_plan(end_date, 800, (10, 10, ror_bar), plan_type, period='first')
    stock_plan_500 = sp.gen_stock_plan(end_date, 500, (10, 10, ror_bar), plan_type, period='first')
    stock_plan_200 = sp.gen_stock_plan(end_date, 200, (10, 10, ror_bar), plan_type, period='first')

    best_stock_plan = pd.DataFrame()
    best_stock_plan = best_stock_plan.append(stock_plan_1100)
    best_stock_plan = best_stock_plan.append(stock_plan_800)
    best_stock_plan = best_stock_plan.append(stock_plan_500)
    best_stock_plan = best_stock_plan.append(stock_plan_200)

    if plan_type == 'new':
        stock_plan_200_2 = sp.gen_stock_plan(end_date, 200, (5, 8, 0.05), plan_type, period='first')
        stock_plan_100 = sp.gen_stock_plan(end_date, 100, (10, 10, 0.15), plan_type, period='first', alpha=True)

        best_stock_plan = best_stock_plan.append(stock_plan_200_2)
        best_stock_plan = best_stock_plan.append(stock_plan_100)

    # choose the best model among poslimits
    best_stock_plan['symbol'] = best_stock_plan['stockId']
    all_stock_list = best_stock_plan.symbol.drop_duplicates().reset_index(drop=True)
    final_stock_plan1 = pd.DataFrame()

    for stockId in all_stock_list:
        if str(stockId).startswith('68'):
            temp = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['avgpnl', util], ascending=False)
            final_stock_plan1 = final_stock_plan1.append(temp.iloc[0])
        else:
            temp1 = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['poslimit'])
            temp1 = temp1[temp1.trade_qty > 1.5*temp1.max_qty]
            if len(temp1) > 0:
                final_stock_plan1 = final_stock_plan1.append(temp1.iloc[0])
            else:
                temp2 = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['avgpnl', util], ascending=False)
                final_stock_plan1 = final_stock_plan1.append(temp2.iloc[0])

    # the same process for second period
    # stock_plan_1400 = sp.gen_stock_plan(end_date, 1400, (10, 10, ror_bar), plan_type, period='second')
    stock_plan_1100 = sp.gen_stock_plan(end_date, 1100, (10, 10, ror_bar), plan_type, period='second')
    stock_plan_800 = sp.gen_stock_plan(end_date, 800, (10, 10, ror_bar), plan_type, period='second')
    stock_plan_500 = sp.gen_stock_plan(end_date, 500, (10, 10, ror_bar), plan_type, period='second')
    stock_plan_200 = sp.gen_stock_plan(end_date, 200, (10, 10, ror_bar), plan_type, period='second')

    best_stock_plan = pd.DataFrame()
    best_stock_plan = best_stock_plan.append(stock_plan_1100)
    best_stock_plan = best_stock_plan.append(stock_plan_800)
    best_stock_plan = best_stock_plan.append(stock_plan_500)
    best_stock_plan = best_stock_plan.append(stock_plan_200)

    if plan_type == 'new':
        stock_plan_200_2 = sp.gen_stock_plan(end_date, 200, (5, 8, 0.05), plan_type, period='second')
        stock_plan_100 = sp.gen_stock_plan(end_date, 100, (10, 10, 0.15), plan_type, period='second', alpha=True)

        best_stock_plan = best_stock_plan.append(stock_plan_200_2)
        best_stock_plan = best_stock_plan.append(stock_plan_100)

    # choose the best model among poslimits
    best_stock_plan['symbol'] = best_stock_plan['stockId']
    all_stock_list = best_stock_plan.symbol.drop_duplicates().reset_index(drop=True)
    final_stock_plan2 = pd.DataFrame()

    for stockId in all_stock_list:
        if str(stockId).startswith('68'):
            temp = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['avgpnl', util], ascending=False)
            final_stock_plan2 = final_stock_plan2.append(temp.iloc[0])
        else:
            temp1 = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['poslimit'])
            temp1 = temp1[temp1.trade_qty > 1.5*temp1.max_qty]
            if len(temp1) > 0:
                final_stock_plan2 = final_stock_plan2.append(temp1.iloc[0])
            else:
                temp2 = best_stock_plan[best_stock_plan.symbol == stockId].sort_values(by=['avgpnl', util], ascending=False)
                final_stock_plan2 = final_stock_plan2.append(temp2.iloc[0])

    final_stock_plan1 = final_stock_plan1.set_index('symbol', drop=False)
    final_stock_plan2 = final_stock_plan2.set_index('symbol', drop=False)
    final_stock_plan = final_stock_plan1.join(final_stock_plan2, rsuffix='_second')
    final_stock_plan['poslimit_ten'] = final_stock_plan[['poslimit']].apply(lambda x: 800 if x['poslimit'] > 800 and
                                       x.index > '688000' else min(500, x['poslimit']), axis=1)
    final_stock_plan['maxloss'] = final_stock_plan.avgpnl

    # save as csv files
    file_dir = f'/mnt/hdd_storage_2/stock_plan/{plan_type}_stock_plan/{end_date}'
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    final_stock_plan.to_csv(os.path.join(file_dir, 'stock_plan.csv'), index=False)
    final_stock_plan.to_csv(f'./stock_plan_{run_date}.csv', index=False)

    # second standard for stocks chosen in stock_plan
    stocks_to_improve = final_stock_plan[(final_stock_plan.ror < 0.2) | (final_stock_plan.zero_cros < 15) |
                                         (final_stock_plan.sharpe_5min < 15) | (final_stock_plan.ror_second < 0.2)
                                         | (final_stock_plan.zero_cros_second < 15) |
                                         (final_stock_plan.sharpe_5min_second < 15)]
    stocks_to_improve.to_csv(os.path.join(file_dir, 'stocks_to_improve.csv'), index=False)
    stocks_to_improve.to_csv(f'./stocks_to_improve_{run_date}.csv', index=False)

    # get the performance of stocks not chosen in stock_plan
    if plan_type == 'new':
        borrow_stock = pd.read_json(f'http://192.168.11.109:3000/api/tops/loans?date={run_date}', dtype={'symbol': str})
        borrow_stock = borrow_stock[borrow_stock.symbol > '600000']
        borrow_stock = borrow_stock.groupby('symbol').sum()
        final_stock_plan = final_stock_plan.set_index('symbol')
        compare = borrow_stock.join(final_stock_plan, rsuffix='_sp')

        bad_stock = compare[np.isnan(compare.poslimit_ten)]

        bad_per = []
        for item in bad_stock.index:
            files1 = glob.glob(f'/mnt/hdd_storage_2/stock_plan/new_stock_plan/first_stock_plan/{end_date}/*/stock_performance_{item}.csv')
            files2 = glob.glob(f'/mnt/hdd_storage_2/stock_plan/new_stock_plan/second_stock_plan/{end_date}/*/stock_performance_{item}.csv')

            stock_plan_f = []
            if len(files1) > 0:
                for file in files1:
                    stock_plan_f.append(pd.read_csv(file, dtype={"stockId" : str, "symbol" : str}))
                stock_plan_f = pd.concat(stock_plan_f, sort=True)
            else:
                stock_plan_f = pd.DataFrame(columns=['symbol', 'poslimit'])

            stock_plan_s = []
            if len(files2) > 0:
                for file in files2:
                    stock_plan_s.append(pd.read_csv(file, dtype={"stockId" : str, "symbol" : str}))
                stock_plan_s = pd.concat(stock_plan_s, sort=True)
            else:
                stock_plan_s = pd.DataFrame(columns=['symbol', 'poslimit'])

            stock_plan_f.set_index(['symbol','poslimit'], inplace=True)
            stock_plan_s.set_index(['symbol','poslimit'], inplace=True)
            stock_plan = stock_plan_f.join(stock_plan_s, how='outer', rsuffix='_second')
            bad_per.append(stock_plan)
        if len(bad_per) > 0:
            bad_per = pd.concat(bad_per, sort=True)
        else:
            bad_per = pd.DataFrame()
        bad_per.to_csv(os.path.join(file_dir, 'filtered_out_stocks.csv'))
        bad_per.to_csv(f'./filtered_out_stocks_{run_date}.csv')

        # Notice report generation must be place after data saving
        # report generation may be destructive to the original data
        if log is not None:
            daily_report_log((run_date, fit_date_star, start_date,
                end_date, fit_date_sh, fit_date_sz),(final_stock_plan, stocks_to_improve, bad_per),
                (os.path.join(file_dir, 'stock_plan.csv'),os.path.join(file_dir, 'stocks_to_improve.csv'),os.path.join(file_dir, 'filtered_out_stocks.csv')), log)

    return 'finish'


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'addscript':
        pass
    else:
        print(sys.argv)
        args = parser.parse_args()
        generate(run_date=args.run_date, fit_date_star=args.fit_date_star,
                 plan_type=args.plan_type, start_date=args.start_date, end_date=args.end_date,
                 fit_date_sh=args.fit_date_sh, fit_date_sz=args.fit_date_sz, log=args.log)
