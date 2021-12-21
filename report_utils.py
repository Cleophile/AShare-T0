#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import datetime
import numpy as np
import pandas as pd
import stock_plan_utils as sp
import py_doraemon.trading_days as td

def _get_prev_fitting_date(fitting_date):
    current_date = pd.to_datetime(str(fitting_date))
    weekday = current_date.weekday()
    if weekday==2:
        # notice 0 is Monday
        # 2 is Wednesday
        current_date -= datetime.timedelta(days=4)
        return int(current_date.strftime("%Y%m%d"))
    current_date -= datetime.timedelta(days=3)
    return int(current_date.strftime("%Y%m%d"))

def _to_html_with_hierarchy(df, initial_index=None):
    if initial_index is None:
        initial_index = df.index.copy()
    pattern = '<(/)?table.*>'
    index_depth = df.index.nlevels
    tbody = {}
    for i, j in df.groupby(level=0):
        tbody[i] = re.sub(pattern,"", j.to_html(header=False, na_rep='-'))

    tbody = [tbody[i] for i in initial_index]

    if df.columns.empty:
        col_str = ""
    else:
        col_str = "<th>" + "</th><th>".join(df.columns) + "</th>"
    thead = "<thead><tr>\n" + "<th></th>\n" * index_depth + col_str + "</tr></thead>"
    return "<table class=\"dataframe\">\n" + thead + "\n".join(tbody) + "\n</table>"

def _day_verbose(day):
    return pd.to_datetime(str(day)).strftime("%Y.%m.%d")

def _show_simple_table(final_result, single_column=['quant','quantity'],
                       double_column=['price', 'avgpnl', 'ror', 'sharpe_5min', 'daily_sharpe', 'zero_cros', 'trade_qty', 'poslimit']):
    initial_index = final_result.index.get_level_values(0).unique().copy()
    double_dict = {}
    for col in double_column:
        data_var = final_result[[col, col+"_second"]]
        data_var.columns = ['am','pm']
        double_dict[col] = data_var.stack(list(range(data_var.columns.nlevels)))
    single_column_data = final_result[single_column]
    double_column_data = pd.DataFrame(double_dict)
    full_data = double_column_data.join(single_column_data, how="inner", on=list(single_column_data.index.names))[single_column + double_column]

    if 'ror' in full_data.columns:
        full_data['ror'] *= 100
        full_data.rename(columns={'ror':'ror(%)'}, inplace=True)

    full_data = full_data.round(2)
    full_data.rename(columns={'sharpe_5min':"sharpe\n5min", 'daily_sharpe': "sharpe\ndaily"}, inplace=True)
    full_data.index.names = [None for _ in range(full_data.index.nlevels)]
    return _to_html_with_hierarchy(full_data, initial_index)

def daily_report_log(inputs, outputs, output_destinations, log):
    run_date, fit_date_star, start_date, end_date, fit_date_sh, fit_date_sz = inputs
    final_stock_plan, stocks_to_improve, bad_performance = outputs
    final_destination, improve_destination, bad_destination = output_destinations

    final_stock_plan.sort_values("price", ascending=False, inplace=True)
    stocks_to_improve.sort_values("price", ascending=False, inplace=True)

    real_info_day = td.get_prev_trading_day(run_date)
    np_table = sp.get_stock_info(real_info_day, plan_type='show', n_days=1, report_simu=True)
    prev_fitting_date = _get_prev_fitting_date(fit_date_star[0])
    def _daily_simu_wrapper(x):
        try:
            sp.simulation_summary(x['stockId'],
                start_date=run_date, end_date=run_date,
                fitting_date=prev_fitting_date,
                poslimits=[x['poslimit'], x['poslimit_ten'],x['poslimit_second']],
                periods=['half_hour','ten','second'],
                max_quantity=x['max_qty'])
        except Exception as e:
            print("Error in daily report simulation", e)
            return np.nan
    prev_fitting_date_np  = final_stock_plan.apply(_daily_simu_wrapper, axis=1)
    prev_fitting_date_np.name = "np\nprev_fitting"
    np_table = np_table.join(prev_fitting_date_np)

    np_table['ror_r(%)'] = np_table['ror_r'] * 100
    np_table['ror_s(%)'] = np_table['ror_s'] * 100
    np_table = np_table[['np_r', 'np_s', 'np\nprev_fitting', 'ror_r(%)', 'ror_s(%)', 'fr', 'use_ratio', 'quantity', 'price']]
    np_table.sort_values("price", ascending=False, inplace=True)
    np_table.index.name = None

    summary_file = os.path.join(log, "daily_summary.html")

    good_in_plan = final_stock_plan.loc[final_stock_plan.index.difference(stocks_to_improve.index)].copy()
    good_in_plan.sort_values("price", ascending=False, inplace=True)

    with open(summary_file, "w") as summary:
        first_half = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <link rel="stylesheet" type="text/css" href="../style.css">
            <title>New Stock Plan Tables</title>
        </head>
        <body>
        """

        second_half = """
        </body>
        </html>
        """
        summary.write(first_half)
        summary.write(f'''
        <h1 align="center" class="bigHeader"> Daily Stock Plan Summary </h1>
        <h2 align="left" class="smallHeader lightBlueHeader no-bottom-margin">Parameters</h2>
        <div class="parameter-flex-container">
            <div class="parameter-flex-item">
                <h3 class="item-header"> Run Date </h3>
                <p class="description">
                    {_day_verbose(run_date)}
                </p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header"> Plan Date Start </h3>
                <p class="description">
                    {_day_verbose(start_date)} </br>
                </p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header"> Plan Date End </h3>
                <p class="description">
                    {_day_verbose(end_date)}
                </p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header"> Simulation Days </h3>
                <p class="description">
                    5 (Fixed)
                </p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header"> Fitting Date </h3>
                <p class="description">
                    SH:  {", ".join([_day_verbose(i) for i in fit_date_sh])} </br>
                    SZ:  {", ".join([_day_verbose(i) for i in fit_date_sz])} </br>
                    STAR:{", ".join([_day_verbose(i) for i in fit_date_star])}
                </p>
            </div>
        </div>
        ''')

        #  number of stocks we borrowed is : {len(borrow_stock)}
        summary.write(f'''
        <h2 class="smallHeader lightBlueHeader no-bottom-margin"> Stock Pool Information </h2>
        <div class="parameter-flex-container">
            <div class="parameter-flex-item">
                <h3 class="item-header"> Run Date Plan </h3>
                <table class="raw-table description">
                <tbody>
                    <tr>
                        <th> Stocks in plan:</th>
                        <td> {len(final_stock_plan.index.unique())}</td>
                    </tr>
                    <tr>
                        <th> Stocks to improve:</th>
                        <td> {len(stocks_to_improve.index.unique())}</td>
                    </tr>
                    <tr>
                        <th> Bad Stocks:</th>
                        <td> {len(bad_performance.index.unique())}</td>
                    </tr>
                </tbody>
                </table>
            </div>
        </div>
        ''')
        effective_np_table = np_table[(~np.isnan(np_table['np_r'])) & (~np.isnan(np_table['np_s']))][['np_r','np_s','np\nprev_fitting']]
        simu_pnl = sum(effective_np_table['np_s'])
        real_pnl = sum(effective_np_table['np_r'])
        prev_simu_sum = sum(effective_np_table['np\nprev_fitting'])
        summary.write(f'''
        <h2 class="smallHeader lightBlueHeader no-bottom-margin"> One-Day PnL Information, Real vs. Simulation </h2>
        <table class="raw-table description">
        <tbody class="description">
            <tr>
                <th> Simu PnL: </th>
                <td> {simu_pnl:,.2f} </td>
            </tr>
            <tr>
                <th> Real PnL: </th>
                <td> {real_pnl:,.2f} </td>
            </tr>
            <tr>
                <th> Previous fitting_date\n({_day_verbose(prev_fitting_date)}) PnL: </th>
                <td> {prev_simu_sum:,.2f} </td>
            </tr>
        </tbody>
        </table>
        ''')

        summary.write(f'''
        <h2 class="smallHeader lightBlueHeader no-bottom-margin"> Money Used </h2>
        ''')

        total_money = (final_stock_plan['price'] * final_stock_plan['max_qty'] / 2.4).sum()
        summary.write(f"<p class=\"description\"> Total money: {total_money:,.2f} </p>")

        summary.write(f'''
        <h2 class="smallHeader lightBlueHeader"> Special Poslimit Options </h2>
        ''')

        special_stocks = [[i, 'first', j] for i, j in sp.special_poslimit_first.items()]
        special_stocks.extend([[i, 'second', j] for i, j in sp.special_poslimit_second.items()])
        special_stocks = pd.DataFrame(special_stocks, columns=['stocks','period','poslimit'])
        special_stocks.sort_values(['stocks','period'], inplace=True)
        special_counts = (len(special_stocks) + 1) // 2

        special_stocks1 = special_stocks.iloc[:special_counts].copy()
        special_stocks2 = special_stocks.iloc[special_counts:].copy()

        special_stocks1.set_index(['stocks','period'], inplace=True)
        special_stocks1.index.names = [None, None]

        special_stocks2.set_index(['stocks','period'], inplace=True)
        special_stocks2.index.names = [None, None]
        summary.write("<div class=\"table-flex-container\">")
        summary.write("<div class=\"flex-column-table\">")
        summary.write(_to_html_with_hierarchy(special_stocks1, special_stocks1.index.get_level_values(0).unique()))
        summary.write("</div>")
        summary.write("<div class=\"flex-column-table\">")
        summary.write(_to_html_with_hierarchy(special_stocks2, special_stocks2.index.get_level_values(0).unique()))
        summary.write("</div></div>")

        # add page-breaker
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        good_in_plan['poslimit'] = good_in_plan.apply(lambda x: (x['poslimit'], x['poslimit_ten']), axis=1)
        summary.write("<h3 class=\"greenHeader\">&nbsp;Good stocks in plan</h3>")
        summary.write(f"<p class=\"folder\"> Complete chart, including stocks to be improved: {final_destination} </p>")
        summary.write(_show_simple_table(good_in_plan, single_column=[],
                      double_column=['max_qty','price', 'avgpnl', 'ror', 'sharpe_5min', 'daily_sharpe', 'zero_cros', 'trade_qty', 'poslimit']))
        stocks_to_improve['poslimit'] = stocks_to_improve.apply(lambda x: (x['poslimit'],x['poslimit_ten']), axis=1)

        # add page-breaker
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        summary.write("<h3 class=\"yelloHeader\">&nbsp;Stocks needed to be improved in plan</h3>")
        summary.write(f"<p class=\"folder\"> Complete chart: {improve_destination} </p>")
        summary.write(_show_simple_table(stocks_to_improve, single_column=[],
                      double_column=['max_qty','price', 'avgpnl', 'ror', 'sharpe_5min', 'daily_sharpe', 'zero_cros', 'trade_qty', 'poslimit']))

        # add page-breaker
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        summary.write("<h3 class=\"redHeader\">&nbsp;Bad Stocks (not in the plan)</h3>")
        summary.write(f"<p class=\"folder\"> Complete chart: {bad_destination} </p>")
        if bad_performance.empty:
            summary.write("<br />" * 10 + "<h1 class=\"bigHeader\">No bad stocks.</p>")
            summary.write("<h1 class=\"bigHeader\">Relax.</h1>")
        else:
            bad_performance.sort_values(["price", 'poslimit'], ascending=[False, True], inplace=True)
            summary.write(_show_simple_table(bad_performance, single_column=[],
                                       double_column=['price', 'avgpnl', 'ror', 'sharpe_5min', 'daily_sharpe', 'zero_cros', 'trade_qty']))

        # add page-breaker
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        summary.write("<h3 class=\"purpleHeader\">&nbsp; All stocks real-simulation comparison</h3>")
        real_simu_file = os.path.join(sp.StockInfoCacheFolder, f"real_simu_shstar_{real_info_day}_True.xlsx")
        summary.write(f"<p class=\"folder\"> Complete chart: {real_simu_file} </p>")
        summary.write(_to_html_with_hierarchy(np_table))

        summary.write(second_half)

def borrow_summary(inputs, outputs, output_destinations, log):
    # unpack data
    plan_date, fitting_date, percentile = inputs
    good, new, bad, final_result = outputs
    good_destination, new_destination, bad_destination, final_result_destination = output_destinations

    start_date = td.get_trading_days_before(plan_date, 10)[-1]

    # calculate overall pnl
    def _borrow_simu_wrapper(x):
        try:
            return sp.simulation_summary(x['stockId'], start_date=start_date, end_date=plan_date,
                               fitting_date=x['forecaster_file'][34:42],
                               poslimits=[x['poslimit'], x['poslimit_ten'],x['poslimit_second']],
                               periods=['half_hour','ten','second'],
                               max_quantity=x['quant'] * 2)
        except Exception as e:
            print("Error in Borrow report generation", e)
            return np.nan
    dta = final_result.apply(_borrow_simu_wrapper, axis=1)
    dta = pd.concat(dta.to_list())
    pnl = dta['avgpnl'].sum()

    first_half = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css" href="../style.css">
        <title>Borrow Stock Plan</title>
    </head>
    <body>
    """

    second_half = """
    </body>
    </html>
    """

    summary_file = os.path.join(log,"borrow_summary.html")
    with open(summary_file, "w") as summary:
        summary.write(first_half)
        summary.write('''
        <h1 align="center" class="bigHeader"> Borrow Stock Plan Summary</h1>
        ''')
        plan_date_str = _day_verbose(plan_date)
        if isinstance(fitting_date, list):
            fitting_date_str = ", ".join([_day_verbose(i) for i in fitting_date])
        else:
            fitting_date_str = _day_verbose(fitting_date)

        summary.write(f'''
        <h3 align="left" class="smallHeader lightBlueHeader no-bottom-margin">Parameters</h3>
        <div class="parameter-flex-container">
            <div class="parameter-flex-item">
                <h3 class="item-header">plan_date</h3>
                <p class="description">{plan_date_str}</p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header">fitting_date</h3>
                <p class="description">{fitting_date_str}</p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header">percentile</h3>
                <p class="description">{percentile}% </p>
            </div>
            <div class="parameter-flex-item">
                <h3 class="item-header">Simulation History</h3>
                <p class="description">5 (Fixed)</p>
            </div>
        </div>
        ''')

        summary.write(f"""
        <h3 class="smallHeader lightBlueHeader no-bottom-margin">Stock Pool Information</h3>
        <p class="description">Total stocks on the plan: {len(set(final_result.index))}, including {len(set(good.index))} old ones and {len(set(new.index))} new ones. </p>
        <p class="description">Total stocks already have: {len(set(good.index)) + len(set(bad.index))} </p>
        <p class="description">Total bad stocks: {len(set(bad.index))}</p>
        """)

        summary.write(f"""
        <h3 class="smallHeader lightBlueHeader no-bottom-margin">PnL Information</h3>
        <p class="description">Total average PnL: {pnl:,.2f}</p>
        """)

        summary.write('''
        <h3 align="left" class="smallHeader lightBlueHeader"> Money Summary </h3>
        ''')

        good_money = good.money.sum()
        new_money = new.money.sum()
        old_money = sum(good['quantity'] * good['price']) + sum(bad['quantity'] * bad['price'])

        summary.write(f'''
        <table class="raw-table description">
        <tbody class="description">
            <tr>
                <th>Total money we may use:</th>
                <td>{good_money + new_money:,.2f}</td>
            </tr>
            <tr>
                <th>Money we may use for stocks we have:</th>
                <td>{good_money:,.2f}</td>
            </tr>
            <tr>
                <th>Money we may use for stocks we do not have:</th>
                <td>{new_money:,.2f}</td>
            </tr>
            <tr>
                <th>Previously used money:</th>
                <td>{old_money:,.2f}</td>
            </tr>
            <tr>
                <th>Delta:</th>
                <td>{good_money + new_money - old_money:+,.2f}</td>
            </tr>
        </tbody>
        </table>
        ''')

        summary.write('''
        <h3 align="left" class="smallHeader lightBlueHeader"> Special Poslimit Options </h3>
        ''')

        special_stocks = [[i, 'first', j] for i, j in sp.special_poslimit_first.items()]
        special_stocks.extend([[i, 'second', j] for i, j in sp.special_poslimit_second.items()])
        special_stocks = pd.DataFrame(special_stocks, columns=['stocks','period','poslimit'])
        special_stocks.sort_values(['stocks','period'], inplace=True)
        special_counts = (len(special_stocks) + 1) // 2

        special_stocks1 = special_stocks.iloc[:special_counts].copy()
        special_stocks2 = special_stocks.iloc[special_counts:].copy()

        special_stocks1.set_index(['stocks','period'], inplace=True)
        special_stocks1.index.names = [None, None]

        special_stocks2.set_index(['stocks','period'], inplace=True)
        special_stocks2.index.names = [None, None]
        summary.write("<div class=\"table-flex-container\">")
        summary.write("<div class=\"flex-column-table\">")
        summary.write(_to_html_with_hierarchy(special_stocks1, special_stocks1.index.get_level_values(0).unique()))
        summary.write("</div>")
        summary.write("<div class=\"flex-column-table\">")
        summary.write(_to_html_with_hierarchy(special_stocks2, special_stocks2.index.get_level_values(0).unique()))
        summary.write("</div></div>")

        # page-break
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        summary.write('''
        <h3 class="greenHeader">&nbsp;Borrowed Stocks</h3>
        ''')
        summary.write(f"<p class=\"folder\"> Complete chart: {final_result_destination} </p>")
        final_result.sort_values("price", ascending=False, inplace=True)
        summary.write(_show_simple_table(final_result))

        # page-break
        summary.write('''
        <p class="pageBreak"></p>
        ''')

        summary.write('''
        <h3 class="redHeader">&nbsp;Bad Stocks</h3>
        ''')
        summary.write(f"<p class=\"folder\"> Complete chart: {bad_destination} </p>")
        bad.sort_values("price", ascending=False, inplace=True)
        summary.write(_show_simple_table(bad,single_column=['quantity']))
        summary.write(second_half)

if __name__ == "__main__":
    inputs = [20210325, [20210310], 20210318, 20210324, [20210308], [20210308]]
    final_stock_plan = pd.read_csv("/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210324/stock_plan.csv", dtype={"symbol":str})
    final_stock_plan.set_index("symbol", inplace=True)
    bad_performance = pd.read_csv("/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210324/filtered_out_stocks.csv", dtype={"symbol":str})
    if not bad_performance.empty:
        bad_performance.set_index(["symbol",'poslimit'], inplace=True)
    #  bad_performance = pd.DataFrame()
    stocks_to_improve = pd.read_csv("/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210324/stocks_to_improve.csv", dtype={"symbol":str})
    stocks_to_improve.set_index("symbol", inplace=True)
    outputs = final_stock_plan, stocks_to_improve, bad_performance
    log = "./report_cache/"

    output_destinations = [
        '/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210311/stock_plan.csv',
        "/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210311/stocks_to_improve.csv",
        "/mnt/hdd_storage_2/stock_plan/new_stock_plan/20210324/filtered_out_stocks.csv"
    ]
    daily_report_log(inputs,outputs,output_destinations,log)
