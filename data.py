import fredapi as fa
import pandas as pd
import config
import pandas_datareader.data as web
import numpy as np


def get_stock_data_yahoo(tickers: list, date_from: str, date_to: str, interval: str):
    """
    This function download data from Yahoo Finance by given a list of tickers, date range & data frequency.
    :param tickers: list of tickers
    :param date_from: str, start date of the data time series, for example '2000-01-01'
    :param date_to: str, end date of the data time series, for example '2000-01-01'
    :param interval: str, frequency of the data, for example: 'd' as daily data
    :return: DataFrame, the stock price time series
    """
    stock_list = []
    for i in range(len(tickers)):
        # Read data from Yahoo
        data = web.get_data_yahoo(tickers[i], start=date_from, end=date_to, interval=interval)
        # Add stock ticker column
        data['ticker'] = tickers[i]
        # Append current stock to the list
        stock_list.append(data)

    # Combine all stocks in one DF
    data = pd.concat(stock_list).reset_index()

    # Data Cleaning
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.sort_values(by=['ticker', 'Date'])
    return data


def daily_to_monthly_yahoo(df):
    """
    This function convert daily time series Yahoo Finance data to monthly time series data.
    :param df: DataFrame, daily stock data
    :return: DataFrame, monthly stock data
    """
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month

    # Get month end date
    month_end = df.groupby(['year', 'month'])['Date'].max()
    df = df[df["Date"].isin(month_end)]
    df['date'] = df['Date'] - pd.offsets.MonthBegin(1)
    return df.drop(columns=['year', 'month', 'Date'])


def data_employment():
    """
    Download employment data (Nonfarm payroll) from Fed and calculate the 12m vs 1m, 3m, 6m % changes.
    :return: processed employment data, unit in %
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)

    # 1. Employment, unit in %
    data_emp = fred.get_series(config.TOTAL_NONFARM).rename('nonfarm_payroll').reset_index().rename(
        columns={'index': 'date'})

    for i in [1, 3, 6, 12]:
        data_emp[f'payroll_delta_{i}m'] = data_emp['nonfarm_payroll'].pct_change(i) * 100

    # Get the difference
    data_emp['payroll_diff_12m_1m'] = data_emp['payroll_delta_12m'] - data_emp['payroll_delta_1m']
    data_emp['payroll_diff_12m_3m'] = data_emp['payroll_delta_12m'] - data_emp['payroll_delta_3m']
    data_emp['payroll_diff_12m_6m'] = data_emp['payroll_delta_12m'] - data_emp['payroll_delta_6m']

    data_emp = data_emp.set_index('date')
    return data_emp


def data_monetary():
    """
    Download monetary policy data (Fed fund rate) from Fed.
    Calculate the current policy rate vs 1m, 3m, 6m & 12m ago changes.
    :return: processed monetary data, unit in %
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)

    # 2. Monetary, unit in %
    data_monetary = fred.get_series(config.FED_FUNDS_RATE_MONTHLY).rename('policy_rate').reset_index().rename(
        columns={'index': 'date'})
    for i in [1, 3, 6, 12]:
        data_monetary[f'policy_rate_{i}m_ago'] = data_monetary['policy_rate'].shift(i)
        data_monetary[f'policy_rate_delta_{i}m'] = data_monetary['policy_rate'] - data_monetary[f'policy_rate_{i}m_ago']

    data_monetary = data_monetary.set_index('date')
    return data_monetary


def data_inflation():
    """
    Download inflation data (CPI) from Fed.
    Calculate the  1m, 3m, 6m & 12m CPI %change.
    :return: processed inflation data, unit in %
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)
    # 3. Inflation
    data_inflation = fred.get_series(config.CPI_US_ADJUSTED['ALL_URBAN']).rename('CPI').reset_index().rename(
        columns={'index': 'date'})

    for i in [1, 3, 6, 12]:
        data_inflation[f'CPI_delta_{i}m'] = data_inflation['CPI'].pct_change(i) * 100

    data_inflation = data_inflation.set_index('date')
    return data_inflation


def data_yield_curve():
    """
    Download yield curve data from Fed.
    Calculate current 10y yield vs 1, 3, 6, 12m ago difference.
    Calculate the 10y vs 3m yield spread.
    :return: processed yield curve data, unit in %
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)
    # 4.1 Bond Market - 10y Treasury Yield
    data_10y = fred.get_series(config.TREASURY_MONTHLY['10Y']).rename('yield_10y').reset_index().rename(
        columns={'index': 'date'}).set_index('date')
    data_3m = fred.get_series(config.TBILL_MONTHLY['3M']).rename('yield_3m').reset_index().rename(
        columns={'index': 'date'}).set_index('date')
    yield_curve = data_10y.join(data_3m)

    for i in [1, 3, 6, 12]:
        yield_curve[f'yield_10y_{i}m_ago'] = yield_curve['yield_10y'].shift(i)
        yield_curve[f'yield_10y_delta{i}m'] = yield_curve['yield_10y'] - yield_curve[f'yield_10y_{i}m_ago']

    # 4.2 Bond Market - Yield Spread
    yield_curve['yield_spread'] = yield_curve['yield_10y'] - yield_curve['yield_3m']
    return yield_curve


def data_stock_market():
    """
    Download SP500 from Yahoo Finance, convert to monthly frequency.
    Calculate SP500 1m, 3m, 6m & 12m %changes.
    :return: processed stock market data, unit in %
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)
    # S&P 500
    # data_stock = get_stock_data_yahoo(['^GSPC'],
    #                                   date_from='1920-01-01',
    #                                   date_to='2022-03-31',
    #                                   interval='d')[['Date', 'Adj Close']]
    data_stock = pd.read_csv('sp500_daily_data.csv')
    data_stock['Date'] = pd.to_datetime(data_stock['Date'])

    data_stock = daily_to_monthly_yahoo(data_stock).rename(columns={'Date': 'date',
                                                                    'Adj Close': 'sp500'})
    for i in [1, 3, 6, 12]:
        data_stock[f'sp500_delta_{i}m'] = data_stock['sp500'].pct_change(i) * 100

    data_stock = data_stock.set_index('date')
    return data_stock


def data_volatility():
    """
    Use SP 500 daily return standard deviation as a proxy for volatility, calculated based on 30 days window.
    Convert the daily series to monthly series.
    :return: volatility monthly data.
    """
    # data_stock = get_stock_data_yahoo(['^GSPC'],
    #                                   date_from='1920-01-01',
    #                                   date_to='2022-03-31',
    #                                   interval='d')[['Date', 'Adj Close']]
    data_stock = pd.read_csv('sp500_daily_data.csv')
    data_stock['Date'] = pd.to_datetime(data_stock['Date'])

    data_stock['return'] = data_stock['Adj Close'].pct_change() * 100
    data_stock['vol_30d'] = data_stock['return'].rolling(30).std()
    # Transform to monthly data
    data_vol = daily_to_monthly_yahoo(data_stock)[['date', 'vol_30d']].set_index('date')
    return data_vol


def data_recession():
    """
    Download historical recession data from Fed.
    Get recession indicator that will happen in 3m, 6m & 12m.
    :return: recession data -- binary variable, monthly frequency.
    """
    # Connect API
    fred = fa.Fred(api_key=config.FRED_API_KEY)

    # Dependent Variable - Y: Recession Indicator
    recession = fred.get_series(config.RECESSION_MONTHLY).rename('recession').reset_index().rename(
        columns={'index': 'date'})

    # Recession in 3m, 6m, 12m
    for period in ['3m', '6m', '12m']:
        month = int(period[:-1])
        recession[f'recession_{period}'] = np.where(recession['recession'].shift(-month) == 1, 1, 0)

    recession = recession.set_index('date')
    return recession


def save_all_data(directory):
    """
    Calculate all the potential economic features & labels.
    Save the data to a csv file by specify the directory.
    """
    # X
    df_emp = data_employment()
    df_monetary = data_monetary()
    df_inflation = data_inflation()
    yield_curve = data_yield_curve()
    df_stock = data_stock_market()
    df_vol = data_volatility()
    # Y
    df_recession = data_recession()

    # Join X & Y to match index
    all_data = df_emp.join(df_monetary).join(df_inflation).join(yield_curve).join(df_stock).join(df_vol).join(df_recession)
    # all_data.to_csv('data_folder/all_data.csv')
    all_data.to_csv(directory)
