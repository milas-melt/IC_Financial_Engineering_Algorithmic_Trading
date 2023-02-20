import numpy as np
import pandas as pd
import stat as st
import scipy
import os
import itertools as it
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)

folder_path = "C:\\Users\\Arman\\Desktop\\IC business school\\AlgoTradingTutorial\\Data"
cr_dir = os.getcwd()
os.chdir(folder_path)

ticker_list = ['AMZN','MSFT','INTC','GOOG','AAPL']
n_level = 10
df_stream_t = []
startTrad = 9.5*60*60       # 9:30:00.000 in ms after midnight
endTrad = 16*60*60        # 16:00:00.000 in ms after midnight
day_duration = (endTrad - startTrad)
for ticker in ticker_list:

    stream_file = f"{ticker}_2012-06-21_34200000_57600000_{'message'}_{str(n_level)}"
    stream_file = os.path.join(folder_path,stream_file+'.csv')
    stream_file_headers = ['Time','Type','OrderID','Size','Price','TradeDirection']

    lob_file = f"{ticker}_2012-06-21_34200000_57600000_{'orderbook'}_{str(n_level)}"
    lob_file = os.path.join(folder_path,lob_file+'.csv')
    lob_file_headers_t = [['AskPrice'+str(i+1),'AskSize'+str(i+1),'BidPrice'+str(i+1),'BidSize'+str(i+1)] for i in range(n_level)]
    lob_file_headers = []
    for col_i in lob_file_headers_t:
        lob_file_headers +=col_i

    # csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    # Reading sample files
    stream_df = pd.read_csv(stream_file, header=None,names=stream_file_headers)
    lob_df = pd.read_csv(lob_file, header=None,names=lob_file_headers)




    stream_df = stream_df[stream_df['Time'] >= startTrad]
    stream_df = stream_df[stream_df['Time'] <= endTrad]


    timeIndex = stream_df.index[(stream_df.Time >= startTrad) & (stream_df.Time <= endTrad)]
    lob_df = lob_df[lob_df.index == timeIndex]

    # for i in list(range(0, len(lob_df.columns), 2)):
    #     lob_df[lob_df.columns[i]] = lob_df[
    #                                                                 lob_df.columns[i]] / 10000
    # adding mid Price
    stream_df['MidPrice'] = (((lob_df['AskPrice1'] + lob_df['BidPrice1'])/2)/10_000)
    # adding Spread
    stream_df['BidAskSpread'] =(((lob_df['AskPrice1'] - lob_df['BidPrice1']))/10_000)
    stream_df['Price'] /=10_000

    stream_df['sym'] = ticker
    stream_df['Index'] = stream_df.index

    df_stream_t.append(stream_df)

df_stream = pd.concat(df_stream_t)
df_stream.index = range(len(df_stream))

execution_df = df_stream[df_stream.Type.isin([4,5,6])]
daily_volume = execution_df.groupby('sym')['Size'].sum().reset_index().rename(columns={'Size':'DailyVolume'})
# execution_df = execution_df.merge(daily_volume,on='sym',how='left').drop_duplicates()

trade_duration_in_time = df_stream.groupby(['sym','OrderID']).agg({'Time':['min','max']}).reset_index()
trade_duration_in_time.columns = ['sym','OrderID','Time_min','Time_max']
trade_duration_in_time['Duration'] = (trade_duration_in_time['Time_max'] - trade_duration_in_time['Time_min'])/(day_duration)
execution_df = execution_df.merge(trade_duration_in_time[['sym','OrderID','Duration']],on=['sym','OrderID'],how='left').drop_duplicates()
execution_df = execution_df.sort_values(['sym','OrderID','Type','Time'])
execution_df['Size'] = (execution_df['Size']*execution_df['TradeDirection'])

intraday_volatility_spread = df_stream[['sym','Time','BidAskSpread','MidPrice']].drop_duplicates()
intraday_volatility_spread['Spread'] = 10_000 * intraday_volatility_spread['BidAskSpread']/intraday_volatility_spread['MidPrice']
intraday_volatility_spread = intraday_volatility_spread.sort_values(['sym','Time']).drop_duplicates(['sym','Time'],keep='first')
intraday_volatility_spread['min'] = (intraday_volatility_spread['Time']/60).astype(int)
intraday_avg_spread = intraday_volatility_spread.groupby(['sym']).Spread.mean().reset_index()

intraday_vol = intraday_volatility_spread.groupby(['sym','min']).agg({'MidPrice':['first','last']}).reset_index()
intraday_vol.columns = ['sym','min','first_p','last_p']
intraday_vol['min_retun'] = (intraday_vol['last_p'] - intraday_vol['first_p'])/intraday_vol['first_p']

intraday_vol = intraday_vol.groupby(['sym']).agg({'min_retun':'std','min':'nunique'}).reset_index()
intraday_vol['Sigma'] = intraday_vol['min_retun'] * np.sqrt( intraday_vol['min']) * np.sqrt(252)

vol_spread_df = intraday_vol[['sym','Sigma']].merge(intraday_avg_spread, on='sym',how='left').drop_duplicates()


execution_df_s_0 = execution_df.groupby(['sym','OrderID']).MidPrice.first().reset_index().rename(columns={'MidPrice':'S_0'})
execution_df_Q = execution_df.groupby(['sym','OrderID']).Size.sum().reset_index().rename(columns={'Size':'Q'})

execution_df_s_star = execution_df.copy()
execution_df_s_star['S*'] = execution_df_s_star['Size'] * execution_df_s_star['Price']
execution_df_s_star = execution_df_s_star.groupby(['sym','OrderID'])['S*'].sum().reset_index()

execution_df_data = execution_df_s_0.merge(execution_df_Q,on=['sym','OrderID'],how='left').drop_duplicates()
execution_df_data = execution_df_data.merge(execution_df_s_star,on=['sym','OrderID'],how='left').drop_duplicates()
execution_df_data['S*'] = execution_df_data['S*']/execution_df_data['Q']

execution_df_data = execution_df_data.merge(daily_volume,on=['sym'],how='left').drop_duplicates()
execution_df_data = execution_df_data.merge(vol_spread_df,on=['sym'],how='left').drop_duplicates()

trade_duration_in_time = trade_duration_in_time.groupby(['sym','OrderID'])['Duration'].mean().reset_index()
execution_df_data = execution_df_data.merge(trade_duration_in_time,on=['sym','OrderID'],how='left').drop_duplicates()
execution_df_data['TradeDirection'] = np.where(execution_df_data['Q']>=0,1,-1)
execution_df_data['Slippage'] = 10_000 * execution_df_data['TradeDirection']*((execution_df_data['S*']-execution_df_data['S_0'])/execution_df_data['S_0'])
execution_df_data['pcp'] = np.abs(execution_df_data['Q']/execution_df_data['DailyVolume'])





# # converting the Time(sec) to date time
# stream_df['Time'] = pd.to_timedelta(stream_df['Time'], unit='s')
#
# # lob fillna
# min_val = min(lob_df.min().min(),-9999999999)
# max_val = max(lob_df.max().max(),9999999999)
# for col_i in lob_df.columns.to_list():
#     indx_t = lob_df[col_i].isin([min_val,max_val])
#     lob_df.loc[indx_t,col_i] = np.nan
#
# # selecting the execution data
# execution_df = stream_df[stream_df.EventType.isin([4,5])]





os.chdir(cr_dir)


