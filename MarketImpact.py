import numpy as np
import pandas as pd
import stat as st
import scipy
import os
import itertools as it
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.width', 0)
pd.set_option('display.expand_frame_repr', False)

folder_path = "C:\\Users\\Arman\\Desktop\\IC business school\\Data\\LOBSTER\\"
cr_dir = os.getcwd()
os.chdir(folder_path)

# wiki to see the columns of the data
# https://lobsterdata.com/info/DataStructure.php
# https://www.r-bloggers.com/2019/12/converting-lobster-demo-r-code-into-python/
# Time: Seconds after midnight with decimal precision of at least milliseconds and up to nanoseconds depending on the period requested
# Event Type:
# 1: Submission of a new limit order
# 2: Cancellation (partial deletion of a limit order)
# 3: Deletion (total deletion of a limit order)
# 4: Execution of a visible limit order
# 5: Execution of a hidden limit order
# 6: Indicates a cross trade, e.g. auction trade
# 7: Trading halt indicator (detailed information below)
# Order ID: Unique order reference number
# Size: Number of shares
# Price: Dollar price times 10000 (i.e. a stock price of $91.14 is given by 911400)
# Direction:
# -1: Sell limit order
# 1: Buy limit order
# Note: Execution of a sell (buy) limit order corresponds to a buyer (seller) initiated trade, i.e. buy (sell) trade.

# level.
#
# The term 'level' refers to occupied price levels. The difference between two levels in the LOBSTER output is not necessarily the minimum tick size.
#
# message and orderbook file.
#
# The 'message' and 'orderbook' files can be viewed as matrices of size (Nx6) and (Nx(4xNumLevel)), respectively, where N is the number of events in the requested price range and NumLevel is the number of levels requested.
#
# The k-th row in the 'message' file describes the limit order event causing the change in the limit order book from line k-1 to line k in the 'orderbook' file.
#
# Consider the 'message' and 'orderbook' figures above. The limit order deletion (event type 3) in the second line of the 'message' file removes 100 shares from the ask side at price 118600. The change in the 'orderbook' file from line one to two corresponds to this removal of liquidity. The volume available at the best ask price of 118600 drops from 9484 to 9384 shares.
#
# period available.
#
# The initial period for which limit order books are available covers all trading days on NASDAQ from January 6th 2009 to the day before yesterday. In the future, we aim to increase the available period further into the past.
#
# unoccupied price levels.
#
# When the selected number of levels exceeds the number of levels available, the empty order book positions are filled with dummy information to guarantee a symmetric output. The extra bid and/or ask prices are set to -9999999999 and 9999999999, respectively. The corresponding volumes are set to 0.
#
# trading halts.
#
# When trading halts, a message of type '7' is written into the 'message' file. The corresponding price and trade direction are set to '-1' and all other properties are set to '0'. Should the resume of quoting be indicated by an additional message in NASDAQ's Historical TotalView-ITCH files, another message of type '7' with price '0' is added to the 'message' file. Again, the trade direction is set to '-1' and all other fields are set to '0'. When trading resumes a message of type '7' and price '1' (Trade direction '-1' and all other entries '0') is written to the 'message' file. For messages of type '7', the corresponding order book rows contain a duplication of the preceding order book state.

# selecting the files
ticker = 'AMZN'
n_level = 10
stream_file = f"{ticker}_2012-06-21_34200000_57600000_{'message'}_{str(n_level)}"
stream_file = os.path.join(folder_path,stream_file+'.csv')
stream_file_headers = ['Time','Type','OrderID','Size','Price','TradeDirection']

lob_file = f"{ticker}_2012-06-21_34200000_57600000_{'orderbook'}_{str(n_level)}"
lob_file = os.path.join(folder_path,lob_file+'.csv')
lob_file_headers_t = [['AskPrice'+str(i+1),'AskSize'+str(i+1),'BidPrice'+str(i+1),'BidSize'+str(i+1)] for i in range(n_level)]
lob_file_headers = []
for col_i in lob_file_headers_t:
    lob_file_headers +=col_i

csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
# Reading sample files
stream_df = pd.read_csv(stream_file, header=None,names=stream_file_headers)
lob_df = pd.read_csv(lob_file, header=None,names=lob_file_headers)


startTrad = 9.5*60*60       # 9:30:00.000 in ms after midnight
endTrad = 16*60*60        # 16:00:00.000 in ms after midnight

stream_df = stream_df[stream_df['Time'] >= startTrad]
stream_df = stream_df[stream_df['Time'] <= endTrad]

tradingHaltIdx = stream_df.index[
    (stream_df.Type == 7) & (stream_df.TradeDirection == -1)]
tradeQuoteIdx = stream_df.index[
    (stream_df.Type == 7) & (stream_df.TradeDirection == 0)]
tradeResumeIdx = stream_df.index[
    (stream_df.Type == 7) & (stream_df.TradeDirection == 1)]
if (len(tradingHaltIdx) == 0 | len(tradeQuoteIdx) == 0 | len(tradeResumeIdx) == 0):
    print("No trading halts detected.")

if (len(tradingHaltIdx) != 0):
    print("Data contains trading halt! at time stamp(s): ");
    print(list(tradingHaltIdx))

if (len(tradeQuoteIdx) != 0):
    print(" Data contains quoting message! at time stamp(s)");
    print(list(tradeQuoteIdx))

if (len(tradeResumeIdx) != 0):
    print(" Data resumes trading! at time stamp(s) ");
    print(list(tradeResumeIdx))

# Define interval length
freq = 5 * 60  # Interval length in ms 5 minutes
# Number of intervals from 9:30 to 4:00
noint = int((endTrad - startTrad) / freq)
stream_df.index = range(0, len(stream_df), 1)
# Variables for 'for' loop
j = 0
l = 0
bound = []  # Variable for inverval bound
visible_count = []  # visible_count calculates the number of visible trades in an interval of 5 min
hidden_count = []  # hidden_count calculates the number of visible trades in an interval of 5 min
visible_size = []  # Total volume of visible trades in an interval of 5 minutes
hidden_size = []  # Total volume of hidden trades in an interval of 5 minutes
# Set Bounds for Intraday Intervals
bound = []
for j in range(0, noint):
    bound.append(startTrad + j * freq)
# _____________________________________________________________________________
#
# Plot - Number of Executions and Trade Volume by Interval
# _____________________________________________________________________________

# Note: Difference between trades and executions
#
#    The LOBSTER output records limit order executions
#    and not what one might intuitively consider trades.
#
#    Imagine a volume of 1000 is posted at the best ask
#    price. Further, an incoming market buy order of
#    volume 1000 is executed against the quote.
#
#    The LOBSTER output of this trade depends on the
#    composition of the volume at the best ask price.
#    Take the following two scenarios with the best ask
#       volume consisting of ...
#        (a) 1 sell limit order with volume 1000
#        (b) 5 sell limit orders with volume 200 each
#           (ordered according to time of submission)
#
#     The LOBSTER output for case ...
#       (a) shows one execution of volume 1000. If the
#           incoming market order is matched with one
#           standing limit order, execution and trade
#           coincide.
#       (b) shows 5 executions of volume 200 each with the
#           same time stamp. The incoming order is matched
#           with 5 standing limit orders and triggers 5
#           executions.
#
#       Bottom line:
#       LOBSTER records the exact limit orders against
#       which incoming market orders are executed. What
#       might be called 'economic' trade size has to be
#       inferred from the executions.
# Logic to calculate number of visible/hidden trades and their volume
for l in range(1, noint):
    visible_count.append(len(stream_df[(stream_df.Time > bound[l - 1]) & (
                stream_df.Time < bound[l]) & (stream_df.Type == 4)]))
    visible_size.append(sum(stream_df['Size'][(stream_df.Time > bound[l - 1]) & (
                stream_df.Time < bound[l]) & (stream_df.Type == 4)]) / 100)

    hidden_count.append(len(stream_df[(stream_df.Time > bound[l - 1]) & (
                stream_df.Time < bound[l]) & (stream_df.Type == 5)]))
    hidden_size.append(sum(stream_df['Size'][(stream_df.Time > bound[l - 1]) & (
                stream_df.Time < bound[l]) & (stream_df.Type == 5)]) / 100)
# First plot : Number of Execution by Interval (Visible + Hidden)
plt.title('Number of Executions by Interval for ' + ticker)
plt.fill_between(range(0, len(visible_count)),
                 visible_count,
                 color='#fc0417',
                 label='Visible')
plt.ylabel('Number of Executions per 5 min buckets')
plt.xlabel('# 5 min Interval')
plt.legend()
plt.fill_between(range(0, len(visible_count)),
                 [x * (-1) for x in hidden_count],
                 color='#0c04fc',
                 label='Hidden')
plt.legend()
plt.savefig('Number of Execution by 5min Interval (Visible + Hidden)' + '.png',dpi=300)
plt.close()

# Second plot : Trade Volume by Interval (Visible + Hidden)
plt.title('Trade Volume by Interval for ' + ticker)
plt.fill_between(range(0, len(visible_size)),
                 visible_size,
                 color='#fc0417',
                 label='Visible')
plt.ylabel('Volume of Trades(x100 shares)')
plt.xlabel('# 5 min Interval')
plt.legend()
plt.fill_between(range(0, len(visible_size)),
                 [x * (-1) for x in hidden_size],
                 color='#0c04fc',
                 label='Hidden')
plt.legend()
plt.savefig('Trade Volume by 5min Interval (Visible + Hidden)' + '.png',dpi=300)
plt.close()




timeIndex = stream_df.index[(stream_df.Time >= startTrad) & (stream_df.Time <= endTrad)]
lob_df = lob_df[lob_df.index == timeIndex]

for i in list(range(0, len(lob_df.columns), 2)):
    lob_df[lob_df.columns[i]] = lob_df[
                                                                lob_df.columns[i]] / 10000
# _____________________________________________________________________________
#
# Plot - Snapshot of the Limit Order Book
# _____________________________________________________________________________
# Note: Pick a random row/event from the order book
totalrows = len(lob_df)
random_no = np.random.choice(range(0, totalrows + 1), size=None, replace=False, p=None)
theAsk = lob_df[lob_df.columns[range(0, len(lob_df.columns), 4)]]
theAskVolume = lob_df[lob_df.columns[range(1, len(lob_df.columns), 4)]]

theAskValues = list(it.chain.from_iterable(theAsk[theAsk.index == random_no].values))
theAskVolumeValues = list(it.chain.from_iterable(theAskVolume[theAskVolume.index == random_no].values))
theDataAsk = pd.DataFrame({'Price': theAskValues, 'Volume': theAskVolumeValues})
theDataAsk = theDataAsk.sort_values(by=['Price'])
theBid = lob_df[lob_df.columns[range(2, len(lob_df.columns), 4)]]
theBidVolume = lob_df[lob_df.columns[range(3, len(lob_df.columns), 4)]]

theBidValues = list(it.chain.from_iterable(theBid[theBid.index == random_no].values))
theBidVolumeValues = list(it.chain.from_iterable(theBidVolume[theBidVolume.index == random_no].values))
theDataBid = pd.DataFrame({'Price': theBidValues, 'Volume': theBidVolumeValues})
theDataBid = theDataBid.sort_values(by=['Price'])
# Chart
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim(0, max(theDataBid['Volume'].max(), theDataAsk['Volume'].max()) + 200)
plt.xlim(min(theDataBid['Price'].min(), theDataAsk['Price'].min()),
         max(theDataBid['Price'].max(), theDataAsk['Price'].max()))
plt.suptitle('Limit Order Book Volume for ' + ticker + ' at random row' + str( random_no))
plt.ylabel('Volume')
plt.xlabel('Price($)')
ax.bar(theDataBid['Price'], theDataBid['Volume'], width=0.007, color='#13fc04', label='Bid')
ax.bar(theDataAsk['Price'], theDataAsk['Volume'], width=0.007, color='#fc1b04', label='Ask')

plt.legend()
plt.savefig('Limit Order Book Volume for ' + ticker + ' at random row' + '.png',dpi=300)
plt.close()

#_____________________________________________________________________________
#
# Plot - Relative Depth in the Limit Order Book
#_____________________________________________________________________________
# Plot variables
theAskVolume = lob_df[lob_df.columns[range(1,len(lob_df.columns),4)]]
totalSizeAsk = list(theAskVolume[theAskVolume.index == random_no].values.cumsum())
percentAsk = totalSizeAsk/totalSizeAsk[len(totalSizeAsk)-1]
theBidVolume = lob_df[lob_df.columns[range(3,len(lob_df.columns),4)]]
totalSizeBid = list(theBidVolume[theBidVolume.index == random_no].values.cumsum())
percentBid = totalSizeBid/totalSizeBid[len(totalSizeBid)-1]
# Chart
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim(-1,1)
plt.xlim(1,10)
plt.suptitle('Relative Depth in the Limit Order Book for ' + ticker + ' at ' + str(random_no))
plt.ylabel('% Volume')
plt.xlabel('Level')
ax.step(list(range(1,11)), percentBid, color='#13fc04', label='Bid')
ax.step(list(range(1,11)), -percentAsk, color='#fc1b04', label='Ask')
plt.legend()       
plt.savefig('Relative Depth in the Limit Order Book for ' + ticker + '.png',dpi=300)
plt.close()



#_____________________________________________________________________________
#
# Plot - Intraday Evolution of Depth
#_____________________________________________________________________________
# Calculate the max/ min volume to set limit of y-axis
maxAskVol = max(lob_df['AskSize1'].max()/100,lob_df['AskSize2'].max()/100,lob_df['AskSize3'].max()/100)  # calculate the maximum ask volume
# Calculate the max Bid volume , we use negative here and calculate min as we plot Bid below X-axis
maxBidVol = min(-lob_df['BidSize1'].max()/100,-lob_df['BidSize2'].max()/100,-lob_df['BidSize3'].max()/100)  # calculate the maximum ask volume
aa = range(int(stream_df['Time'].min()/(60*60)), int(stream_df['Time'].max()/(60*60))+2)
theTime = [int(i) for i in aa] 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylim(maxBidVol,maxAskVol)
plt.xlim(theTime[0],theTime[len(theTime)-1])
plt.suptitle('Intraday Evolution of Depth for ' + ticker + ' for 3 levels')
plt.ylabel('BID              No of Shares(x100)               ASK')
plt.xlabel('Time')
#plt.grid(True)
askSizeDepth3 = (lob_df['AskSize1']/100) + (lob_df['AskSize2']/100) + (lob_df['AskSize3']/100)
ax.plot((stream_df['Time']/(60*60)), 
        askSizeDepth3, 
        color='#fc1b04', 
        label='Ask 3')
askSizeDepth2 = (lob_df['AskSize1']/100) + (lob_df['AskSize2']/100)
ax.plot((stream_df['Time']/(60*60)), 
        askSizeDepth2, 
        color='#eeba0c', 
        label='Ask 2')
askSizeDepth1 = (lob_df['AskSize1']/100)
ax.plot((stream_df['Time']/(60*60)), 
        askSizeDepth1, 
        color='#3cee0c', 
        label='Ask 1')
bidSizeDepth3 = (lob_df['BidSize1']/100) + (lob_df['BidSize2']/100) + (lob_df['BidSize3']/100)
ax.plot((stream_df['Time']/(60*60)), 
        -bidSizeDepth3, 
        color='#0c24ee', 
        label='Bid 3')
bidSizeDepth2 = (lob_df['BidSize1']/100) + (lob_df['BidSize2']/100)
ax.plot((stream_df['Time']/(60*60)), 
        -bidSizeDepth2, 
        color='#e40cee', 
        label='Bid 2')
bidSizeDepth1 = (lob_df['BidSize1']/100)
ax.plot((stream_df['Time']/(60*60)), 
        -bidSizeDepth1, 
        color='#0ceee7', 
        label='Bid 1')
plt.legend()       
plt.savefig('Intraday Evolution of Depth for ' + ticker + ' for 3 levels' +  '.png',dpi=300)
plt.close()


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


