import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame as df
from scipy import stats
from pandas import Series
from pandas import Timestamp
from single_factor_test import Single_factor_test
from pre_treatment import Pretreatment
from return_backtest_model import Return_peformance

#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)
#匯入數據
coins_prices=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/ftx_log_return0101.txt"
                         "", sep = ",", encoding = "utf-8", engine = "c")
factor_exposure=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/ftx_daily_exbalance_circulation1D0101.txt"
                            , sep = ",", encoding = "utf-8", engine = "c")
filter_1=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/ftx_volume_0101.txt"
                     , sep = ",", encoding = "utf-8", engine = "c")

#設置Datetime
coins_prices['Datetime']=pd.to_datetime(coins_prices['Datetime'])
log_return=coins_prices.set_index(coins_prices.Datetime).drop(columns={'Datetime'})
factor_exposure['Datetime']=pd.to_datetime(factor_exposure['Datetime'])
factor_exposure=factor_exposure.set_index(factor_exposure.Datetime).drop(columns={'Datetime'})
filter_1['Datetime']=pd.to_datetime(filter_1['Datetime'])
filter_1=filter_1.set_index(filter_1.Datetime).drop(columns={'Datetime'})

#resample
factor_exposure=factor_exposure.resample(rule='D',closed='right',label='right').sum()
log_return=log_return.resample(rule='D',closed='right',label='right').sum(min_count=1)
filter_1=filter_1.resample(rule='D',closed='right',label='right').last()
volume = pd.DataFrame()
volume['volume'] = filter_1['bitcoin']


#累積成長
factor_exposure=factor_exposure.rolling(window=7,min_periods=7,axis=0).mean()
filter_1=filter_1.rolling(window=7,min_periods=1,axis=0).mean() #過去7日之交易量


#對齊
factor_exposure=factor_exposure.dropna(how='all',axis=1).dropna(how='all',axis=0)
log_return=log_return.reindex(index = factor_exposure.index).dropna(how='all',axis=1).dropna(how='all',axis=0)
factor_exposure = factor_exposure.reindex(columns=log_return.columns)
filter_1=filter_1.reindex(columns=log_return.columns,index=log_return.index)
index=pd.Series(factor_exposure.index)



#filter 根據幣交易量過濾因子值
condition=filter_1.apply(lambda x: ( x >15000000),axis=1)#(500000000> x) &
factor_exposure=factor_exposure.where(cond=condition.values,other=np.nan)

#每期poll數量
poll=factor_exposure.count(axis=1)
poll=np.floor(poll/2).shift(1).dropna()
poll.loc[poll>=10]=10#下一期部位

#ic濾網
result=Single_factor_test(10,0.05)
IC=[]
for i in range(1,len(log_return.index)):
    factor_exposureT,log_returnT1=result.FL_generator(i,log_return=log_return,factor_exposure=factor_exposure)
    IC.append(result.ICrank((factor_exposureT),(log_returnT1)))
IC=pd.Series(data=IC,index=index[1:])
IC = IC.shift(1).rolling(window=2).mean().dropna()

cta_cond = IC.apply(lambda x:  x > 0.22 )
cta_cond = cta_cond.reindex(index=poll.index).bfill()
poll = poll.mask(cond= cta_cond.values, other=2)

#volume濾網
volume['volume_mean'] = volume['volume'].rolling(window=30,min_periods=1,axis=0).mean()
volume['diff'] = (volume['volume'].rolling(window=7,min_periods=1,axis=0).mean())-volume['volume_mean']
volume = volume.shift(1).reindex(index=log_return.index)

cta_cond = volume['diff'].apply(lambda x:  x < -1*10**9 )
cta_cond = cta_cond.reindex(index=poll.index).bfill()
poll = poll.mask(cond= cta_cond.values, other=2)



###收益率回測
result=Return_peformance(1/288)

Np_factor=np.array(factor_exposure)
Np_logreturn=np.array(log_return)
column_names=np.array(factor_exposure.columns)

long_position_names = []
short_position_names = []
long_position = []
short_position = []
position = pd.DataFrame()
#建立部位
for i in range(1, len(factor_exposure.index)):
    factor_exposure_T = Np_factor[i - 1]
    log_returnT1 = Np_logreturn[i]
    returnT1, name = result.Position_sort(exposureT=factor_exposure_T, returnT1=log_returnT1, column_names=column_names)
    long_position.append(returnT1[0:10:1])
    short_position.append(np.flipud(returnT1)[0:10:1])
    long_position_names.append((name[0:10:1]))
    short_position_names.append((np.flipud(name)[0:10:1]))

long_position = pd.DataFrame(data=long_position,index=index[1:])
short_position = pd.DataFrame(data=short_position,index=index[1:])

long_position_names = pd.DataFrame(data=long_position_names,index=index[1:])
short_position_names = pd.DataFrame(data=short_position_names,index=index[1:])



long_position_names.reset_index().to_csv("long_name0101.txt", index = None, sep = ",", encoding = "utf-8")
short_position_names.reset_index().to_csv("short_name0101.txt", index = None, sep = ",", encoding = "utf-8")


#總收益，5分k資料
long_position = pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/"
                            "long_daily0101.txt", sep = ",", encoding = "utf-8", engine = "c")
short_position = pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/"
                             "short_daily0101.txt", sep = ",", encoding = "utf-8", engine = "c")
long_position['Datetime'] = pd.to_datetime(long_position['Datetime'])
long_position = long_position.set_index(long_position.Datetime).drop(columns={'Datetime'})#.tz_localize('UTC')
short_position['Datetime'] = pd.to_datetime(short_position['Datetime'])
short_position = short_position.set_index(short_position.Datetime).drop(columns={'Datetime'})#.tz_localize('UTC')

long_position['poll'] = poll.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = long_position.index).shift(3).ffill().bfill()
short_position['poll'] = poll.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = long_position.index).shift(3).ffill().bfill()
long_position['L_total'] = long_position.apply(lambda x:(np.nansum(x[0:int(x[-1]):1])),raw=True,axis=1)/10
short_position['S_total'] = short_position.apply(lambda x:(np.nansum(x[0:int(x[-1]):1])),raw=True,axis=1)/10
position['returnT'] = (long_position['L_total'] - short_position['S_total'])/2

#stop loss

stop_loss = position.cumsum().shift(-3).resample(rule='D',closed='left',label='left')\
                                                                    .first().rename(columns={'returnT':'limit'}).dropna()
stop_loss['limit'] = stop_loss['limit']+np.log(0.95)
stop_loss = stop_loss.resample(rule='5Min',closed='right',label='right').first().ffill()\
                                                            .reindex(index = long_position.index).shift(3).ffill().bfill()
signal = pd.DataFrame(data=(stop_loss['limit'] > position['returnT'].cumsum()),columns=['signal'])
signal = signal.shift(-3)
stop_signal = signal.loc[signal['signal'] == 1].index
for i in stop_signal:
    to_index = (pd.Timestamp.timestamp(pd.to_datetime(Timestamp.date(i)))+86400-300)
    signal.loc[i:Timestamp(to_index,unit='s').tz_localize('UTC'),] = True
signal = signal.shift(3).bfill()
position['returnT'] = position['returnT'].mask(cond= signal['signal'].values, other=0)
signal = signal.resample(rule='D',closed='right',label='left').last()*1


#cost 計算更動倉位次數
cost_count = [10]

long_position_names = pd.DataFrame(data=long_position_names, index=index[1:])
short_position_names = pd.DataFrame(data=short_position_names, index=index[1:])
long_position_names = np.array(long_position_names)
short_position_names = np.array(short_position_names)

for i in range(1, len(index[1:])):
    long_nameT_1 = long_position_names[i - 1][0:int(poll.iloc[i-1,]):1]
    long_nameT = long_position_names[i][0:int(poll.iloc[i,]):1]
    long_count = len(np.unique(np.concatenate([long_nameT_1, long_nameT]))) - ((len(long_nameT_1)+len(long_nameT))/2) #有多少倉位有變動，手續費為此值*2
    short_nameT_1 = short_position_names[i - 1][0:int(poll.iloc[i - 1,]):1]
    short_nameT = short_position_names[i][0:int(poll.iloc[i,]):1]
    short_count = len(np.unique(np.concatenate([short_nameT_1, short_nameT]))) - ((len(short_nameT_1) + len(short_nameT)) / 2)
    total_count = long_count + short_count
    cost_count.append(total_count)
cost = pd.DataFrame(data=cost_count,index=index[1:])
cost[0] = cost[0] +(signal['signal']*20)
cost = cost*(0.00063+0.0015)*2/20
cost = (cost.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = long_position.index).shift(3).ffill().bfill())/288



"""
#資金費率成本
long_future_fund = pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/"
                               "long_fund419.txt", sep = ",", encoding = "utf-8", engine = "c")
short_future_fund = pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/future_test/"
                                "short_fund419.txt", sep = ",", encoding = "utf-8", engine = "c")
long_future_fund['Datetime'] = pd.to_datetime(long_future_fund['Datetime'])
long_future_fund = long_future_fund.set_index(long_future_fund.Datetime).drop(columns={'Datetime'}).tz_localize('UTC')
long_future_fund = long_future_fund.resample(rule='D',closed='right',label='left').sum()
short_future_fund['Datetime'] = pd.to_datetime(short_future_fund['Datetime'])
short_future_fund = short_future_fund.set_index(short_future_fund.Datetime).drop(columns={'Datetime'}).tz_localize('UTC')
short_future_fund = short_future_fund.resample(rule='D',closed='right',label='left').sum()


long_future_fund['poll'] = poll
short_future_fund['poll'] = poll
long_future_fund['total'] = long_future_fund.apply(lambda x:(np.nansum(x[0:int(x[-1]):1])),raw=True,axis=1)/10
short_future_fund['total'] = short_future_fund.apply(lambda x:(np.nansum(x[0:int(x[-1]):1])),raw=True,axis=1)/10
"""

#fund_cost = -long_future_fund['total']+short_future_fund['total']
#fund_cost = (fund_cost.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = long_position.index).shift(3).ffill().bfill())/288
position['returnT'] = np.log(np.exp(position['returnT'])-cost[0])#+fund_cost )


#position = position.shift(-3).resample(rule='D',closed='right',label='left').sum(min_count=1).dropna()

#累積報酬
position['cumulative_return'] = position['returnT'].cumsum()

#benchmark
#bitcoin = pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/"
#                               "bitcoin_daily_return.txt", sep = ",", encoding = "utf-8", engine = "c")
#bitcoin['Datetime'] = pd.to_datetime(bitcoin['Datetime'])
#bitcoin = bitcoin.set_index(bitcoin.Datetime).drop(columns={'Datetime'}).tz_localize('UTC').cumsum()
#position=pd.merge(position,bitcoin,how='inner',left_index=True,right_index=True)

#績效指標
monthly_average_return = result.monthly_average_return(position['cumulative_return'])
daily=result.daily_average_retuen(position['cumulative_return'])
MDD, MDD_rate, date=result.MaxDrawDown(cumulative_return=position['cumulative_return'])
Drawdown = result.DrawDown(position['cumulative_return'])


print(daily, monthly_average_return, daily*360, np.exp(position['cumulative_return'].iloc[-1,])-1, MDD_rate,
      (np.exp(position['cumulative_return'].iloc[-1,])-1)/MDD,
      (np.exp(position['cumulative_return'].iloc[-1,])-1)/MDD/len(index)*365, Drawdown.min(),
      date)



volume = volume.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = position.index).shift(3).ffill().bfill()
IC = IC.resample(rule='5Min',closed='right',label='right').ffill().reindex(index = long_position.index).shift(3).ffill().bfill()


#plot

plt.figure(figsize=(40, 15))
Drawdown.plot().legend(loc='lower left')
position['cumulative_return'].plot().legend(loc='lower left')
log_return['bitcoin'].cumsum().plot().legend(loc='lower left')


'''
Drawdown.loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]\
    .plot().legend(loc='lower left')
position['bitcoin'].loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]\
    .plot().legend(loc='lower left')
#IC.loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]\
#    .plot().legend(loc='lower left')
position['cumulative_return'].loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]\
    .plot().legend(loc='lower left')
plt.axhline(-0.0833816,color='green', lw=2, alpha=0.7)
plt.axhline(-0,color='lightgray', lw=2, alpha=0.7)
plt.fill_between(Drawdown.index, -0.15, 0.75, where=Drawdown.values < -0.0833816,
                color='lightskyblue', alpha=0.5)
#plt.fill_between(IC.index, -0.25, 0.64, where=IC.values > 0.22,
#                color='lime', alpha=0.5)

plt.twinx()
volume['diff'].loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]\
    .plot(color = 'blue').legend(loc='lower left')
plt.fill_between(volume.index, -1.5*10**9, 2.5*10**9, where=volume['diff'].values < -1*10**9,
                color='violet', alpha=0.5)

(long_position['L_total'].loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]
 .cumsum()/2).plot().legend(loc='lower left')
(short_position['S_total'].loc[pd.to_datetime('2021-02-02').tz_localize('UTC'):pd.to_datetime('2022-04-20').tz_localize('UTC'),]
 .cumsum()/2*-1).plot().legend(loc='lower left')
'''

plt.xticks(rotation='vertical')
plt.grid()
plt.show()

