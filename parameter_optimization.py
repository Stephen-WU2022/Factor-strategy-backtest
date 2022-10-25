import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame as df
from scipy import stats
from pandas import Series
from single_factor_test import Single_factor_test
from pre_treatment import Pretreatment
from return_backtest_model import Return_peformance

coins_prices=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/""after_pretreatment/"
                         "fx_ex_flow/ftx_log_return.txt", sep = ",", encoding = "utf-8", engine = "c")
factor_exposure=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/after_pretreatment/"
                         "fx_ex_flow/ftx_daily_exbalance_per_of_circulation1D.txt", sep = ",", encoding = "utf-8", engine = "c")
filter_1=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/"
                     "ftx_volume_daily.txt", sep = ",", encoding = "utf-8", engine = "c")

#設置Datetime
coins_prices['Datetime']=pd.to_datetime(coins_prices['Datetime'])
log_return=coins_prices.set_index(coins_prices.Datetime).drop(columns={'Datetime'})
factor_exposure['Datetime']=pd.to_datetime(factor_exposure['Datetime'])
factor_exposure=factor_exposure.set_index(factor_exposure.Datetime).drop(columns={'Datetime'})
filter_1['Datetime']=pd.to_datetime(filter_1['Datetime'])
filter_1=filter_1.set_index(filter_1.Datetime).drop(columns={'Datetime'})

#resample
factor_exposure_A = factor_exposure.resample(rule='D',closed='right',label='right').sum()
log_return_A = log_return.resample(rule='D',closed='right',label='right').sum(min_count=1)
filter_1_A = filter_1.resample(rule='D',closed='right',label='right').last()#前一天交易量


# 最佳化
IC_all = []
IR_all = []
Year_RRR = []

for i in range(1,61): #因子rolling天數最佳化
    factor_exposure_B=factor_exposure_A.rolling(window=i,min_periods=1,axis=0).mean()
    filter_1_B = filter_1_A.rolling(window=7, min_periods=1, axis=0).mean()  # 過去7日之交易量
    factor_exposure_B=factor_exposure_B.dropna(how='all',axis=1).dropna(how='all',axis=0).loc[pd.to_datetime('2021-02-01').tz_localize('UTC')\
                                                                                              :pd.to_datetime('2022-01-01').tz_localize('UTC'),]
    log_return_B = log_return_A.reindex(index=factor_exposure_B.index,columns=factor_exposure_B.columns)
    filter_1_B=filter_1_B.reindex(columns=log_return_B.columns,index=log_return_B.index)
    index=pd.Series(factor_exposure_B.index)

    condition=filter_1_B.apply(lambda x: ( x >10000000),axis=1)
    factor_exposure_B=factor_exposure_B.where(cond=condition.values,other=np.nan)


    #IC值
    result = Single_factor_test(10, 0.05)
    IC = []
    for j in range(1, len(log_return_B.index)):
        factor_exposureT, log_returnT1 = result.FL_generator(j, log_return=log_return_B, factor_exposure=factor_exposure_B)
        IC.append(result.ICrank((factor_exposureT), (log_returnT1)))
    IC = pd.Series(data=IC, index=index[1:])
    IC_mean,IC_std,IR = result.IR(IC)
    IC_all.append(IC_mean)
    IR_all.append(IR)

    #年化風報比
    result = Return_peformance(1)
    Np_factor = np.array(factor_exposure_B)
    Np_logreturn = np.array(log_return_B)
    column_names = np.array(factor_exposure.columns)
    position = []
    # 建立部位
    for j in range(1, len(factor_exposure_B.index)):
        factor_exposure_T = Np_factor[j - 1]
        log_returnT1 = Np_logreturn[j]
        returnT1, name = result.Position_sort(exposureT=factor_exposure_T, returnT1=log_returnT1,
                                              column_names=column_names)
        long_position_T = returnT1[0:10:1]
        short_position_T = np.flipud(returnT1)[0:10:1]
        total_position_T = (np.sum(long_position_T) - np.sum(short_position_T)) / 20
        position.append(total_position_T)

    position = pd.DataFrame(data=position, index=index[1:], columns=['returnT'])
    position['cumulative_return'] = position['returnT'].cumsum()

    cost = (0.00063 + 0.003) * 2
    position = np.log((np.exp(position) - cost))
    MDD, MDD_rate, day = result.MaxDrawDown(cumulative_return=position['cumulative_return'])
    y_rrr = (np.exp(position['cumulative_return'].iloc[-1,]) - 1)/MDD/len(position.index)*365
    Year_RRR.append(y_rrr)


Year_RRR = pd.Series(Year_RRR)
print(Year_RRR)

plt.plot(Year_RRR.index,Year_RRR.values,color = 'lightblue')
plt.xticks(rotation='vertical')
plt.show()

IC_all = pd.Series(IC_all)
IR_all = pd.Series(IR_all)
print(IC_all)
print(IR_all)
plt.plot(IC_all.index,IC_all.values,color = 'lightblue')
plt.xticks(rotation='vertical')
plt.show()
