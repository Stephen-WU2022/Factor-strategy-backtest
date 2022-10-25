import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from single_factor_test import Single_factor_test

#pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)
#匯入數據
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
factor_exposure=factor_exposure.resample(rule='D',closed='right',label='right').sum()
log_return=log_return.resample(rule='D',closed='right',label='right').sum(min_count=1)
filter_1=filter_1.resample(rule='D',closed='right',label='right').last()#前一天交易量

#累積成長
factor_exposure=factor_exposure.rolling(window=7,min_periods=6,axis=0).mean()
filter_1=filter_1.rolling(window=7,min_periods=1,axis=0).mean() #過去7日之交易量

#對齊
factor_exposure=factor_exposure.dropna(how='all',axis=1).dropna(how='all',axis=0).loc[pd.to_datetime('2021-02-01').tz_localize('UTC'):,]
log_return=log_return.reindex(index=factor_exposure.index,columns=factor_exposure.columns)
filter_1=filter_1.reindex(columns=log_return.columns,index=log_return.index)
bitcoin = pd.DataFrame(log_return['bitcoin'])
index=pd.Series(factor_exposure.index)

#filter 根據情況過濾數據(7日平均需大於15m)
condition=filter_1.apply(lambda x: ( x >10000000),axis=1)
factor_exposure=factor_exposure.where(cond=condition.values,other=np.nan)

###單因子顯著性檢測 分層回測部位數量及p_value alpha
result=Single_factor_test(10,0.05)
'''
# p_value 回歸
p_value=[]
for i in range(1,len(log_return.index)):
    a,b=result.FL_generator(i,log_return=log_return,factor_exposure=factor_exposure)
    p_value.append(result.t_test((a),(b)))
#    p_value.append(result.t_test(np.exp(a),np.exp(b)))
p_value=pd.Series(data=p_value,index=index[1:])
print(result.p_value_absmean(p_value),result.p_value_abs2(p_value),p_value)
plt.bar(x=p_value.index,height=p_value.values,width=5, color = 'lightblue', label =p_value)
plt.xticks(rotation='vertical')
'''
#IC
IC=[]
for i in range(1,len(log_return.index)):
    factor_exposureT,log_returnT1=result.FL_generator(i,log_return=log_return,factor_exposure=factor_exposure)
    IC.append(result.ICrank((factor_exposureT),(log_returnT1)))
IC=pd.Series(data=IC,index=index[1:])
print(IC.values)
print(result.IR(IC))

plt.bar(x=IC.index,height=IC.values,width=5, color = 'lightblue', label =IC)
plt.xticks(rotation='vertical')

#分層回測
bucket_logreturn=[]
for i in range(1,len(log_return.index)):
    factor_exposureT,log_returnT1=result.FL_generator(i,log_return=log_return,factor_exposure=factor_exposure)
    bucket_logreturn.append(result.bucket_logreturnT1(exposureT=(factor_exposureT),logreturnT1=log_returnT1))
#benchmark
bucket_logreturn=pd.DataFrame(data=bucket_logreturn,index=index[1:]).cumsum()
benchmark=bitcoin['bitcoin'].cumsum()
bucket_logreturn=pd.merge(bucket_logreturn,benchmark,how='inner',left_index=True,right_index=True)
#plot
(bucket_logreturn).plot().legend(loc='lower left')
plt.xticks(rotation='vertical')
plt.show()
