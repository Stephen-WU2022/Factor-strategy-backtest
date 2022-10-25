import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame as df
from scipy import stats
from pandas import Series
from pre_treatment import Pretreatment




pd.set_option('display.max_rows',None)
#pd.set_option('display.max_columns',None)
#匯入數據
coins_prices=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/ftx_price_0101.txt", sep = ",", encoding = "utf-8", engine = "c")
factor_exposure=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/ftx_circulation_0101.txt", sep = ",", encoding = "utf-8", engine = "c")
#market_cap=pd.read_csv("C:/Users/Owner/factor_strategy/src/factor_price_data/ftx_marketcap_usd.txt", sep = ",", encoding = "utf-8", engine = "c")


#設置Datetime
coins_prices['Datetime']=pd.to_datetime(coins_prices['Datetime'])
coins_prices=coins_prices.set_index(coins_prices.Datetime).drop(columns={'Datetime'})
factor_exposure['Datetime']=pd.to_datetime(factor_exposure['Datetime'])
factor_exposure=factor_exposure.set_index(factor_exposure.Datetime).drop(columns={'Datetime'})
#market_cap['Datetime']=pd.to_datetime(market_cap['Datetime'])
#market_cap=market_cap.set_index(market_cap.Datetime).drop(columns={'Datetime'})


#resample
factor_exposure=factor_exposure.resample(rule='W',closed='right',label='right').sum().dropna(how='all')
#factor_exposure=factor_exposure.resample(rule='D',closed='left',label='left').first().dropna(how='all')
coins_prices=coins_prices.resample(rule='D',closed='left',label='left').last()
#market_cap=market_cap.resample(rule='D',closed='left',label='left').last()
#market_cap=market_cap.resample(rule='W',closed='right',label='right').last()


#基本數據 將數據取log計算成長率
pretreatment=Pretreatment()
log_return=pretreatment.caculate_logreturn(rawdata=coins_prices)
#factor_exposure=pretreatment.caculate_logreturn(rawdata=factor_exposure)
#market_cap=pretreatment.log(market_cap)

factor_exposure=factor_exposure.rolling(window=4,min_periods=1,axis=0).sum()

#去極值、標準化
#market_cap.replace([np.inf,-np.inf],np.nan,inplace=True)
factor_exposure.replace([np.inf,-np.inf],np.nan,inplace=True)
#factor_exposure=pretreatment.winsorize(rawdata=factor_exposure,limits=0.03)
factor_exposure=pretreatment.z_scoreing(factor_exposure)
factor_exposure=factor_exposure.dropna(how='all',axis=1).dropna(how='all',axis=0)#.iloc[:-5,:]

#適中類型因子值處理
#factor_exposure=factor_exposure.abs()



#對齊數據
log_return=log_return.reindex(columns=factor_exposure.columns)
#market_cap=market_cap.reindex(columns=factor_exposure.columns,index=factor_exposure.index)#.dropna(how='all',axis=1).dropna(how='all',axis=0)

print(log_return.dropna(axis=0,how='all'))

'''
y=factor_exposure.loc['2022-02-06',:]
sns.distplot(y,hist=False)
plt.show()

#市值中性化
factor_exposure_neutralized=[]
for i in range(0,len(factor_exposure.index)):
    factor_exposureT,market_capT=pretreatment.df_to_np(term=i,A=factor_exposure,B=market_cap)
    Residual=pretreatment.factor_neutralize(factor_exposure=factor_exposureT,market_cap=market_capT)
    factor_exposure_neutralized.append(Residual)

factor_exposure_neutralized=pd.DataFrame(data=factor_exposure_neutralized,index=factor_exposure.index,columns=factor_exposure.columns)
factor_exposure_neutralized.dropna(how='all',inplace=True)
print(factor_exposure_neutralized)
'''
log_return.dropna(axis=0,how='all').reset_index().to_csv("ftx_log_return0101.txt", index = None, sep = ",", encoding = "utf-8")
#factor_exposure.reset_index().to_csv("ftx_daily_active_addresses_neu_week.txt", index = None, sep = ",", encoding = "utf-8")




