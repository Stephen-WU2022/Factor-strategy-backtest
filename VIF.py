from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt




factor1['Datetime']=pd.to_datetime(factor1['Datetime'])
factor1=factor1.set_index(factor1.Datetime).drop(columns={'Datetime'})
factor2['Datetime']=pd.to_datetime(factor2['Datetime'])
factor2=factor2.set_index(factor2.Datetime).drop(columns={'Datetime'})

factor1=factor1.resample(rule='W',closed='right',label='right').sum()
factor2=factor2.resample(rule='W',closed='right',label='right').sum()

idx=factor1.index.intersection(factor2.index)
col=factor1.columns.intersection(factor2.columns)
factor1=factor1.reindex(index=idx,columns=col)
factor2=factor2.reindex(index=idx,columns=col)


vif=[]
for i in range(len(idx)):
    factor_1 = np.array(factor1.iloc[i])
    factor_2 = np.array(factor2.iloc[i])
    mask = ~np.isnan(factor_1) & ~np.isnan(factor_2)
    ck = np.column_stack([factor_1[mask], factor_2[mask]])
    ck = add_constant(ck)
    vif.append([variance_inflation_factor(ck, j) for j in range(ck.shape[1])])
VIF=pd.DataFrame(data=vif,columns=['cons','factor1','factor2'],index=idx)


plt.plot(VIF.index,VIF['factor1'].values,color = 'lightblue')
plt.show()
plt.plot(VIF.index,VIF['factor2'].values,color = 'red')
plt.xticks(rotation='vertical')
plt.show()
















