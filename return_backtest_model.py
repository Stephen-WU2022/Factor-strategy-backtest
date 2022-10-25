import pandas as pd
import numpy as np
import scipy
from pandas import DataFrame

class Return_peformance:

    def __init__(self, resample_frequency: int):
        self.frequency=resample_frequency

    def daily_average_retuen(self, cumulative_return: DataFrame):
        return (np.exp(cumulative_return.iloc[-1,])-1)/(self.frequency*len(cumulative_return.index))

    def monthly_average_return(self, cumulative_return: DataFrame):
        return self.daily_average_retuen(cumulative_return)*30

    def DrawDown(self, cumulative_return: DataFrame):
        return cumulative_return-cumulative_return.cummax()

    def MaxDrawDown(self, cumulative_return: DataFrame):
        idx = (1-np.exp(self.DrawDown(cumulative_return))).reset_index(drop=True).idxmax()
        high_value = cumulative_return.cummax().iloc[idx]
        mdd_rate = (1-np.exp(self.DrawDown(cumulative_return))).max()
        mdd = np.exp(high_value)*mdd_rate
        return mdd, mdd_rate, idx

    def Position_sort(self, exposureT: np.ndarray, returnT1: np.ndarray, column_names: np.ndarray):
        returnT1 = np.where(~np.isnan(exposureT), returnT1, np.nan)
        exposureT = np.where(~np.isnan(returnT1), exposureT, np.nan)
        column_names = np.where(~np.isnan(exposureT), column_names, np.nan)
        returnT1 = returnT1[~np.isnan(returnT1)]
        exposureT = exposureT[~np.isnan(exposureT)]
        column_names = column_names[~pd.isnull(column_names)]
        returnT1 = returnT1[np.argsort(exposureT[~np.isnan(exposureT)])]
        position_names = column_names[np.argsort(exposureT[~np.isnan(exposureT)])]
        # np.flipud ic為正值須將returnT1矩陣反轉，前為多頭部位，後為空頭部位
        return returnT1 , position_names














