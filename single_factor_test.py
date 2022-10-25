import scipy
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy import stats
from pandas import Series
from numpy import ma as ma
from pre_treatment import Pretreatment
from return_backtest_model import Return_peformance

class Single_factor_test(Pretreatment):
    def __init__(self, bucket_amount: int, alpha: float):
        self.bucket_amount=bucket_amount
        self.alpha=alpha


#產生當期因子暴露及收益矩陣
    def FL_generator(self,term: int, factor_exposure: DataFrame, log_return: DataFrame):
        factor_exposureT=np.array(factor_exposure.iloc[term-1])
        log_returnT1 = np.array(log_return.iloc[term])
        return factor_exposureT , log_returnT1
#  回歸、T_value
    def regression(self,factor_exposure_T: np.ndarray, return_T1: np.ndarray):

        mask = ~np.isnan(factor_exposure_T) & ~np.isnan(return_T1)
        return stats.linregress(x=factor_exposure_T[mask],y=return_T1[mask],alternative='two-sided')
    def t_test(self,factor_exposure_T: np.ndarray,return_T1: np.ndarray):
        p_value=self.regression(factor_exposure_T,return_T1).pvalue
        return p_value
    def p_value_absmean(self,p_value_series: Series):
        return Series.mean(Series.abs(p_value_series))
    def p_value_abs2(self,p_value_series: Series):
        p_value_series=np.array(Series.abs(p_value_series))
        return  np.sum(p_value_series<=self.alpha)/np.size(p_value_series)
#ICIR Rank
    def ICrank(self,exposureT: np.ndarray, returnT1: np.ndarray):
        mask = ~np.isnan(exposureT) & ~np.isnan(returnT1)
        IC=stats.spearmanr(a=exposureT[mask],b=returnT1[mask])
        return IC.correlation
    def IR(self,IC_Series: Series):
        IC_mean=IC_Series.mean(axis=0)
        IC_std=IC_Series.std(axis=0)
        IR=IC_mean/IC_std
        return IC_mean,IC_std,IR
#分層回測
    #針對本期因子做排序
    def sortbyfactorT(self,exposureT: np.ndarray,returnT1: np.ndarray) -> np.ndarray:
        returnT1 = np.where(~np.isnan(exposureT), returnT1, np.nan)
        exposureT = np.where(~np.isnan(returnT1), exposureT, np.nan)
        returnT1=returnT1[~np.isnan(returnT1)]
        exposureT=exposureT[~np.isnan(exposureT)]
        returnT1=returnT1[np.argsort(exposureT[~np.isnan(exposureT)])]
        return (returnT1)
    #np.flipud ic為正值須將矩陣反轉
    #本期分層部位對數收益
    def bucket_logreturnT1(self,exposureT: np.ndarray,logreturnT1: np.ndarray) -> np.array:
        bucket_logreturnT1 =np.array_split(self.sortbyfactorT(exposureT, logreturnT1), self.bucket_amount)
        k = np.zeros([len(bucket_logreturnT1), len(max(bucket_logreturnT1, key=lambda x: len(x)))])
        for i, j in enumerate(bucket_logreturnT1):
            k[i][0:len(j)] = j
        return np.average(k,axis=1,weights=(k != 0))




















