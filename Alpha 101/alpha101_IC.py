import pandas as pd
import numpy as np
from scipy import stats
import os
import openpyxl

returns=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\wind_price_20030301.pickle')
returns=returns['s_dq_close'].to_frame()
returns=returns.unstack()
returns_20=returns.ffill().pct_change(10)
returns_20=returns_20.shift(-10)
returns_20=returns_20.stack()
returns_20.columns=['returns']


def func_ic(group):
    _ic=stats.spearmanr(group['returns'],group[alphaName])[0]
    return _ic



r'''
for i in range(1,102):
    alphaName='alpha{:0>3d}'.format(i)
    alpha=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\alpha_zz500\zz500_%s'%alphaName)
    if i==1:
        mergeData = pd.merge(returns_20, alpha, how='inner', right_index=True, left_index=True)
        mergeData=mergeData.replace([np.inf,-np.inf],0)
        mergeData.fillna(0,inplace=True)
        grouper = [mergeData.index.get_level_values('trade_dt')]
        ic=mergeData.groupby(grouper).apply(func_ic).to_frame()
        ic.columns=[alphaName]
    else:
        mergeData = pd.merge(mergeData, alpha, how='inner', right_index=True, left_index=True)
        mergeData=mergeData.replace([np.inf,-np.inf],0)
        mergeData.fillna(0,inplace=True)
        grouper = [mergeData.index.get_level_values('trade_dt')]
        ic[alphaName]=mergeData.groupby(grouper).apply(func_ic)
'''

alphaName = 'bigAlpha'
alpha = pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\20_bigAlpha.pickle')
mergeData = pd.merge(returns_20, alpha, how='inner', right_index=True, left_index=True)
mergeData = mergeData.replace([np.inf, -np.inf], 0)
mergeData.fillna(0, inplace=True)
grouper = [mergeData.index.get_level_values('trade_dt')]
ic = mergeData.groupby(grouper).apply(func_ic).to_frame()
ic.columns = [alphaName]




ic_cumsum=ic.cumsum(axis=0)
ic_mean=ic.mean(axis=0)
ic_std=ic.std(axis=0)
#alpha_corr=mergeData.loc[:,'alpha001':].corr()
ic_ir=ic_mean/ic_std

writer=pd.ExcelWriter(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\strategy\new_20bigAlpha_ic_corr.xlsx')
ic.to_excel(writer,sheet_name='ic')
ic_cumsum.to_excel(writer,sheet_name='ic_cumsum')
ic_mean.to_excel(writer,sheet_name='ic_mean')
ic_ir.to_excel(writer,sheet_name='ic_ir')
#alpha_corr.to_excel(writer,sheet_name='corr')
writer.save()

#ic_ir.to_pickle('10_ic_ir.pickle')