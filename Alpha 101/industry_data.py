import cx_Oracle
import pandas as pd
import numpy as np

conn = cx_Oracle.connect('windread','windread','WIND')

# 个股行业变动表
sqlquery = 'select S_INFO_WINDCODE "s_info_windcode", CITICS_IND_CODE "inducode", ENTRY_DT "trade_dt" from wind.AShareIndustriesClassCITICS' \
           ' where ENTRY_DT<=20190720 order by ENTRY_DT,S_INFO_WINDCODE'
data = pd.read_sql(sqlquery, conn)

# 各级行业代码
sqlquery = 'select IndustriesCode "inducode",Industriesname "induname",levelnum "levelnum" from wind.AShareIndustriesCode where levelnum>1 and levelnum<5 order by IndustriesCode'
induname = pd.read_sql(sqlquery, conn)

conn.close()


def indudata_changename_sub(colname, lnum):
    # 将行业代码转换成行业名称 子函数
    induname1=induname[induname['levelnum']==lnum].copy()
    induname1['inducode1']=induname1['inducode'].apply(lambda x:x[0:lnum*2])
    induname1.set_index('inducode1',inplace=True)
    dic=induname1['induname'].to_dict()
    data[colname]=data['inducode'].apply(lambda x:dic[x[0:lnum*2]])

indudata_changename_sub('industry',2)
indudata_changename_sub('subindustry',3)
#indudata_changename_sub('induname3',4)

data = data.sort_values(by=['trade_dt','s_info_windcode'])
data['trade_dt']=data['trade_dt'].astype('datetime64')
#data.set_index(['trade_dt','s_info_windcode'],inplace=True)

# 先把行业分类填充到每个trade_dt
codes=np.unique(data.s_info_windcode)
data_all = pd.DataFrame(np.array([list(codes)]*len(pd.date_range('20030101','20190730',freq='1D'))).T,columns=pd.date_range('20030101','20190730',freq='1D'))
data_all=data_all.stack().to_frame()
data_all.reset_index(inplace=True)
data_all.drop(columns=['level_0'],inplace=True)
data_all.rename(columns={'level_1':'trade_dt',0:'s_info_windcode'},inplace=True)
data_all.sort_values(by=['trade_dt','s_info_windcode'],inplace=True)
#data_all.set_index(['trade_dt','s_info_windcode'],inplace=True)

data_all = pd.merge(data_all,data,left_on=['trade_dt','s_info_windcode'],right_on=['trade_dt','s_info_windcode'],how='outer')
group=['industry','subindustry']
grouped=data_all[group].groupby(data_all['s_info_windcode'])
data_all[group] = grouped.ffill()[group]
#data_all.set_index(['trade_dt','s_info_windcode'],inplace=True)
data_all.to_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\indu_20030101.pickle')

