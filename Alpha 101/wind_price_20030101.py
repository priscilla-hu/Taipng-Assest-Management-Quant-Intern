import cx_Oracle
import pandas as pd
dd=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\wind_price_indu_20030101.pickle')


conn = cx_Oracle.connect('windread','windread','WIND')

sql1 = 'select trade_dt,s_info_windcode,s_dq_open,s_dq_close,s_dq_high,\
 s_dq_low,s_dq_pctchange,s_dq_volume,s_dq_amount,S_dq_avgprice from wind.AShareEODPrices where trade_dt>20030301'
sql2 = 'select trade_dt,s_info_windcode,s_dq_mv,s_val_mv from wind.AShareEODDerivativeIndicator where trade_dt>20030301'

df1=pd.read_sql(sql1,conn)
df1.rename(columns=lambda x:x.lower(),inplace=True)
df1=df1.sort_values(by=['trade_dt','s_info_windcode'])
df1=df1.set_index(['trade_dt','s_info_windcode'])

df2=pd.read_sql(sql2,conn)
df2.rename(columns=lambda x:x.lower(),inplace=True)
df2=df2.sort_values(by=['trade_dt','s_info_windcode'])
df2=df2.set_index(['trade_dt','s_info_windcode'])
df_all=pd.merge(df1,df2,how='left',left_index=True,right_index=True)
#df_all['s_dq_mv','s_val_mv']=df_all['s_dq_mv','s_val_mv'].ffill()
df_all.to_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\wind_price_20030301.pickle')



exit()
# 1、示例，使用wind数据产生Alphas
# 定义文件路径，此处使用的是wind数据日行情表
FilePath = 'D:/wind_calc_data/data_000021.csv'
# 读入数据
stock_data_all = pd.read_csv(FilePath)
# 获取所有股票代码
stock_list = stock_data_all.S_INFO_WINDCODE.unique()

# 循环按股票代码循环计算Alphas，此处偷懒了不写了~~
#    for i in stock_list:
#        print(i)

# 此处使用000021股票，计算每只股票的82个Alphas值
data_000021 = stock_data_all[stock_data_all['S_INFO_WINDCODE'] == '000021.SZ']

# 我的数据已经按股票代码和交易日期排好序了，所以这里不需要排序
# 如果你的数据没有排序请使用一下语句排序
# data_000021=data_000021.sort_values(['S_INFO_WINDCODE','TRADE_DT'])
# 调用get_alpha函数
get_alpha(data_000021)
# 输出结果至文件
data_000021.to_csv('D:/wind_calc_data/stock_data_with_alpha_170816.csv')
