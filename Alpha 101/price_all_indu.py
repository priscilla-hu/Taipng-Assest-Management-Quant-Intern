import pandas as pd

data_all=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\indu_20030101.pickle')
data_indu=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\wind_price_20030301.pickle')
data_indu.reset_index(inplace=True)
data_indu['trade_dt']=data_indu['trade_dt'].astype('datetime64')
#data_indu.set_index(['trade_dt','s_info_windcode'],inplace=True)

data_indu=pd.merge(data_indu,data_all,left_on=['trade_dt','s_info_windcode'],right_on=['trade_dt','s_info_windcode'],how='left')
data_indu.trade_dt = data_indu.trade_dt.apply(lambda x:x.strftime('%Y%m%d'))
data_indu.set_index(['trade_dt','s_info_windcode'],inplace=True)
data_indu.to_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\pickle_data\wind_price_indu_20030101.pickle')
pass