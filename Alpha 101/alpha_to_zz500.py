import pandas as pd
zz_500_flag=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\交易\zz500_stflag.pkl')
zz_500_flag.drop(['stflag'], axis=1, inplace=True)


for num in range(100,101):
    alpha_name='alpha{:0>3d}'.format(num)
    data=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\alpha_pickle\%s'%alpha_name)
    zz_500_flag[alpha_name] = data
    zz_500_flag.sort_index(inplace=True)
    zz_500_flag.to_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\alpha_zz500\zz500_alpha{:0>3d}'.format(num))
    zz_500_flag.drop([alpha_name], axis=1, inplace=True)