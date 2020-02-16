import pandas as pd

num=[13,3,16,55,44,50,15,40]
ic_ir=pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\ic_ir.pickle')

t=0
for i in num:
    alphaName = 'alpha{:0>3d}'.format(i)
    alpha = pd.read_pickle(r'D:\Users\huyx02\PycharmProjects\untitled\101_Formulaic_Alphas_＃1_＃101\alpha_zz500\zz500_%s' % alphaName)
    if t==0:
        mergeData=alpha*ic_ir[alphaName]
        t=1
    else:
        mergeData=pd.merge(mergeData,alpha*ic_ir[alphaName],how='inner',left_index=True,right_index=True)

mergeData['bigAlpha']=mergeData.sum(axis=1)
mergeData['bigAlpha'].to_pickle('20_bigAlpha.pickle')


