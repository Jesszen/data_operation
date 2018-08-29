import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  #交叉检验库
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM

raw_data=pd.read_table('G:\data_operation\python_book\chapter6\products_sales.txt',delimiter=',')#read table  需要查看分隔符

def set_summary(df):
    """
    查看数据基本状态
    :param df:
    :return:
    """
    print('{:*^80}'.format('查看前五行'))
    print(df.head(5))
    print('{:*^80}'.format('查看后五行'))
    print(df.tail(5))
    print('{:-^80}'.format('data dtypes'))
    print(df.dtypes)
    print('{:-^80}'.format('data describle'))
    print(df.describe().round(1).T)#转置后查看，当字段不是太多时，方便查看

set_summary(raw_data)
#发现limit infor描述变量中，二分变量的最大值10，有问题
#又发现campaign_fee标准差也很大，最大的促销费用居然到了3w多，明显时异常值
#根据count计数，发现有一列计数项729，明显存在缺失值

#查看值域分布，目的为查看分类变量的值域，因为在数据describe中发现异常

def classified_range(df,fenlei):
    """
    查看分类变量的值域
    :param df:
    :return:
    """
    c_range=[]
    for i in fenlei:
        k=np.sort(df[i].unique()) #sort paixu
        print('{:-^80}'.format('{0} unique values{1}').format(i,k))  #format 可以嵌套
        c_range.append(k.tolist())
    return c_range
col_names= ['limit_infor', 'campaign_type', 'campaign_level', 'product_level']  # 定义要查看的列

c_r=classified_range(raw_data,col_names)
print(c_r)
#由此看出limit infor值域确实存在问题

def na_summary(df):
    """
    cha kan NAN
    :param df:
    :return:
    """
    print('{:*^80}'.format('cols has nan'))
    print(df.isnull().any(axis=0))
    print('{:*^80}'.format('rows has nan'))
    print(df.isnull().any(axis=1).sum())#sum 才是求和，而不是计数
    print('{:*^80}'.format('nan '))
    print(df[df.isnull().values == True].T)
    print('{:*^80}'.format('nan numbers'))
    print(df.isnull().sum()) #显示每一列的缺失值个数
na_summary(raw_data)

#相关性分析
short_name=['li', 'ct', 'cl', 'pl', 'ra', 'er', 'price', 'dr', 'hr', 'cf', 'orders']
long_name=raw_data.columns
name_dict=dict(zip(long_name,short_name))
corr=raw_data.corr().round(2).rename(index=name_dict,columns=name_dict)#可以字典形式，快速rename
print('{:*^80}'.format('corr matrix'))
print(type(corr))
print(corr)
# u=corr >0.6  #利用魔法，比较每个相关系数是否大于0.6
# print(u)
print(name_dict)

def vif(df):
    """
    方差膨胀因子,查看某一列对其他所有列做回归，相当于整体性指标，相对于相关系数矩阵
     TypeError: unhashable type: 'slice',因为variance_inflation_factor值接受矩阵格式，而非dataframe
    :param df:
    :return:
    """
    #ss=StandardScaler()
    vif_list=[]
    df=df.fillna(value=df['price'].mean()) #左边一定要等于df，否则没有保存
    # t=df['campaign_fee'].max()
    # print(t)
    # df=df[df['campaign_fee']!=t]
    print(df.isnull().any().sum())
    #data2=ss.fit_transform(df)
    for i in range(0,df.shape[1]):
        vif_tmp=variance_inflation_factor(df.values,i)
        vif_list.append(vif_tmp)
    return vif_list
# vif_values=vif(raw_data)
# print(dict(zip(long_name,vif_values)))


#预处理
#缺失值处理
print(raw_data.shape)
print(raw_data.isnull().any())
sales_data=raw_data.fillna(raw_data['price'].mean())
print(sales_data.shape)


#只保留促销值时1和0
sales_data=sales_data[sales_data['limit_infor'].isin((1,0))] #isin((1,0)) 结果是一个布尔值数组，在是True，不再是False,因为重新赋值，所以相当于丢掉一行，即infor=10的

#sales_data=sales_data.drop(sales_data[sales_data['limit_infor']==10].index[0])
print(sales_data.shape)

plt.figure()
plt.plot(np.arange(sales_data.shape[0]),sales_data['campaign_fee'])
plt.show()

outliter_score=(sales_data['campaign_fee'] - sales_data['campaign_fee'].mean())/sales_data['campaign_fee'].std()
out_data=sales_data['campaign_fee'][outliter_score >2.2]
print(type(out_data))
print(out_data.shape)
print(out_data)
print(out_data.index)
u=out_data.index[0]
print(u)
print(sales_data['campaign_fee'][633])



#针对极大值，利用均值替代
max_fee=sales_data['campaign_fee'].max()
print(type(max_fee))
sales_data['campaign_fee'] = sales_data['campaign_fee'].replace(max_fee, sales_data['campaign_fee'].mean())  #老眼昏花，replace
print('{:*^80}'.format('transformed data:'))
print(sales_data.describe().round(2).T.rename(index=name_dict))



#xy
X=sales_data.iloc[:,:-1]
y=sales_data.iloc[:,-1]

model_gbr=GradientBoostingRegressor()
parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
              'min_samples_leaf': [1, 2, 3, 4, 5],
              'alpha': [0.1, 0.3, 0.6, 0.9]}  # 定义要优化的参数信息
#网格搜索
model_gs=GridSearchCV(estimator=model_gbr,param_grid=parameters,cv=5)
model_gs.fit(X,y)

print('best score is : %f'%model_gs.best_score_)
print('best parameters is ',model_gs.best_params_)

model_best=model_gs.best_estimator_  #提取经过交叉验证的最优模型

model_best.fit(X,y)
print(model_best.score(X,y))
y_pred=model_best.predict(X)
# model_best=GradientBoostingRegressor(alpha= 0.9, loss='huber', min_samples_leaf= 3)


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(X.shape[0]),y,label='true y')
plt.plot(np.arange(X.shape[0]),y_pred,label='pred y')
plt.legend(loc=0)
plt.show()
