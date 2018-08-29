"""
离散化
定义：无限空间映射到有限空间，大多数针对连续型数据，处理后的数据值域分布从连续性，转为离散型
必要性：
  1、分类模型算法需要
  2、图像处理方面需要

不同类型的离散化
1、时间数据的离散化
   a、针对一天中的时间离散化：时间戳转化成分，小时，上下午等；
   b、日粒度以上的离散化：一般把日期转化陈周数，周几，月，季度等

2、多值离散数据的离散化
   a、原始数据已经离散类型的数据，例子用户划分123类，新的业务逻辑要细分成123456类

3、针对连续性数据离散化【主要应用领域】
   分类或者关联中应用广泛
   离散化结果分为两类
   a、连续性数据转化成特定区间的集合
   b、转化成特定类
   方法：
   a、分位数法：eg 四分位【0.25，0.5，0.75，1】化成4类
   b、聚类法：k均值据类转化成几个簇
   c、距离区间法：等距离或者自定义距离离散
   d、频率区间法：不同数据的频率分布进行排序
4、连续性数据二值法
   变量的特征进行二值法操作：每个数据和阈值比较，高1，低0，然后得到只用于两个值域的数据集
   注意：二值化处理的数据集，所有属性值代表的含义相同或者相近。
         当然也可以单独针对某列进行二值化操作
   eg：图像的处理

"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import math
df=pd.read_table('G:\data_operation\python_book\chapter3\data7.txt',names=['id','amount','income','datetime','age'])
# print(df.head())
# print(df.iloc[:,-1].unique())#查看所有独一无二的值
df.drop_duplicates()

#针对时间数据离散化

dfc=df.copy()
j=dfc.iloc[:,0].size #获取数据框的行数
z=np.zeros(shape=(j,1),dtype=int)#设置和数据库行数相同的np数组，并设置整数型int

for i ,date in enumerate(dfc.iloc[:,3]):
    single_date_tmp=pd.to_datetime(date)#格式转化成datetime格式
    z[i]=single_date_tmp.weekday()  #for循环里 引用dfc.iloc[:,3][i]=single_date_tmp.weekday()  ，则会报错

u=pd.DataFrame(z,columns=['weekday'])
dfc=pd.concat((dfc,u),axis=1)#按照行合并数据

print(dfc.head())

#针对多值数据的离散化

#新建一个数据框，前后离散化的区间一一对应，并保持一个列名和数据库中需要再离散化的列名相同
dft=pd.DataFrame([['0-10','0-40'],['10-20','0-40'],['20-30','0-40'],['30-40','0-40'],
                  ['40-50','40-80'],['50-60','40-80'],['60-70','40-80'],['70-80','40-80'],
                  ['80-90','>80'],['>90','>80']],
                 columns=['age','age2'])
df_tmp=df.merge(dft,how='inner')#以相同的列名合并数据集

#pd.merge(df1,df2,left_on="name1",right_on='name2') 当左右连接字段不相同时，使用left_on,right_on
print(df_tmp.head())
df_duo=df_tmp.drop('age',axis=1)
print(df_duo.head())

#针对连续性数据的离散化
#方法1：自定义分箱区间  即将原连续数据转化成一个区间
bins=[0,200,1000,5000,10000]
df['amount1']=pd.cut(df['amount'],bins)#还有labels属性，设置标签
print(df.head())

#方法3：适用四分类法

df['amount3']=pd.qcut(df['amount'],4,labels=['bad','medium',
                                           'good','awesome'])
print(df.head())


#方法2：适用聚类
k=KMeans(n_clusters=4,random_state=0)#手工指定簇
data=df['amount']#没有列值，所有我们给整形,print的结果(100,)
data1=data.values.reshape(data.shape[0],1)#整形指定一列，print结果(100, 1)

k_re=k.fit_transform(data1)#fit_transform  结果是每个样本距离簇的距离

k_result=k.fit_predict(data1)#每个样本的分类
df['amout2']=k_result
print(df.head())


#连续性二值法

binarizer_scaler=preprocessing.Binarizer(threshold=df['income'].mean())#建立二值化模型，以income的均值做区分
data2=df['income'].values.reshape(df['income'].shape[0],1)  #重新整形 以列的形状
imcome_tmp=binarizer_scaler.fit_transform(data2)

df['income2']=imcome_tmp

print(imcome_tmp.shape)
print(df.head())