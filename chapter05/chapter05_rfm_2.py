import pandas as pd
import numpy as np
import datetime
import time
import MySQLdb
import sqlalchemy
import os
import _mysql_connector



mysql_name='root'
mysql_password='123456789'
mysql_host='localhost'
mysql_database='sales'
table_name='yunying'


#从mysql数据库中读取数据
#engine=sqlalchemy.create_engine('mysql + mysqldb://root:123456@118.24.26.227:3306/python_yuny')
engine=sqlalchemy.create_engine('mysql+mysqlconnector://{user}:{password}@{host}:3306/{database}'.format
                                (user=mysql_name,password=mysql_password,host=mysql_host,database=mysql_database))

#data=pd.read_sql('select * from %s'%table_name,con=engine,index_col='USERID')
# print(df.dtypes)
# df.iloc[:,1]=pd.to_datetime(df.iloc[:,1])
# print(df.head(5))
# print(len(df[df.isnull().values==True]))
# print(len(df))


#从本地文件读取csv
#一般不要设置每列数据类型
#dtyes={'ORDERDATE':object,'ORDERID':object,'AMOUNTINFO':np.float32}
data=pd.read_csv('G:\data_operation\python_book\chapter5\\sales.csv',index_col='USERID')#设置索引，

print('{:*^80}'.format('data overview'))
#查看不同列的数据格式，看是否需要转换，或者转换的是否正确
print(data.head(5))

#查看统计上数据，发现amountinfo，订单金额最小值0.5元，我们甚至可以删除小于1块钱的订单，因为这些订单，都是促销券的订单，无意义可以删除
print(data.describe())


print('{:-^80}'.format('nan'))
#缺失值检测

#查看那一列有缺失值
col_nan=data.isnull().any(axis=0)
print('列含有缺失值')
print(col_nan)

#查看哪一行有缺失值
#因为布尔值，所以可以直接计算，r_nan.sum()
r_nan=data.isnull().any(axis=1)


#整个缺失值矩阵
nan=data.isnull()

#只打印缺失值的行

nan_data=data[data.isnull().values==True]
print(len(nan_data))
print(nan_data)
#等价
print('{:-^80}'.format('等价'))
print(r_nan.sum())
print(data[r_nan])

#删除缺失值，筛选订单金额大于1的订单

sales_data=data.dropna()
sales_data=sales_data[sales_data['AMOUNTINFO'] >1]

#日期格式转换,计算模型RFM中的r

sales_data['ORDERDATE']=pd.to_datetime(sales_data['ORDERDATE'],format='%Y-%m-%d')
print(sales_data.head(5))
print(sales_data.dtypes)

#计算模型RFM的值
#groupby()得到一维数组，并且索引是groupby 的索引

#以id汇总销售日期，并取其最大值
recency_data=sales_data['ORDERDATE'].groupby(sales_data.index).max()

#以id汇总订单，比计算订单个数
frequency_values=sales_data['ORDERID'].groupby(sales_data.index).count()

#订单金额
monetary_value=sales_data['AMOUNTINFO'].groupby(sales_data.index).sum()
print(type(monetary_value))

#计算RMF的得分

deadline_data=pd.datetime(2017, 1, 1) #指定最后期限，计算时间间隔，也即是模型R值

r_interval=(deadline_data - recency_data).dt.days #针对series格式：Series.dt.days，Number of days for each element

#cut将根据值本身来选择箱子均匀间隔，qcut是根据这些值的频率来选择箱子的均匀间隔
#因此cut，lables可能不全部表现，比如最小值1，最大值10，除此之外没有，分5分，labels=【1，2，3，4，5】，最后的label只有1和5

r_score=pd.cut(x=r_interval,bins=5,labels=[5,4,3,2,1])#日期越小，越好；分位数的labels，针对的排序是由小到大，值最大，labels最靠后
f_score=pd.cut(x=frequency_values,bins=5,labels=[1,2,3,4,5])
m_score=pd.cut(x=monetary_value,bins=5,labels=[1,2,3,4,5])


#rfm数值合并,数据框
rfm_list=[r_score,f_score,m_score] #组成列表
rfm_col_names=['r_score','f_score','m_score']

df_rfm=pd.DataFrame(np.array(rfm_list).T,index=monetary_value.index,columns=rfm_col_names)

print(df_rfm.head(5))

#计算加权得分
df_rfm['rfm_wscore']=df_rfm['r_score']*0.6 + df_rfm['f_score']*0.3 + df_rfm['m_score']*0.1


#字符合并

df_rfm_tmp=df_rfm.copy()
print(df_rfm_tmp.head(5))
df_rfm_tmp['r_score']=df_rfm_tmp['r_score'].astype(str)
df_rfm_tmp['f_score']=df_rfm_tmp['f_score'].astype(str)
df_rfm_tmp['m_score']=df_rfm_tmp['m_score'].astype(str)
print(df_rfm_tmp.dtypes)
print(type(df_rfm_tmp['r_score']))
# pandas的dtype字符串的是str，简写即可
#str.cat连接字符串的

df_rfm['rfm_comb']=df_rfm_tmp['r_score'].str.cat(df_rfm_tmp['f_score']).str.cat(df_rfm_tmp['m_score'])
df_rfm['rfm_comb']=df_rfm['rfm_comb'].astype(int)

print('{:-^80}'.format('final rfm '))
print(df_rfm.head(5))
print(df_rfm.describe())
#print(frequency_values.describe())

if not os.path.exists('G:\data_operation\python_book\chapter5\\sales2.csv'):
  df_rfm.to_csv('G:\data_operation\python_book\chapter5\\sales2.csv')

#pd.io.sql.to_sql(frame=df_rfm,con=engine,name='rfmscore2',if_exists='replace')