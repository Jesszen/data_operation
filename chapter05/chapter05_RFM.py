import pandas as pd
import time
import MySQLdb
import numpy as np
import datetime

mysql_name='root'
#腾讯云
# mysql_password='123456'
# mysql_host='118.24.26.227'
# mysql_database='python_yuny'
table_name='yuny_sales'
#本地mysql
mysql_password='123456789'
mysql_host='localhost'
mysql_database='sales'

class Jess_mysql():
    """
    设置mysql类，实现创建数据框，表，及添加数据
    """
    def __init__(self):
        self.mysql=MySQLdb.connect(user=mysql_name,host=mysql_host,password=mysql_password,database=mysql_database)
        self.conn=self.mysql.cursor()

    def create_table(self,table_names,col_names):
        """
        创建表
        :param table_names: 表名
        :param col_names: 列名，列表格式
        :return:
        """
        tables=' varchar(255),'.join(['%s'] *len(col_names))
        sql_yuju='create table if not exists `{t}` ({v} varchar(255))'.format(t=table_names,v=tables)#字段需要标注格式
        print(sql_yuju)
        ss=sql_yuju %(tuple(col_names))
        print(ss)
        self.conn.execute(ss)
        self.mysql.commit()

    def add_data(self,table_name,col_names,col_data):
        """

        :param table_name: 表名
        :param col_names: 列名，字段名
        :param col_data: 字段值
        :return:
        """
        colname=','.join(['%s']*len(col_names))
        data=','.join(['%s']*len(col_data))
        sql_yuju='INSERT INTO `{t}` ({name}) VALUES ({data});'.format(t=table_name,name=colname,data=data)
        ss=sql_yuju%(*col_names,*col_data)
        print(ss)
        self.conn.execute(ss)
        self.mysql.commit()
"""
#缺点太慢了
dtyes={'USERID':int,'ORDERDATE':object,'ORDERID':object,'AMOUNTINFO':float}
data=pd.read_csv('G:\data_operation\python_book\chapter5\\sales.csv')
data.iloc[:,1]=pd.to_datetime(data.iloc[:,1])
print(data.dtypes)

df=data.copy()
df=df.where(pd.notnull(df),'null')#将pdans Df中的NAN转化成mysql中的null，主要df.astype(object)改变df的格式为字符串
col_names=list(data.columns.values)

print(df.head(5))

t=Jess_mysql()
t.create_table(table_name,col_names)

x=data.shape[0]
for i in range(x):
    col_data=list(df.iloc[i,:])
    col_data[1]=datetime.date.strftime(col_data[1],'%Y%d%m')
    print(*col_data)
    t.add_data(table_name,col_names,col_data)
"""
'--------------------------------------------------------------------------------------------------------'

import sqlalchemy

#engine=sqlalchemy.create_engine('mysql + mysqldb://root:123456@118.24.26.227:3306/python_yuny')
engine=sqlalchemy.create_engine('mysql+mysqlconnector://{user}:{password}@{host}:3306/{database}'.format
                                (user=mysql_name,password=mysql_password,host=mysql_host,database=mysql_database))
#pd.io.sql.to_sql(frame=data,name='yunying',con=engine,index=False,if_exists='append')



#sheet_name=None，返回所有sheet，以字典格式
dict_order=pd.read_excel('G:\data_operation\python_book\chapter5\\order.xlsx',sheet_name=None,header=0)
print(dict_order.keys())
print(type(dict_order))

#第一个sheet1
df_order_train=dict_order['Sheet1']
print(df_order_train.columns.values)
print(df_order_train.dtypes)
print(df_order_train.head(5))

df_order_test=dict_order['Sheet2']

#如果第一列时索引，没有列名，默认作为索引列
df_predict=pd.read_excel('G:\data_operation\python_book\chapter5\\order_predict_result.xlsx')
print(df_predict.columns.values)
print(df_predict.head(5))


# pd.io.sql.to_sql(frame=df_order_train,name='df_order_train',con=engine,index=False,if_exists='replace')
# pd.io.sql.to_sql(frame=df_order_test,name='df_order_test',con=engine,index=False,if_exists='replace')

#pd.io.sql.to_sql(frame=df_predict,name='predict_order111',con=engine,index=False,if_exists='replace')
# chapter06_product_oders=pd.read_table('G:\data_operation\python_book\chapter6\products_sales.txt',delimiter=',')#read table  需要查看分隔符
# print(chapter06_product_oders.head(5))
#pd.io.sql.to_sql(frame=chapter06_product_oders,con=engine,name='cp6orders',index=False,if_exists='replace')

raw_data=pd.read_table('G:\data_operation\python_book\chapter6\\abnormal_orders.txt',delimiter=',',encoding='utf-8')

pd.io.sql.to_sql(frame=raw_data,name='chp6_abnormal_od',con=engine,index=False,if_exists='replace')