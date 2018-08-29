import pandas as pd
import sqlalchemy
import _mysql_connector
mysql_name='root'
import pymysql

#腾讯云
# mysql_password='123456'
# mysql_host='118.24.26.227'
# mysql_database='python_yuny'
# table_name='yuny_sales'
#本地mysql
mysql_password='123456789'
mysql_host='127.0.0.1'
mysql_database='sales'

raw_data=pd.read_table('G:\data_operation\python_book\chapter6\\abnormal_orders.txt',delimiter=',',encoding='utf-8')

engine=sqlalchemy.create_engine('mysql+mysqldb://{name}:{password}@{host}:3306/{databasse}?charset=utf8mb4'.format
                                (name=mysql_name,password=mysql_password,host=mysql_host,databasse=mysql_database))

# Lost connection to MySQL server at 'localhost:3306', system error: 10054 远程主机强迫关闭报错，原因是适用这个驱动mysqlconnector，改为pymysql，就好了

pd.io.sql.to_sql(frame=raw_data,name='chp6_abnormal_od',con=engine,index=False,if_exists='replace')