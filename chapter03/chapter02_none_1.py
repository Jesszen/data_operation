import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
"""
pandas缺失值用  NaN  来表示
mysql 缺失值用 NULL 来表示
python 缺失值用 None 来表示

"""

df=pd.DataFrame(np.random.rand(6,4),columns=['col1','col2','col3','col4'])#生成6行4列的随机数
print(df)
df.iloc[1:3,2]=np.nan  #【第二+第三行】的第三列设为缺失值

df.iloc[4,3]=np.nan
print(df)

nan_all = df.isnull()#查看所有的缺失值【返回的结果是布尔值矩阵】
print(nan_all)

nan_col1=df.isnull().any()#获取含缺失值的列【某列至少含有一个缺失值】
nan_col2=df.isnull().all()#获取整个列都是缺失值的列

print(nan_col1)
print(nan_col2)

df2=df.dropna()#直接丢弃,默认是按照行来丢弃，哪一行包含缺失值，整行丢弃
print(df2)

#*******
#使用sklearn 将缺失值替换为特定值

nan_model = Imputer(missing_values='NaN',strategy='mean',axis=0)# 建立替换规则模型，用列均值替代缺失值
#axis=0  表示向下【沿着列的方向垂直向下】执行具体的方法
#axis=1  表示水平向右【沿着行的方向水平朝右】执行具体的方法
#执行的结果，由具体的方法决定，我们这mean，axis=0，自然求的是列的均值

nan_result = nan_model.fit_transform(df)# 应用模型规则,,,为什么nan_model.fit_transform() 因为我们是对数据进行整理，而不是训练模型  nan_model.fit_transform()
                                        #fit  和transform联合起来使用，第一步适配也就是算出缺失值列的均值，第二步transform 
print(nan_result)

# 使用pandas将确实值替换为特定值

nan_result_1 = df.fillna(value=df.mean()['col3':'col4'])#  等价于sklearn的模型，只是要指定列名，skleran是自动定位包含确实的列!!!只有df对应方法才可以['col3':'col4']选中两个列名，否则只能['col3'】
print(nan_result_1)

nan_result_2 = df.fillna(method='backfill')#用列下一行的值替代缺失值【如果下一行依然是缺失值，那么在下一行】
print(nan_result_2)

nan_result_3 = df.fillna(method='pad')#用列前一行的值替代缺失值【如果前一行依然是缺失值，那么在下一行】
print(nan_result_3)

nan_result_4 = df.fillna(value=0)# 直接用0值代替
print(nan_result_4)

nan_result_5 =df.fillna(value={'col3':1.1,'col4':2.0})#指定列名【含缺失值的列名】指定值替代，如果没有正确指定含缺失值的列名，则不执行
print(nan_result_5)
print(df.mean()['col2':'col3'])#取数据框的指定列，必须用冒号，而非逗号

nan_result_6=df.replace(np.nan,0)
print(nan_result_6)