

import pandas as pd
"""
异常值处理
"""

df=pd.DataFrame({'col1':[1,120,3,5,2,12,13],'col2':[12,17,31,53,22,32,43]})
print(df)
df_zscore=df.copy()
print(df_zscore)
cols=df.columns#不需要计算则，不需要加()号，这里获得列名
print(cols)

for col in cols:
    df_col=df[col]
    print(df_col)
    z_score=((df_col-df_col.mean()))/df_col.std()#得到的col列，所有行的数据，pd的魔法
    print(z_score)
    df_zscore[col] = z_score.abs() >2.2#把布尔值赋值了df score

print(df_zscore)

"""
重复值
"""

data1=['a',3]
data2=['b',2]
data3=['a',3]
data4=['c',2]

df=pd.DataFrame([data1,data2,data3,data4],columns=['col1','col2'])
print(df)
isduplicated=df.duplicated()#判断是否重复，结果是布尔值
print(isduplicated)
new_df1=df.drop_duplicates()#删除一行 完全重复的
new_df2=df.drop_duplicates(['col1'])#删除col1列中的所有的重复值，所在行的一整行
new_df3=df.drop_duplicates(['col2'])#删除col2列中的所有的重复值，所在行的一整行
new_df4=df.drop_duplicates(['col1','col2'])#col1和col2列，每一行重复的，所在的列；必须要求，两列的列值在同一行完全重复，这里是用逗号因为列名构成的列表，而非对数据库的切片操作
print(new_df1)
print(new_df2)
print(new_df3)
print(new_df4)

"""
分类数据和顺序数据转换为标准变量
真值转化法，分类或者顺序数据，从一列多值，转换成一行多列
"""

from sklearn.preprocessing import OneHotEncoder

df=pd.DataFrame({'id':[3566841,6541227,3512441],
                 'sex':['male','Female','Female'],
                 'level':['high','low','middle']})
print(df)

df_nw=df.copy()
"""
针对列名是字符串的原始数据
"""
for col_num,col_name in enumerate(df):# enumerate列的索引012，和列名【id，sex,level】
    col_data=df[col_name]
    col_type=col_data.dtype#每列的数据类型
    if col_type == 'object':
        df_nw=df_nw.drop(col_name,axis=1)#删除字符串的列
        value_sets=col_data.unique()
        for value_unique in value_sets:
            col_name_new=col_name +'_'+value_unique#新的列名
            col_tmp=df.iloc[:,col_num]#原始数据对应的colname的列，只是因为iloc的切片需要用掉索引而言,等价于df[col_name]
            new_col=(col_tmp==value_unique)#魔法一一比对，产生一列！！！直接利用矩阵series对象而无需遍历每一个值进行矩阵比较
            df_nw[col_name_new]=new_col
print(df_nw)

"""
针对原始数据，分类变量已经被数据取代，但是依然在一列中，
的真值转化法

sklearn基于numpy，所以无法处理字符串
"""

df=pd.DataFrame({'id':[3566841,6541227,3512441],
                 'sex':[1,2,2],
                 'level':[3,1,2]})

id_data=df['id']
transform_data=df.iloc[:,1:]
enc=OneHotEncoder()
df2_new=enc.fit_transform(transform_data).toarray()#toarray()把稀疏矩阵转化为矩阵，也就是稀疏的用0替代【稀疏矩阵3*5，仅仅标记行列索引及存在的值】
print(df2_new)
df2_all=pd.concat((pd.DataFrame(id_data),pd.DataFrame(df2_new)),axis=1)#拼接数组
print(df2_all)

