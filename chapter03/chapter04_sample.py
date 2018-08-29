"""
抽样
第一：为什么要抽样？
     1、数据计算能力不足
     2、数据采集限制
第二：抽样目的？
     1、快速概念验证
     2、样本不均衡【上一级的过抽样，欠抽样】
     3、定性的分析
第三：抽样注意的问题？
     A：反映运营背景
         1、数据时效性
         2、缺少关键数据【大型促销的数据】
         3、数据来源多样性
     B、满足数据分析和建模需求
         1、抽样样本数量的多少问题
             a、时间维度的
                数据只是包含一个完整的周期【月度销售预测，至少有12个月的数据】
                而且考虑季节性，节假日的波动
             b、回归/分类的建模
                特征维度*单个特征值域*100
             c、关联规则
                每个主体至少1000条以上数据
             d、异常检测建模
                因为异常数据本身小概率，所以越多越好
        2、抽样样本的分布问题
           a、非数值型的特征值域分布【分类问题】，要与整体分布一致
           b、数值型，抽样的统计量【方差，均值，偏差等】与整体分布一致
           c、异常值：全部包括进去
第四：抽样的方法
     A、简单随机抽样【前提时总体数据分布均匀】
        random.sample(,)
     B、等距抽样【总体数据分布均匀，且无周期性的数据】
     C、分层抽样【适用分类逻辑的属性，标签等数据】

"""
import numpy as np
import random
import pandas as pd

#简单随机抽样,random.sample()不能对ndarray格式的数据进行随机抽样
#可以用pandas的datafrom.sample抽样
#也可以把array转化列表在抽样
data1=np.loadtxt('G:\data_operation\python_book\chapter3\data3.txt')

#转化成数据框
data11=pd.DataFrame(data1)
print(data11.count())
print(len(data1[:]))
print(type(data1))
sample_data1=pd.DataFrame.sample(data11,10)#抽取两千个
print(sample_data1)


#转化成列表
data1_3=data1.tolist()
sample_data1_2=random.sample(data1_3,2000)
print(sample_data1_2[1:10])
#--------------------------------------------------------------------------------------------------------------------------------------------------------------

#等距抽样
#data2=pd.read_table('G:\data_operation\python_book\chapter3\data3.txt',sep=' ',names=['col1','col2','col3','col4','label'])#不确认列名name，则默认第一行为列名
data2=np.loadtxt('G:\data_operation\python_book\chapter3\data3.txt')
data22=pd.DataFrame(data2)
print(data22.columns)#查看列名
sample_count=2000
n=data22.count()[0]/sample_count
m=data22.count()[0]
print(n)
sample_data2=[]
"""
#可以用for循环，也可以while循环
for i in range(0,data22.count()[0]):
    if i < sample_count:
      sample_data2.append(data22.iloc[i*int(n),:])
    else:
        break
print(len(sample_data2))
print(type(sample_data2))
"""
i=0
while len(sample_data2) < sample_count and i*n < data22.count()[0]:
    sample_data2.append(data22.iloc[i*int(n),:])
    i+=1
sample_data2_2=np.asarray(sample_data2)
print(type(sample_data2_2))
print(sample_data2_2[:2])
print(len(sample_data2_2))

#--------------------------------------------------------------------------------------------------------------------------------------------------------------
#分层抽样
data3=np.loadtxt('G:\data_operation\python_book\chapter3\data2.txt')
print(data3[:3])
lable_data_unique=np.unique(data3[:,-1])#得到分层的值域，针对特定列+
print(lable_data_unique)
n=200
sample_tem=[]
sample_dict={}
sample_data3=[]
for k in lable_data_unique:
    for m in data3:
        if m[-1] ==k:
            sample_tem.append(m)
    each_sample=random.sample(sample_tem,n)
    sample_data3.extend(each_sample)
    sample_dict[k]=len(each_sample)
print(sample_data3[:10])
for i in sample_data3:
    print(i)
