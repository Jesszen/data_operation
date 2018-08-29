"""
数据预处理之样本分类不均衡
第一：场景：分类数据的样本不均衡【】
eg：信用卡诈骗，一个样本总量1w，其中正常样本0.95w，诈骗样本只有500个。
    如果不做处理的化，直接拟合模型，两个后果：
    1、拟合模型，包含少数样本的特征过少，无法提取规律【类似当作噪声数据被处理掉了】
    2、严重依赖少数样本的数据，造成过拟合，泛化性能差
第二：那些运营场景出现：
    1、异常值检测：信用卡欺诈，电商漏洞
    2、客户流失：垄断行业，比如移动大量顾客流失
    3、罕见事件
    4、频率发生低的事件
第三：怎么解决？
    A、过抽样，欠抽样
        1、过抽样over-sampling：最简单的就是直接把少数样本直接复制
            imblearn.over_sampling.SMOTE算法
        2、欠抽样under-sampling：直接把样本中多数，随机减少到和小样本数量一个量级
           imblearn.under_sampling.RandomUnderSampler
    B、通过正负样本的惩罚权重
        权重设置为不同样本的数量呈反比
        sklearn.SVC（）一个参数class_weight

    3、通过组合/集成的方法解决
       把样本分成10份，每份的样本数量中：少数样本数量=多数样本数量
       我们就得到10个训练集和对应的训练模型
       最后应用时，使用组合法产生最终的分类结果
第四：如何选择处理方法？
    1、考虑每个样本分布情况和总体样本的分布情况
    2、后续数据建模算法的适应性

"""

import pandas as  pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsemble

df=pd.read_table('G:\data_operation\python_book\chapter3\data2.txt',sep=' ',names=['col1','col2','col3','col4','col5','label'])
#pd.read_table适合读取数据框形式的数据
print(df.iloc[1:5,:])
"""
da=np.loadtxt('G:\data_operation\python_book\chapter3\data2.txt')
#np.loadtxt读取数值型数据，可以通过pd.Dataframe()函数转换
print(da[1])
"""
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
groupby_origin=df.groupby('label').count()
print(groupby_origin)
#print(df.count())
print(df.describe())

#使用smote进行过抽样处理
model_smote=SMOTE()
x_smote,y_smote=model_smote.fit_sample(x,y)#输入数据，并进行过抽样处理
print(type(x_smote))#<class 'numpy.ndarray'>
x_smote_frame=pd.DataFrame(x_smote,columns=['col1','col2','col3','col4','col5'])
y_smote_frame=pd.DataFrame(y_smote,columns=['label'])
smote_resample=pd.concat((x_smote_frame,y_smote_frame),axis=1)#不平衡样本预处理后的数据,可以用于后续操作
print(smote_resample.groupby('label').count())

#使用RandomUnderSampler

model_under=RandomUnderSampler()
x_under,y_under=model_under.fit_sample(x,y)
x_under_frame=pd.DataFrame(x_under,columns=['col1','col2','col3','col4','col5'])
y_under_frame=pd.DataFrame(y_under,columns=['label'])
under_redample=pd.concat((x_under_frame,y_under_frame),axis=1)
print(under_redample.groupby('label').count())


#使用svc
model_svc=SVC(class_weight='balanced')#类别权重设置为balance
k=model_svc.fit(x,y)
print(model_svc.score(x,y))
print('svc',k)


#使用集成方法EasyEnsemble

model_ensemble=EasyEnsemble()
x_ensemble,y_ensemble=model_ensemble.fit_sample(x,y)
print(x_ensemble.shape,y_ensemble.shape)#分成了10份数，每一份的数量时58*2，因为多数样本：少数样本=1：1
#(10, 116, 5) (10, 116)
index_num=1#对应第一个维度【10, 116, 5) (10, 116)】
#抽取其中一份
x_ensemble_frame=pd.DataFrame(x_ensemble[index_num],columns=['col1','col2','col3','col4','col5'])
y_ensemble_frame=pd.DataFrame(y_ensemble[index_num],columns=['label'])
ensemble_frame=pd.concat((x_under_frame,y_ensemble_frame),axis=1)
print(ensemble_frame.groupby('label').count())