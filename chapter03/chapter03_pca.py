import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

data=np.loadtxt('G:\data_operation\python_book\chapter3\data1.txt')
print(data[1])
x=data[:,:-1]
y=data[:,-1]
# print(x[0])
# print(y[0])

#使用决策树decisiontreeclassifier
model_tree=DecisionTreeClassifier()
model_tree.fit(x,y)
feature_importance=model_tree.feature_importances_#输出重要性
print(feature_importance)

#使用pca,pca针对非监督学习，所以不涉及y
model_pca=PCA()
model_pca.fit(x)#输入数据，得到映射关系
u=model_pca.transform(x)#降维转换
print('niho',u.shape)
components=model_pca.components_#主成分
components_var=model_pca.explained_variance_
components_var_ratio=model_pca.explained_variance_ratio_

print(components[:2])#前两个主成分，此时维度等于原始x的维度【可以把主成分理解成原始各列的系数，新的维度最高等于原始的维度，所以有是个不同 的系数】
print(components_var[:2])
print(components_var_ratio)#每个主成分方差占比
print(pd.DataFrame(components))#compentents的数据类型是narray格式，可以转化成数据框
print(components_var_ratio[:5].sum())#前5个主成分解释比例


