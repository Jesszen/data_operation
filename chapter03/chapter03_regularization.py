"""
正则化：
英文：Regularizaiton
1、实际上更好的理解是翻译成：规则化【就是添加一个规则来约束】
2、从贝叶斯角度，正则化项就是一个先验信息
岭回归：最小二乘法+规则项【(∑iwi2)1/2】，一个圆，l2范数，压缩接近0
lasso回归：l1范数，可以筛选变量，
*********************************************
数据共线性
第一：如何检测共线？
      1、容忍度tolerance：
         每个自变量作为因变量对其他自变量做回归建模得到的残差比例。
         公式=1-R平方
         实际是rss/tss【残差平方和/固有方差】
      2、方差膨胀因子variance inflation factor,VIF
         就是容忍度的倒数。
      3、特征值
         pca主成分分析的时，原始的样本维度的特征值。
第二：解决共线问题？
     1、增大样本量
     2、岭回归/lasso回归
     3、主成分【降维】
     4、逐步回归法

"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge,Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor



data=np.loadtxt('G:\data_operation\python_book\chapter3\data5.txt')
#print(data[:10,:])
x=data[:,:-1]
y=data[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# mode_vif=variance_inflation_factor(x,exog_idx=1)#exog_idx  解释变量中作为因变量的索引值
# print('ll',mode_vif)
def vif(data):
    """

    :param data:
    :return:
    """
    n=data.shape[1]
    vif_list=[]
    for i in range(0,n):
        vif_tmp=variance_inflation_factor(data,i)
        vif_list.append(vif_tmp)
    return vif_list

kk=vif(x)
print(kk)


model_rigge=Ridge(alpha=1)#岭回归
model_rigge.fit(x_train,y_train)
print(model_rigge.coef_)
print(model_rigge.score(x_test,y_test))


model_lasso=Lasso(alpha=1)
model_lasso.fit(x_train,y_train)
print(model_lasso.coef_)
print(model_lasso.score(x_test,y_test))



model_pca=PCA()

"""
pca需要事先变量标准化，不同变量的量纲差异太大，导致载荷过大，因为其方差最大+
"""
standscalar=StandardScaler()
x_train_pca=standscalar.fit_transform(x_train)
x_test_pca=standscalar.fit_transform(x_test)
data_pca_trian=model_pca.fit_transform(x_train_pca)#已经把原始的x，变成新的数据了,并且赋值


data_pca_test=model_pca.transform(x_test_pca)
ratio_cumsm=np.cumsum(model_pca.explained_variance_ratio_)#累计求和
print(ratio_cumsm)
#print(model_pca.components_)#主成分
rule_index=np.where(ratio_cumsm > 0.8)#获取方差占比超过0.8的所有索引,结果打印(array([2, 3, 4, 5, 6, 7, 8], dtype=int64),)
min_index=rule_index[0][0]
print(rule_index)
print('kk',min_index)
data_pca_result_train=data_pca_trian[:,0:min_index+2]
data_pca_result_test=data_pca_test[:,0:min_index+2]

pca_liner_model=LinearRegression()
pca_liner_model.fit(data_pca_result_train,y_train)
print(pca_liner_model.coef_)

print(pca_liner_model.score(data_pca_result_test,y_test))

"""
相关系分析
1、相关不是因果关系
2、除了线性相关外，还可能存在指数/幂等其他形式的

"""

correlation=np.corrcoef(x,rowvar=False)#计算相关性系数矩阵【rowvar=False,按照列计算相关系数】
#print(correlation.round(2))#round保留两位小数
