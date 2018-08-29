import sklearn.linear_model
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split,cross_val_score
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor




raw_data=pd.read_table('G:\data_operation\python_book\chapter4\\regression.txt',sep=' ',header=None)  # 首行不设置为列名，none
print(raw_data.head())
print(raw_data.describe())

u=raw_data.values  # pands.dataframe转化成np.array格式
print(u.shape)
#print(u[:,0:10])
cormatrix=np.corrcoef(u) #查看相关系数
print(cormatrix)


x_data=u[:,:-1]
y=u[:,-1]

"""
查看vif得分
并且删除vif 大于15的部分
"""
def vif(data):
    """
    返回vif方差膨胀因子列表
    :param data:
    :return:
    """
    vif_list=[]
    for i in range(data.shape[1]):
        vif_list.append(variance_inflation_factor(data,int(i)))
    return vif_list

vif_l=vif(x_data)
# print(vif_l)
np_vif=np.asarray(vif_l).reshape(len(vif_l),1)

def select_feature(x_data,vif_score_list,vif_value):
    """
    x-data：dataframe格式
    :param x_data:
    :param vif_score_list:
    :return:
    """
    vif_np=np.asarray(vif_score_list).reshape(len(vif_score_list),1)
    df_t=x_data.T
    new_df = pd.concat((df_t,pd.DataFrame(vif_np, columns=['vif'])), axis=1)
    new_df2=new_df[new_df.iloc[:,-1] <vif_value]
    new_x_data=new_df2.iloc[:,:-1].T
    return new_x_data

x_data2=select_feature(raw_data.iloc[:,:-1],vif_l,vif_value=15)
print(x_data2.shape)



#多重交叉
n_flods=6
model_ridge=sklearn.linear_model.Ridge(alpha=2)
model_lasso=sklearn.linear_model.Lasso(alpha=2)
model_br=sklearn.linear_model.BayesianRidge()  #贝叶斯岭回归
model_lf=sklearn.linear_model.LinearRegression()
model_etc=sklearn.linear_model.ElasticNet()  #弹性网络线性回归
model_svr=SVR()
model_liner_svr=LinearSVR()
model_gbr=GradientBoostingRegressor()

#model_gbr.score()

model_names=['ridge','lasso','bayes','linear','elasticnet','svr','linear svr','gradient']
model_dic=[model_ridge,model_lasso,model_br,model_lf,model_etc,model_svr,model_liner_svr,model_gbr]  # 不同模型对象
cv_score_list=[]
pre_y_list=[]
#交叉验证，获取模型得分，和预测y
for model in model_dic:
    scores=cross_val_score(model,x_data,y,cv=n_flods)
    cv_score_list.append(scores)
    pre_y_list.append(model.fit(x_data,y).predict(x_data))

df_cv=pd.DataFrame(cv_score_list,index=model_names)#设置索引值为各模型的名称
df_cv['mean']=df_cv.mean(axis=1)#求出每个模型评价score
print('{:-^90}'.format('model score'))
print(df_cv)
print('ddddddddddddddddddddddddddddddddddddddddd',len(pre_y_list))

#模型评价
n_sample,n_feature=x_data.shape

#m模型评估指标名称,引用函数对象
model_metrics_name=[sklearn.metrics.explained_variance_score,sklearn.metrics.mean_absolute_error,sklearn.metrics.mean_squared_error,sklearn.metrics.r2_score]

model_metrics_list=[]

for i in range(len(model_names)):
    tmp_list=[]
    for m in model_metrics_name:
        tmp_score=m(y,pre_y_list[i])
        tmp_list.append(tmp_score)
    model_metrics_list.append(tmp_list)

df_score=pd.DataFrame(model_metrics_list,index=model_names,columns=['exp','mae','mse','r2'])


print('%d samples,%d features'%(n_sample,n_feature))

print(70*'-')
print('cross validation result:')
print(df_cv)
print(70*'-')
print('regression metric score')
print(df_score)
print(70*'-')

plt.figure()

plt.plot(np.arange(x_data.shape[0]),y,color='blue',label='true y')

colors=['grey','black','green','grey','orange','red']

for i in range(len(pre_y_list)):
    residual=np.asarray(pre_y_list[i]) -np.asarray(y) # 画残差图
    plt.plot(np.arange(x_data.shape[0]),pre_y_list[i],label=model_names[i])

plt.title('regerssion result comparison')
plt.legend(loc='upper right')
plt.ylabel('real and predicted y')
plt.show()

new_point_set = [[1.05393, 0., 8.14, 0., 0.538, 5.935, 29.3, 4.4986, 4., 307., 21., 386.85, 6.58],
                 [0.7842, 0., 8.14, 0., 0.538, 5.99, 81.7, 4.2579, 4., 307., 21., 386.75, 14.67],
                 [0.80271, 0., 8.14, 0., 0.538, 5.456, 36.6, 3.7965, 4., 307., 21., 288.99, 11.69],
                 [0.7258, 0., 8.14, 0., 0.538, 5.727, 69.5, 3.7965, 4., 307., 21., 390.95, 11.28]]

for i in new_point_set:
    predict=model_gbr.predict(np.asarray(i).reshape(1,-1))
    print(predict)