import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import prettytable
import pydotplus
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
import imblearn.under_sampling
from sklearn.ensemble import AdaBoostClassifier


raw_data=np.loadtxt('G:\data_operation\python_book\chapter4\classification.csv',delimiter=',',skiprows=1) #delimiter含义分隔符，skiprows=1，跳过列名
print(raw_data[:,0:10])

x_raw=raw_data[:,:-1]
y_raw=raw_data[:,-1]

u=np.unique(y_raw)
print(u)
k=np.where(y_raw==1)#得到元组列表
print(len(k[0]))#查看分类为1的个数

x_train,x_test,y_train,y_test=train_test_split(x_raw,y_raw) #当数组只有一个维度，形状（n,）,不完整的形状
print(x_train.shape,y_train.shape)

#水平合并两个数组
train_data=np.hstack((x_train,y_train.reshape(y_train.shape[0],-1)))
print(train_data.shape)

#过抽样处理样本不均衡,过采样
model_trans1=SMOTE()#  引用模块，在import 函数 如果出现：module 'imblearn' has no attribute 'over_sampling'
x_train_trans,y_train_trans=model_trans1.fit_sample(x_train,y_train)
print(x_train_trans.shape)
#过抽样稍微提供了模型效果

#下采样
model_under=imblearn.under_sampling.RandomUnderSampler()
x_train_trans2,y_train_trans2=model_under.fit_sample(x_train,y_train)
#下采样显而易提高召回率，但是精准都下降的厉害

model_decision=DecisionTreeClassifier() #建立决策树模型

#model_decision=AdaBoostClassifier()
model_decision.fit(x_train,y_train)
y_pred=model_decision.predict(x_test)

n_samples,n_features=raw_data.shape
print('samples: %d,features :%d'%(n_samples,n_features))

#创建混淆矩阵
confusion_matrix_m=sklearn.metrics.confusion_matrix(y_test,y_pred)
print(confusion_matrix_m)

#创建表格
confusion_table=prettytable.PrettyTable()
confusion_table.add_row(confusion_matrix_m[0,:]) #增加第一行数据
confusion_table.add_row(confusion_matrix_m[1,:])

print('confusion matrix')
print(confusion_table)

#h核心评估指标

y_score=model_decision.predict_proba(x_test)  #决策树的预测概率
fpr,tpr,thresholds=sklearn.metrics.roc_curve(y_test,y_score[:,1])  #roc
print(70*'--')
print(fpr,tpr,thresholds)
auc_s=sklearn.metrics.auc(fpr,tpr)  #auc

accuracy_s=sklearn.metrics.accuracy_score(y_test,y_pred)  #准确率：（真阳+真阴）/所有samples
precision=sklearn.metrics.precision_score(y_test,y_pred) #精准度：真阳/（真阳+假阳）   即：真阳/预测阳
recall_s=sklearn.metrics.recall_score(y_test,y_pred) #召回率   预测  真阳/样本事实阳
f1_score=sklearn.metrics.f1_score(y_test,y_pred)  #f1得分

core_metrics=prettytable.PrettyTable()
core_metrics.field_names=['auo','accuracy','precision','recall-s','f1_score']
core_metrics.add_row([auc_s,accuracy_s,precision,recall_s,f1_score])
print('core score')
print(core_metrics)

#模型可视化
names_list=['age','gender','income','rfm_score']
color_list=['red','green','blue','black']

plt.figure()

plt.subplot(1,2,1)
plt.plot(list(fpr),list(tpr),label='ROC')#x,y取值可以是列表，和np数组，否则无法画图
plt.plot([0,1],[0,1],linestyle='--',color='black',label='random chance') #检测参数会不会写错
plt.title('roc')
plt.xlabel('false positive rate')
plt.ylabel('ture postive rate')
plt.legend(loc=0)#参数loc=0，

feature_importance=model_decision.feature_importances_
plt.subplot(1,2,2)
plt.bar(np.arange(feature_importance.shape[0]),feature_importance,tick_label=names_list,color=color_list)
plt.title('feature importance')
plt.xlabel('feature')
plt.ylabel('importance')


plt.suptitle('classification result')
plt.show()

#保存决策图位pdf文件

dot_data=sklearn.tree.export_graphviz(model_decision,max_depth=5,out_file=None,feature_names=names_list,filled=True,rounded=True)#out_file=None，不生成dot文件，赋值给dodata，如果去掉out_file=None，则生成一个tree.dot文件，且无法赋值
graph=pydotplus.graph_from_dot_data(dot_data)
if not os.path.exists('G:\data_operation\python_book\chapter4\\jj.pdf'):
    graph.write_pdf('G:\data_operation\python_book\chapter4\\jj.pdf') #保存为pdf

#模型应用
X_new = [[40, 0, 55616, 0], [17, 0, 55568, 0], [55, 1, 55932, 1]]
for i,data  in enumerate(X_new):
    ped_new=model_decision.predict(np.asarray(data).reshape(1,-1))
    print('classification for %d record is %d' %(i,ped_new))
