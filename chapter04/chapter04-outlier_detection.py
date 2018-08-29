import numpy as np
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split  #针对监督数据
import matplotlib.font_manager

raw_data=np.loadtxt('G:\data_operation\python_book\chapter4\\outlier.txt')
print(len(raw_data[:,1]))

train_set=raw_data[0:900,:]
test_set=raw_data[900:,:]

model_oneclasssvm=OneClassSVM(nu=0.1)
model_oneclasssvm.fit(train_set)

pre_test_outliers=model_oneclasssvm.predict(test_set) #形状时（100，)合并数据集需要reshape（100，1）
print(pre_test_outliers.shape)
total_test_data=np.hstack((test_set,pre_test_outliers.reshape(pre_test_outliers.shape[0],1)))  #合并成一个数据集

normal_test_data=total_test_data[total_test_data[:,-1]==1] #正常的点
outlier_test_data=total_test_data[total_test_data[:,-1]==-1] #异常点

n_outliers=outlier_test_data.shape[0]
print(n_outliers)
total_count_test=test_set.shape[0]

print('outliers : {0}/{1}'.format(n_outliers,total_count_test))
print('{:x^60}'.format('all result data (limit 5)'))

print(total_test_data[:,0:5])

#异常检测结果展示

plt.style.use('ggplot')  #选择样式库
fig=plt.figure()#创建画布对象
ax=Axes3D(fig) #将画布转化成3d模型，事实上我们数据有5个维度，我们只是利用前三个维度作画

s1=ax.scatter(xs=normal_test_data[:,0],ys=normal_test_data[:,1],zs=normal_test_data[:,2],s=100,marker='o',color='green',
              edgecolor='blue')#正常样本3D图
s2=ax.scatter(xs=outlier_test_data[:,0],ys=outlier_test_data[:,1],zs=outlier_test_data[:,2],
              s=100,color='red',edgecolor='blue',marker='o')#异常样本3D图

ax.w_xaxis.set_ticklabels([])#隐藏x轴标签，只保留刻度
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.legend([s1,s2],['mormal points','outliers'],loc=0)# 图列，此外s1 s2加上标签
plt.title('novelty detectiaon')
plt.show()




