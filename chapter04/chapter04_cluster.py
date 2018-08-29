from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sklearn
import  matplotlib.pyplot as plt

raw_data=np.loadtxt('G:\data_operation\python_book\chapter4\cluster.txt')
print(raw_data.shape)
print(raw_data[:,0:10])

u=pd.DataFrame(raw_data,columns=None)
# 查看数据的统计量，看看是否存在量纲差异
print(u.describe())
raw_x=raw_data[:,0:-1]
raw_y=raw_data[:,-1]
n_cluster=3
model_kmeans=KMeans(n_clusters=n_cluster,random_state=0)

model_kmeans.fit(raw_x)
y_predict=model_kmeans.fit_predict(raw_x)  #预测簇类别
print('hhhhhhhhhhhhh',y_predict.shape)
print(raw_x.shape)

n_samples,n_features=raw_data.shape
#效果评估
#调整后的兰德指数
adjusted_rand_s=sklearn.metrics.adjusted_rand_score(raw_y,y_predict)
#互信息  两个相同数据标签间相似度的度量
mutual_info_s=sklearn.metrics.mutual_info_score(raw_y,y_predict)
#调整后的互信息
adjusted_mutual_info=sklearn.metrics.adjusted_mutual_info_score(raw_y,y_predict)
#同质化得分
homogeneity_s=sklearn.metrics.homogeneity_score(raw_y,y_predict)
#完整性得分
completeness_s=sklearn.metrics.completeness_score(raw_y,y_predict)
#v-measure得分
v_measure_s=sklearn.metrics.v_measure_score(raw_y,y_predict)
###非监督得分------------------------------------------------------------------------------
#平均轮廓系数
silhouette_s=sklearn.metrics.silhouette_score(raw_x,y_predict)
#calinski和Harabaz得分
calinski_harabaz_s=sklearn.metrics.calinski_harabaz_score(raw_x,y_predict)
#inertia_
inertia_s=model_kmeans.inertia_  # 样本距离最近的中心点的距离，得分越少，表示样本在各自簇内越集中，类里距离越小

print('samples: %d ,features:%d' %(n_samples,n_features))

print(70*'-')#打印分割线

print('adjusted_rand_s %.2f\nmutual_info_s %.2f\nadjusted_mutual_info %.2f\nhomogeneity_s %.2f\ncompleteness_s %.2f\nv_measure_s %.2f\nsilhouette_s %.2f\ncalinski_harabaz_s %.2f'
      %(adjusted_rand_s,mutual_info_s,adjusted_mutual_info,homogeneity_s,completeness_s,v_measure_s,silhouette_s,calinski_harabaz_s))

print(70*'-')#打印分割线

#模型可视化

centers=model_kmeans.cluster_centers_  #每个簇的中心点坐标,形状n *m
print(centers)
color_list=['red','blue','green']
plt.figure()

for i in range(n_cluster):
    index_sets=np.where(y_predict==i) #找到相同簇的索引，得到的是布尔矩阵
    print(len(index_sets))
    data_cluster=raw_x[index_sets] #获得同簇的所有x
    color=color_list[i]
    plt.scatter(data_cluster[:,0],data_cluster[:,1],c=color,marker='+')# 展示样本点
    plt.plot(centers[i][0],centers[i][1],marker='o',c='black')
plt.show()

#模型持久化  模型的结果保存到本地

import pickle
#保存至本地
pickle.dump(model_kmeans,file=open('G:\data_operation\python_book\chapter4\model_cluster','wb'))
#打开
model_cluster2=pickle.load(file=open('G:\data_operation\python_book\chapter4\model_cluster','rb'))

print(model_cluster2.cluster_centers_)

new_x=np.array([1,3])  #这样定义，打印的结果2行没有列数(2,)
print(new_x)
print(new_x.shape)

new_xx=np.array([[1,3]])# 这样定义才，shape才会是1行两列
print(new_xx)
print(new_xx.shape)

uu=new_x.reshape(1,-1)#转化成一行两列，reshape(1,-1) -1代表的是整个作为1行最大列数，不用人工数
print(uu)
print(uu.shape)
print(model_kmeans.predict(uu))
r=[]
for i in range(5):
    k=int(i+2)
    model_zhou=KMeans(n_clusters=k)
    model_zhou.fit(raw_x)
    y_label=model_zhou.fit_predict(raw_x)
    silhouette__s=sklearn.metrics.silhouette_score(raw_x,y_label)
    r.append(silhouette__s)
plt.figure()
i=range(5)
plt.plot(i,r)
plt.show()

