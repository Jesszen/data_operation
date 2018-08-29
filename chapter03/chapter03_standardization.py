"""
数据标准化
1、Z-Score标准化
   x1=(x-mean)/std
   原始数据减去均值后，在除以标准差。
   处理后的数据，呈现0均值，方差为1的正太分布，是一种去中心化的方法
   局限性：改变数据的分布，不适用对稀疏数据做处理
   稀疏数据的特征：很多值为0，少数为1，主要协同过滤

2、归一化max-min
   x1=(x-min)/(max-min)
   处理后的数据，完全落入【0，1】区间，能较好保存原始数据的数据结构

3、适用稀疏数据的maxabs
   x1=x/abs(x.max)
   原始数据除以绝对值最大值
   处理后数据录入【-1，1】这个区间
4、适用离群点的RobustScaler
   因为离群点记过z-score标准化后，容易失去离群点的信息


"""
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data=np.loadtxt('G:\data_operation\python_book\chapter3\data3.txt')
print(data.size)#个数 =行*列

# z-score标准化
zscore_scaler=preprocessing.StandardScaler()#导入模型
data_scaler_1=zscore_scaler.fit_transform(data)
print(data_scaler_1)

#Max-min标准化
minmax_scaler_2=preprocessing.MinMaxScaler()#导入模型，其实模型被封装成一个类
data_scaler_2=minmax_scaler_2.fit_transform(data)

#maxaba标准化

maxabs=preprocessing.MaxAbsScaler()
data_scaler_3=maxabs.fit_transform(data)

#Robuststand标准化
robuststand=preprocessing.RobustScaler()
data_scaler_4=robuststand.fit_transform(data)

#画图展示

data_list=[data,data_scaler_1,data_scaler_2,data_scaler_3,data_scaler_4]
scalar_list=[15,10,15,10,15,10]# 创建点尺寸列表
title_list=['row data','z score data','maxmin data','maxabs data','toubust data']
col_list=['black','blue','green','yellow','red']
merker_list=['o',',','+','s','p']#样式列表

for i ,data_single in enumerate(data_list):
    plt.subplot(2,3,i+1)#子网格，2*3，从第一个开始画
    plt.scatter(data_single[:,3],data_single[:,-1],s=scalar_list[i],c=col_list[i],marker=merker_list[i])
    plt.title(title_list[i])
plt.suptitle('raw data and standardized data')
plt.show()

