import re
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

"""
1、数据源的读取
"""

#只读元数据，模式只读
fn = open('G:\data_operation\python_book\chapter1\data.txt','r')

#元数据保存成列表形式
all_data=fn.readlines()

#文件读写完成后，及时关闭文件对象的占用。
fn.close()

print(all_data[1:2])

"""
2、数据预处理
"""

x =[]
y =[]

for singal_data in all_data:
    tem_data=re.split(r'\t|\n',singal_data)#分别使用\t   \n   作为分隔符
    x.append(float(tem_data[0]))#浮点型数据
    y.append(float(tem_data[1]))

#由列表转换成数组类型
x= np.array(x).reshape([100,1])
y= np.array(y).reshape([100,1])

"""
3、数据分析
"""
# 通过散点图观察
plt.scatter(x,y)
plt.show()
# 观察显示，销量随着预算增加而增加，也就是呈现线性关系，适合线性回归模型

"""
4、数据建模
"""

#创建线性模型对象，后续所有模型操作都基于该对象实现
line_moder_sales = linear_model.LinearRegression()

#代入数据，x最为自变量，y作为因变量，训练模型
line_moder_sales.fit(x,y)

"""
5、模型评估
"""
#模型系数
model_coef = line_moder_sales.coef_
#模型截距
model_intercept = line_moder_sales.intercept_

#输出线性方程
print('模型： y= %f *x + %d'%(model_coef,model_intercept))

#效果评估，计算R平方
# r2 是以Y的单位衡量的，所以是对数据失拟绝对测量方法。
# r2  = 1 - 残差平方和/数据真实的方差
# r2 表示被解释的方差比例【真实的方差，被模型解释的比列】


r2 = line_moder_sales.score(x,y)
print(r2)





