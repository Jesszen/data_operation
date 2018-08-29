import numpy as np
import pandas as pd
import sqlalchemy
import sklearn
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile,f_classif
import os

def set_summary(df):
    """
    cha
    :param df:数据框
    :return:
    """
    print('{:*^80}'.format('data overview'))
    print('Records: {0} \t Dimension{1}'.format(df.shape[0],(df.shape[1] -1)))
    print('-'*90)
    print(df.head(5))
    print('{:-^80}'.format('data describe'))
    print(df.describe())
    print('{:-^80}'.format('data dtypes'))
    print(df.dtypes)

def na_summary(df):
    """
    缺失值检测
    :param df:
    :return:
    """
    print('{:-^80}'.format('na col count'))
    na_col=df.isnull().any(axis=0) #一维布尔值数组,每列
    print(na_col)
    print('{:-^80}'.format('valid records for each col'))
    print(df.count())  #每个特征值的非缺失值数量
    print('{:-^80}'.format('nan row count'))
    na_row=df.isnull().any(axis=1)#每行，isnull  的操作逻辑，同mean这个
    print('total number of na lines: %d'%(na_row.sum()))
    print(df.isnull().any().sum())#不设置参数，默认时axis=0，也就是每列是否存在缺失值，计数列数
    print(df[df.isnull().values == True])

def label_summary(df):
    """
    查看每个类的样本分布
    :param df:
    :return:
    """
    print('{:-^80}'.format('Labels samples count'))
    print(df['value_level'].groupby(df['response']).count())  #查看每个类别【response确认的】的样本个数，df['value_level']中的特征任意选的
    print('-'*80)

#2.数据预处理
#应用onehotencoder将分类变量/顺序变量，实行真值转化法
#前提是对应的特征数据类型是int32

def type_con(df):
    """
    类型转化,将分类变量/顺序变量的dtype，转为int
    :param df:
    :return:
    """
    var_list={
        'edu':int,
        'user_level': int,
        'industry': int,
        'value_level':int,
        'act_level': int,
        'sex': int,
        'region': int
    }
    for filed,ftype in var_list.items():#dict.items(),可以遍历键值对
        df[filed]=df[filed].astype(ftype)
    print('{:-^80}'.format('df dtypes'))
    print(df.dtypes)
    return df

def na_replace(df):
    """
    前提，是指定哪些列包含缺失值 ，根据每一列自动指定缺失值;
    针对浮点数，利用均值mean()
    针对分类变量，则利用中位数替代
    :param df:
    :return:
    """
    na_rules={
                'age': df['age'].mean(),
                'total_pageviews': df['total_pageviews'].mean(),
                'edu': df['edu'].median(),
                'edu_ages': df['edu_ages'].median(),
                'user_level': df['user_level'].median(),
                'industry': df['user_level'].median(),
                'act_level': df['act_level'].median(),
                'sex': df['sex'].median(),
                'red_money': df['red_money'].mean(),
                'region': df['region'].median()
                }  # 字典：定义各个列数据转换方法
    df=df.fillna(na_rules)#接受字典格式，批量转化
    print('{:-^80}'.format('check na exists'))
    print(df.isnull().any().sum())#打印缺失值的个数，非行列数
    return df
def na_replace2(df):
    """
    前提，是指定哪些列包含缺失值 ，根据每一列自动指定缺失值;
    针对浮点数，利用均值mean()
    针对分类变量，则利用中位数替代
    :param df:
    :return:
    """
    model_inputer=sklearn.preprocessing.Imputer(missing_values='NaN',strategy='mean',axis=0) #列均值替代
    df=model_inputer.fit_transform(df)
    print('{:-^80}'.format('check na exists'))
    print(df.isnull().any().sum())#打印缺失值的个数，非行列数
    return df


def symbol_con(df,enc_object=None,train=True):
    """
    如果是训练集，则定义模型onehotencoder，训练模型，应用模型，并返回结果和模型；
    如果是测试机，传入之前训练的模型，直接transform即可
    注意点：返回的numpy格式的数据
    此外，onehotenconder 训练集，要求值域完整，这样应用到测试集，不会出现，不能转化的类
    :param df:
    :param enc_object: onehotencoder模型对象
    :param train: 是否为训练集
    :return:
    """
    convert_cols=['edu', 'user_level', 'industry', 'value_level', 'act_level', 'sex', 'region'] #需要转化的列
    df_con=df[convert_cols]   #取出需要转化的列，格式还是dataframe
    df_oringn=df[['age', 'total_pageviews', 'edu_ages', 'blue_money', 'red_money', 'work_hours']].values#取不带列名，格式是numpy数组，注意嵌套的列表
    if train == True:
        enc=OneHotEncoder()
        enc.fit(df_con)
        df_con_new=enc.transform(df_con).toarray()  #转换后的稀疏矩阵，所以需要toarray(),且结果是numpy矩阵
        matrix=np.hstack((df_con_new,df_oringn))
        print(type(matrix))
        print(matrix.shape)
        return matrix,enc
    else:
        df_con_new=enc_object.transform(df_con).toarray()
        new_matrix=np.hstack((df_con_new,df_oringn))
        print(type(new_matrix))
        print(new_matrix.shape)
        return new_matrix



#获得最佳模型
def get_best_model(X,y):
    """
    结合交叉验证获得不同参数下，分类模型的准确率
    :param X:
    :param y:真实的y值
    :return:特征选择模型
    """
    trainsform=SelectPercentile(score_func=f_classif,percentile=50) #特征选择模型，不像pca那种特征转换，这结果可解释型好，按照f classif方法选择50%的特征
    model_adaboost=AdaBoostClassifier() #建立adaboost模型对象，并没有传入参数
    model_pipe=Pipeline(steps=[('ANOVA',trainsform),('model_adaboost',model_adaboost)]) #建立管道对象，实际上是把，前后相关的步骤，以元组列表的传入形式，封装在一起
    cv=StratifiedKFold(5) #交叉检验的次数；应该直接指定cv的效果好，因为考虑到样本不平衡的问题
    n_estimators=[20,50,80,100]  #提升法的学习器个数
    score_methods=['accuracy','f1','precision','recall','roc_auc'] #设置交叉检验的指标，最后个式roc的面积
    mean_list=[] #存放不同参数，不同折的，不同评价参数的结果的均值
    std_list=[] #存放不同参数，不同折的，不同评价参数的结果的标准差
    for param in n_estimators:
        t1=time.time()
        score_list=[] #评价得分
        print('set param :%d'%param)
        for score_method in score_methods:
            model_pipe.set_params(model_adaboost__n_estimators=param)  #pipe管道传入参数的格式,两个下划线，表示链接对应步骤和其需要传入发参数
            score_tmp=cross_val_score(model_pipe,X=X,y=y,scoring=score_method,cv=cv)#每次只能得出一个模型，一个指标的得分，只不过，多少这就对应该指标的多少次得分,组成一个一维数组
            score_list.append(score_tmp)
        score_matrix=pd.DataFrame(np.array(score_list),index=score_methods)#转化成数据框
        score_mean=score_matrix.mean(axis=1).rename('mean') #得到同一个n_estimators的，不同折数的，各指标的得分的均值
        score_std=score_matrix.std(axis=1).rename('std')#得到同一个n_estimators的，不同折数的，各指标的得分的均值标准差
        score_pd=pd.concat([score_matrix,score_mean,score_std],axis=1)
        mean_list.append(score_mean) #均值添加到外边的mean list，有多少各n estimators就有多个score mean
        std_list.append(score_std)
        print(score_pd.round(2)) #打印每个n estimastors参数的交叉验证结果，且只保留两个小数
        print('-'*80)
        t2=time.time()
        tt=t2 - t1
        print('time: %d'%tt)
    mean_matrix=np.array(mean_list).T #这样每列局势一个n estimators对应的值
    std_matrix=np.array(std_list).T
    mean_pd=pd.DataFrame(mean_matrix,index=score_methods,columns=n_estimators)
    std_pd=pd.DataFrame(std_matrix,index=score_methods,columns=n_estimators)
    print('mean values for each parameter:')
    print(mean_pd)
    print('std values for each parameter')
    print(std_pd)
    print('-'*80)
    return trainsform

#获得最佳模型
def get_best_model_2(X,y):
    """
    结合交叉验证获得不同参数下，分类模型的准确率
    :param X:
    :param y:真实的y值
    :return:特征选择模型
    """
    model_scaler=StandardScaler()
    model_pca=PCA(n_components=5,svd_solver='full')#比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
    model_adaboost=AdaBoostClassifier() #建立adaboost模型对象，并没有传入参数
    model_pipe=Pipeline(steps=[('scaler',model_scaler),('pca',model_pca),('model_adaboost',model_adaboost)]) #建立管道对象，实际上是把，前后相关的步骤，以元组列表的传入形式，封装在一起
    cv=StratifiedKFold(5) #交叉检验的次数；应该直接指定cv的效果好，因为考虑到样本不平衡的问题
    n_estimators=[20,50,80,100]  #提升法的学习器个数
    score_methods=['accuracy','f1','precision','recall','roc_auc'] #设置交叉检验的指标，最后个式roc的面积
    mean_list=[] #存放不同参数，不同折的，不同评价参数的结果的均值
    std_list=[] #存放不同参数，不同折的，不同评价参数的结果的标准差
    for param in n_estimators:
        t1=time.time()
        score_list=[] #评价得分
        print('-'*120)
        print('set param :%d'%param)
        for score_method in score_methods:
            model_pipe.set_params(model_adaboost__n_estimators=param)  #pipe管道传入参数的格式,两个下划线，表示链接对应步骤和其需要传入发参数
            score_tmp=cross_val_score(model_pipe,X=X,y=y,scoring=score_method,cv=cv)#每次只能得出一个模型，一个指标的得分，只不过，多少这就对应该指标的多少次得分,组成一个一维数组
            score_list.append(score_tmp)
        score_matrix=pd.DataFrame(np.array(score_list),index=score_methods)#转化成数据框
        score_mean=score_matrix.mean(axis=1).rename('mean') #得到同一个n_estimators的，不同折数的，各指标的得分的均值
        score_std=score_matrix.std(axis=1).rename('std')#得到同一个n_estimators的，不同折数的，各指标的得分的均值标准差
        score_pd=pd.concat([score_matrix,score_mean,score_std],axis=1)
        mean_list.append(score_mean) #均值添加到外边的mean list，有多少各n estimators就有多个score mean
        std_list.append(score_std)
        print(score_pd.round(2)) #打印每个n estimastors参数的交叉验证结果，且只保留两个小数
        print('-'*80)
        t2=time.time()
        tt=t2 - t1
        print('time: %d'%tt)
    mean_matrix=np.array(mean_list).T #这样每列局势一个n estimators对应的值
    std_matrix=np.array(std_list).T
    mean_pd=pd.DataFrame(mean_matrix,index=score_methods,columns=n_estimators)
    std_pd=pd.DataFrame(std_matrix,index=score_methods,columns=n_estimators)
    print('mean values for each parameter:')
    print(mean_pd)
    print('std values for each parameter')
    print(std_pd)
    print('-'*80)
    return model_pipe

def pipe_s_p_ada(X,y):
    '''
    管道对象，封装表标准化，pca和ada
    :param X:
    :param y:
    :return:
    '''
    #model_scaler=StandardScaler() ('scaler',model_scaler),
    model_pca=PCA(n_components=10)#比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。,svd_solver='full'
    model_adaboost=AdaBoostClassifier() #建立adaboost模型对象，并没有传入参数
    model_pipe=Pipeline(steps=[('pca',model_pca),('model_adaboost',model_adaboost)]) #建立管道对象，实际上是把，前后相关的步骤，以元组列表的传入形式，封装在一起
    model_pipe.set_params(model_adaboost__n_estimators=80)
    model_pipe.fit(X,y)
    return model_pipe


#添加数据集
raw_data=pd.read_excel('G:\data_operation\python_book\chapter5\\order.xlsx',sheet_name=0)
X=raw_data.iloc[:,:-1]
y=raw_data.iloc[:,-1]

#查看的X的数据，维度减1，定义该函数的时候
set_summary(raw_data)

#查看缺失值
na_summary(raw_data)
#可以发现response这一列没有缺失值

#样本均衡审查
label_summary(raw_data)

#替换缺失值
X_t1=na_replace(X)

#类型转换
X_t2=type_con(X_t1)

#标记转化
x_new,enc_object=symbol_con(X_t2,enc_object=None,train=True)


#分类模型训练,选择最优参数
#model_pppipe=get_best_model_2(x_new,y)#打印训练模型的不同n estimators评价信息，并返回筛选帅选模型trainsfrom

model_pipe_fited=pipe_s_p_ada(x_new,y)


#单独新建一个特征筛选模型，和封装在get best model中的参数一致
select_feature=SelectPercentile(score_func=f_classif,percentile= 50)
#训练模型
select_feature.fit(x_new,y)
#筛选特征变量
x_final=select_feature.transform(x_new)

#选择学习器的数量=80
model_ada_final=AdaBoostClassifier(n_estimators=80)
model_ada_final.fit(x_final,y)

#获得测试集数据
new_data=pd.read_excel('G:\data_operation\python_book\chapter5\\order.xlsx',sheet_name=1)

#删除特定列，需要指定 axis=1,获取特征
x_test=new_data.drop('final_response',axis=1)

set_summary(new_data)
na_summary(new_data)
x_test1=na_replace(x_test)
y_test=new_data['final_response']
x_test2=type_con(x_test1)
#真值转化
x_test3=symbol_con(x_test2,enc_object=enc_object,train=False)

x_test_final=select_feature.transform(x_test3)




#预测值标签
predict_labels=pd.DataFrame(model_ada_final.predict(x_test_final),columns=['labels'])

#预测概率
predict_pro=pd.DataFrame(model_ada_final.predict_proba(x_test_final),columns=['prob1','prob2'])
predict_pd=pd.concat((new_data,predict_labels,predict_pro),axis=1)

print('{:-^80}'.format('predict info'))
print(predict_pd.head(5))
print('{:-^80}'.format('predict accuracy'))
print(accuracy_score(predict_labels,y_test))

if not os.path.exists('G:\data_operation\python_book\chapter5\\predict2.xlsx'):
    # writer=pd.ExcelWriter('G:\data_operation\python_book\chapter5\\predict.xlsx')#创建写入文本对象
    # predict_pd.to_excel(writer,'sheet1')#数据写入sheet
    # writer.save()
    predict_pd.to_excel('G:\data_operation\python_book\chapter5\\predict2.xlsx')


engine=sqlalchemy.create_engine('mysql+mysqlconnector://{user}:{password}@{host}:3306/{database}'.format(user='root',password='123456789',host='localhost',database='sales'))

#pd.io.sql.to_sql(frame=predict_pd,name='order_predict_2',index=False,if_exists='replace',con=engine)

"""
#利用管道pipe函数，测试pca的表现
#获得测试集数据
new_data=pd.read_excel('G:\data_operation\python_book\chapter5\\order.xlsx',sheet_name=1)

#删除特定列，需要指定 axis=1,获取特征
x_test=new_data.drop('final_response',axis=1)

set_summary(new_data)
na_summary(new_data)
x_test1=na_replace(x_test)
y_test=new_data['final_response']
x_test2=type_con(x_test1)
#真值转化
x_test3=symbol_con(x_test2,enc_object=enc_object,train=False)

y_pred=model_pipe_fited.predict(x_test3)
print(accuracy_score(y_pred,y_test))
"""