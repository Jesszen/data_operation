import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,VotingClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import variance_threshold,SelectPercentile,f_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer#特征提取，还有一个词向量
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score


def summary(df):
    """
    描述变量
    :param df:
    :return:
    """

    print('{:-^80}'.format('data shape'))
    print(df.shape)
    print('{:-^80}'.format('data field dtypes'))
    print(df.dtypes)
    print(df.head(5))
    print(df.tail(5))
    print('{:-^80}'.format('data describe'))
    print(df.describe())

def summay_nan(df):
    """
    缺失值
    :param df:
    :return:
    """
    print('{:-^80}'.format('nan summary'))
    print('cols has nan %d'%df.isnull().any().sum())
    print('rows has nan %d' % df.isnull().any(axis=1).sum())
    print('{:-^80}'.format('nan details'))
    print(df[df.isnull().any(axis=1)==True].count())

def label_samples_sumamry(df):
    """
    各类别的样本数量
    :param df:
    :return:
    """
    d=df.iloc[:,1].groupby(df.iloc[:,-1]).count()
    print('{:-^80}'.format('label_samples_sumamry'))
    print(d)

#很多情况下，数据库中的分类/顺序变量为字符串类型，需要转变为数值型，但如果分类变量类别特别多，eg用户id，那么并不要做onehotencoder真值转换，因为那样会
#变成稀疏矩阵，特征大多数被稀疏矩阵产生的特征覆盖的，无法提高准确率
#def str2int(set,convert_object,unique_object,training=True):
    """
    将原始数据中的分类/顺序变量为字符串类型，需要转变为数值型
    :param set: 数据集
    :param convert_object: DictVectorizer转换对象，当Trainning=true，是空，【因为我们要训练模型，才能得到】，trainning =False，需要传入以及训练好的对象
    :param unique_object: 唯一值列表，training=True，为空，当training=False，需要传入已经从训练集中得到的唯一值列表
    :param training: 是否为训练阶段
    :return: 训练阶段：返回训练阶段的模型对象，唯一值列表，train part data ； 训练结果应用阶段，返回预测的数据
    """

dtypes = {
    'order_id':np.object,
    'pro_id':np.object,
    'use_id':np.object
}
raw_data=pd.read_table('G:\data_operation\python_book\chapter6\\abnormal_orders.txt',delimiter=',',encoding='utf-8')

#统计描述
summary(raw_data)

#t通过描述性变量，发现total money和total quantity的最大值存在异常
#我们就是要区分异常数据的，所以不用删除

#缺失值审查
summay_nan(raw_data)

#发现缺失值仅1429行，占比1%，直接删掉

raw_data=raw_data.dropna()

#查看分类变量各自样本数，审查是否存在少数样本
label_samples_sumamry(raw_data)

#结果10：3，这样的结果可以选中处理，也可以选择不处理，我们选择处理
#但再处理之前，先进行因子处理，其实就是把分类变量因子化

def sample_balance(X,y):
    """
    过抽样 解决样本不平衡问题
    :param X:
    :param y: 分类标签
    :return:
    """
    model_smote=SMOTE()
    X_smote_reshape,y_smote_reshape=model_smote.fit_sample(X,y)
    return X_smote_reshape,y_smote_reshape



#因子处理
# model_label=LabelEncoder()
# convert_cols=['cat']
# factor_cols=raw_data[convert_cols].values.reshape(raw_data.shape[0],)#要求输入格式：array-like of shape (n_samples,)，
# 否则报错：sklearn DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ),
# for example using ravel().   y = column_or_1d(y, warn=True)
# print(type(factor_cols))
# print(factor_cols.shape)
# factor_cols_result=model_label.fit_transform(factor_cols)
# print(type(factor_cols_result))
# print(factor_cols_result.shape)
# print(factor_cols_result[0:10])
# print(len(np.unique(factor_cols_result)))

convert_cols = ['cat', 'attribution', 'pro_id', 'pro_brand', 'order_source', 'pay_type', 'use_id',
                    'city']
def factor_deal(df,convert=convert_cols,training=True,factor_model=None):
    """
    LabelEncoder（）因子处理,原本目的是针对y，也就是标签，所以其要求一维数组格式（10，），不是（10，1）这个二维数组
    需要再for循环类每次适配模型后，加到列表中，并返回
    :param df:
    :return:多个labelencoder（）模型，和适配后的数据集
    """
    convert_cols=convert
    factor_last=[]
    if training:
        model_set = []
        for col in convert_cols:
            model_label = LabelEncoder()
            d=df[col].values.reshape(df.shape[0],)
            model_label.fit(d) #模型训练
            d_lab_result=model_label.transform(d)
            model_set.append(model_label)
            factor_last.append(d_lab_result)
            print('col %s has %d unique'%(col,len(np.unique(d_lab_result))))
        fac_last=np.array(factor_last)
        fac_pd=pd.DataFrame(fac_last.T,columns=convert_cols)
        return fac_pd,model_set

    else:
        model_label = factor_model
        for index,col in enumerate(convert_cols):
            d=df[col].values.reshape(df.shape[0],)
            d_lab_result=model_label[index].transform(d)
            factor_last.append(d_lab_result)
            print('col %s has %d unique'%(col,len(np.unique(d_lab_result))))
        fac_last=np.array(factor_last)
        fac_pd=pd.DataFrame(fac_last.T,columns=convert_cols)
        return fac_pd

factor_result_data,model_factor=factor_deal(raw_data)


print(factor_result_data.head(5))
print(factor_result_data.shape)


def combine_data(df1,df):
    """
    合并因子转化后的序列，和源数据集中，剩下的相关列【除去订单列】

    :param df1: 经过因子转化后的列
    :param df: 源数据集
    :return:
    """
    raw_origin=raw_data[['order_date','order_time','total_money','total_quantity','abnormal_label']]
    #删掉原来索引，方便合并两个数据
    raw_origin=raw_origin.reset_index(drop=True)
    # 如果有join_axes的参数传入，可以指定根据那个轴来对齐数据,【我们已经重置了raw origin的索引，所以可以不指定】
    # 不指定join axes则按着两个datarame的索引值合并，因为factor这个索引是新建的，索引不同，合并就会多出很多行
    #当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
    raw_data_factor=pd.concat((factor_result_data,raw_origin),axis=1,join_axes=[raw_origin.index])
    return raw_data_factor

def combine_data_test(df1,df):
    """
    合并因子转化后的序列，和源数据集中，剩下的相关列【除去订单列】

    :param df1: 经过因子转化后的列
    :param df: 源数据集
    :return:
    """
    raw_origin=df[['order_date','order_time','total_money','total_quantity']]
    #删掉原来索引，方便合并两个数据
    raw_origin=raw_origin.reset_index(drop=True)
    # 如果有join_axes的参数传入，可以指定根据那个轴来对齐数据,【我们已经重置了raw origin的索引，所以可以不指定】
    # 不指定join axes则按着两个datarame的索引值合并，因为factor这个索引是新建的，索引不同，合并就会多出很多行
    #当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
    raw_data_factor=pd.concat((factor_result_data,raw_origin),axis=1,join_axes=[raw_origin.index])
    return raw_data_factor

raw_data_factor=combine_data(factor_result_data,raw_data)
print(raw_data_factor.shape)
print(raw_data.shape)

#lambda  定义匿名函数，本质上是函数，需要传入参数
#map(),为一个序列对象，【列表，元组等等】每个元素应用指定函数功能，然后返回一个列表
#map函数+lambda可以实现小型的迭代过程，eg：日期转换，大小写转换
def datetime2int(df):
    """
    python 3.x 中map函数返回的是iterators，无法像python2.x 直接返回一个list
    目的：把数据集中，以字符串格式的日期、时间两列，转化成日期，时间类型，并拓展出，星期几，小时等列
    :param df:
    :return:
    """
    set_date=list(map(lambda dates:pd.datetime.strptime(dates,'%Y-%m-%d'),df['order_date']))#字符串全部转成日期格式
    weekday_data=list(map(lambda k:k.weekday(),set_date))
    daysinmonth_data=list(map(lambda k:k.day,set_date))
    month_date=list(map(lambda k:k.month,set_date))

    time_set=list(map(lambda times:pd.datetime.strptime(times,'%H:%M:%S'),df['order_time']))
    second_time=list(map(lambda t:t.second,time_set))
    minute_time=list(map(lambda t:t.minute,time_set))
    hour_time=list(map(lambda t:t.hour,time_set))
    #print('hour len',len(hour_time))
    final_set=[]
    final_set.extend((weekday_data,daysinmonth_data,month_date,second_time,minute_time,hour_time))
    final_matrix=np.array(final_set).T
    return final_matrix

date2int=datetime2int(raw_data_factor)
raw_data_factor_nodate=raw_data_factor.drop(['order_date','order_time'],axis=1)

final_raw_data=pd.concat((pd.DataFrame(date2int),raw_data_factor_nodate),axis=1,join_axes=[raw_data_factor_nodate.index])
print(final_raw_data.shape)
#print(final_raw_data.columns)
print('**************************',sum(final_raw_data.isnull().any()))
print(final_raw_data.head(5))

x_train,x_test,y_trian,y_test=train_test_split(final_raw_data.iloc[:,:-1],final_raw_data.iloc[:,-1],test_size=0.3)

# X_raw=final_raw_data.iloc[:,:-1]
# y_raw=final_raw_data.iloc[:,-1]


#应用过采样处理
x_train,y_trian=sample_balance(x_train,y_trian)
# print(np.isnan(X).sum())
# print(np.isnan(y).sum())

#组合分类交叉模型，和管道pipe有点相似
model_random_foreset=RandomForestClassifier(n_estimators=20,random_state=0)
model_logist=LogisticRegression()
model_bag=BaggingClassifier(n_estimators=20)
model_linesvc=LinearSVC()
model_de=DecisionTreeClassifier()
model_svc=SVC()
model_graden=GradientBoostingClassifier(learning_rate=0.2)

#组合评估器
estimastors=[('randomforest',model_random_foreset),('logistregression',model_logist),('bagging',model_bag)]
#estimastors=[('linesvc',model_linesvc),('logistregression',model_logist),('svc',model_svc)]
#投票模型
model_vote=VotingClassifier(estimators=estimastors,voting='soft',weights=[0.9,1.2,1.1])
cv=StratifiedKFold(3)

# cv_score=cross_val_score(model_vote,cv=cv,X=x_train,y=y_trian)
# print('{:-^80}'.format('cross val score'))
# print(cv_score)
# print('cross mean score %f'%(cv_score.mean()))
# model_vote.fit(x_train,y_trian)
# y_pre=model_vote.predict(x_test)

#cross 交叉验证模型
# cross_estimastor=[model_random_foreset,model_graden,model_bag,model_de]
# cross_scores=[]
# cross_name=['random forest','gradient','bag','decision']
# for i in cross_estimastor:
#     s=cross_val_score(estimator=i,X=x_train,y=y_trian,cv=cv)
#     cross_scores.append(s)
# cross_score_result=pd.DataFrame(cross_scores,index=cross_name)
# print(cross_score_result)




model_random_foreset.fit(x_train,y_trian)
print('{:*^80}'.format('score'))
print(model_random_foreset.score(x_train,y_trian))
y_pre=model_random_foreset.predict(x_test)

# model_graden.fit(x_train,y_trian)
# print('{:*^80}'.format('score'))
# print(model_graden.score(x_train,y_trian))
# y_pre=model_graden.predict(x_test)

print(accuracy_score(y_test,y_pre))
print('-' *200)

#验证集
"""
# model_logist.fit(X_raw,y_raw)
# print(model_logist.score(X_raw,y_raw))

# model_bag.fit(X_raw,y_raw)
# print('{:*^80}'.format('score'))
# print(model_bag.score(X_raw,y_raw))


raw_test_data=pd.read_csv('G:\data_operation\python_book\chapter6\\new_abnormal_orders.csv',encoding='utf-8')
print('{:*^80}'.format('test '))
print(raw_test_data)
print(model_factor)
factor_test=factor_deal(raw_test_data,training=False,factor_model=model_factor)
print(factor_test)
test_raw_factor=combine_data_test(factor_test,raw_test_data)
test_date=datetime2int(test_raw_factor)
print(test_date)
raw_data_factor_test=test_raw_factor.drop(['order_date','order_time'],axis=1)

final_raw_data=pd.concat((pd.DataFrame(test_date),raw_data_factor_test),axis=1,join_axes=[raw_data_factor_test.index])
y_prd=model_random_foreset.predict(final_raw_data)
print(len(y_prd))
print(y_prd)
"""