import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import prettytable
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf  #自相关，偏相关展示图
from statsmodels.tsa.stattools import adfuller  #adf检测库
from statsmodels.stats.diagnostic import acorr_ljungbox #随机性检测库
from statsmodels.tsa.arima_model import ARMA,ARIMA

data=pd.read_csv('G:\data_operation\python_book\chapter4\\AirPassengers.csv',index_col='Month')
data.index=pd.to_datetime(data.index)
# print(data)
# plt.figure()
# plt.plot(data)
# plt.show()

#多次用到表格

def pre_table(table_name,table_rows):
    """

    :param table_name: 表格列名
    :param table_rows: 表格内容
    :return: 展示表格对象
    """
    table = prettytable.PrettyTable()
    table.field_names=table_name #列名
    for i in table_rows:
        table.add_row(i)
    return table

#稳定性检测+数据平稳处理
def get_best_log(ts,max_log=5,rule1=True,rule2=True):
    """
    稳定性检测+数据平稳处理
    :param ts: 时间序列格式数据，Series格式
    :param max_log: 最大的log处理次数
    :param rule1:
    :param rule2:
    :return:log处理次数，平稳处理的后的时间序列数据
    """
    if rule1 and rule2:
        return 0,ts
    else:
        for i in  range(1,max_log):
            ts=np.log(ts)
            lbvalue,pvalue2=acorr_ljungbox(ts,lags=1) #白噪音简称，目的时间序列是否都是白噪声
            adf,pvalue1,usedlag,nobs,critical_values,icbest=adfuller(ts)  #ADF检测，同样是检测ts是否平稳
            rule1=(adf < critical_values['1%'] and adf <critical_values['5%'] and adf <critical_values['10%'] and pvalue1 <0.01) #稳定性检测
            rule2=(pvalue2 <0.05)
            rule3=(i<5)
            if rule1 and rule2 and rule3:
                print('the best log n is :{0}'.format(i))
                return i,ts

#还原经过平稳处理的数据。
def recover_log(ts,lon_n):
    """
    还原经过平稳处理的数据
    :param ts: 经过log平稳处理后的数据
    :param lon_n: log方法处理的次数
    :return: 还原后的时间序列

    """
    for i in range(lon_n):
        ts=np.exp(ts)
    return ts

#平稳性检测
def adf_val(ts,ts_title,acf_title,pacf_title):
    """
    先时间序列图，然后换自相关，偏相关图，以表格形式输出adf检测结果，
    并返回adf的检测结果  adf值，p值和critical值
    :param ts: 时间序列Series类型
    :param ts_title: 时间序列图的标题名，字符串格式
    :param acf_title: acf图【自相关图】的标题名，字符串格式
    :param pacf_title: pacf图【偏相关图】的标题名，字符串格式
    :return: adf值，adf的p值，三种状态的检验值
    """
    plt.figure()
    plt.plot(ts)#时间序列图
    plt.title(ts_title)
    plt.show()
    plot_acf(ts,lags=20,title=acf_title).show()#自相关图
    plot_pacf(ts,lags=20,title=pacf_title).show()#偏相关图
    adf, pvalue, usedlag, nobs, critical_values, icbest=adfuller(ts) #稳定性检测
    table_names=['adf','pvalue','usedlag','nobs','critical_values','icbest']
    table_rows=[[adf, pvalue, usedlag, nobs, critical_values, icbest]]#嵌套列表,目的是for i in tables rows循环中，作为一行填入
    adf_table=pre_table(table_names,table_rows)
    print('stocjastic score')#打印标题
    print(adf_table)
    return adf,pvalue,critical_values

#白噪声检测
def acorr_val(ts):
    """

    :param ts:时间序列，Series格式
    :return: 白噪声检测的p值和打印数据表格对象
    """
    lbvalue,pvalue=acorr_ljungbox(ts,lags=1) #白噪声检测
    table_name=['lbvalue','pvalue']
    table_rows=[[lbvalue,pvalue]]
    acorr_1jungbox_table=pre_table(table_name,table_rows) #展示表格对象
    print('stationnarity score')
    print(acorr_1jungbox_table)
    return pvalue

#arma最优模型训练
def arma_fit(ts):
    """

    :param ts: 时间序列数据，Series
    :return: 最有状态下的p值，q值，arma模型对象，pdq数据框和展示参数表格对象
    """
    max_count=int(len(ts)/10) #最大的循环次数=时间序列长度的1/10
    bic=float('inf')  #初始值，浮点型数据，值域正无穷
    tem_score=[] #临时保存p,q,aic,bic hqic值
    for tem_p in range(max_count+1):#模型没有停止条件，一直循环到 结束，然后选择bic最小的那个模型
        for tem_q in range(max_count+1):
            model=ARMA(ts,order=(tem_p,tem_q)) #创建arma对象，并传入参数ts序列和pq值
            try:
                result_arma=model.fit(disp=-1,method='css')#模型训练,disp=-1 #不打印收敛信息
            except:
                continue
            finally:
                tem_aic=result_arma.aic
                tmp_bic=result_arma.bic
                tmp_hqic=result_arma.hqic
                tem_score.append([tem_p,tem_q,tem_aic,tmp_bic,tmp_hqic]) #训练参数追加到临时列表中。
                if tmp_bic < bic:#第一次循环，肯定小于bic，真，bic赋值为当前的tmp bic，模型循环结束后，获得bic最小的那个模型和其结果
                    p=tem_p
                    q=tem_q
                    model_arma=result_arma
                    aic=tem_aic
                    bic=tmp_bic
                    hqic=tmp_hqic
    pd_matrix=np.array(tem_score)#所有循环的结果
    pdq=pd.DataFrame(pd_matrix,columns=['p','q','aic','bic','hqic'])
    table_names=['p','q','aic','bic','hqic']
    table_rows=[[p,q,aic,bic,hqic]]
    parameter_table=pre_table(table_names,table_rows)
    print('best p and q')
    print(parameter_table)
    return model_arma #最有状态下的模型对象



#模型训练和效果评估
def train_test(model_arma,ts,lon_n,rule1=True,rule2=True):
    """

    :param model_arma: 最优的模型对象
    :param ts: 时间序列数据
    :param lon_n: 平稳处理的log的次数
    :param rule1:
    :param rule2:
    :return: 还原后的时间序列
    """
    train_predict=model_arma.predict() #训练集的预测值
    if not (rule1 and rule2):#两个条件如有任意一个不满足,若果是原始时间序列【rule1和rule2很大程度不稳定，Falase】，则执行复原为原始序列的操作
        train_predict=recover_log(train_predict,lon_n)#恢复为平稳处理前的时间序列值
        ts=recover_log(ts,lon_n)#恢复原始时间序列
    ts_data_raw=ts[train_predict.index] #
    RMSE=np.sqrt(np.sum((train_predict-ts_data_raw))**2/ts_data_raw.size) #均方误差平方和
    plt.figure()
    plt.plot(train_predict,label='predict data')
    #train_predict.plot(label='predict data',style='--')
    plt.plot(ts_data_raw,label='raw data')
    #ts_data_raw.plot(label='raw data',style='-')
    plt.legend(loc='best')
    plt.title('raw data and predicted data with RMSE %0.2f'%RMSE)
    plt.show()
    return ts

# 预测未来指定时间项的数据
def predict_data(model_arma,ts,log_n,start,end,rule1=True,rule2=True):
    """

    :param model_arma: 最有的arma模型对象
    :param ts: 时间徐磊
    :param log_n: 平稳性处理的次数，方法是log转化
    :param start: 预测数据开始的时间索引
    :param end: 预测数据结束的时间索引
    :param rule1:
    :param rule2:
    :return:
    """
    predict_ts=model_arma.predict(start=start,end=end) #预测未来指定的时间项数据
    print('{:*^60}'.format('predict result'))
    if not (rule2 and rule1):#预测数据是经过平稳处理的数据，则要还原
        predict_ts=recover_log(predict_ts,log_n)
    print(predict_ts)
    plt.figure()
    plt.plot(ts,label='raw time series')
    #ts.plot(label='raw time series',style='-')
    plt.plot(predict_ts,label='predicted data')
    #predict_ts.plot(label='predicted data',style='--')
    plt.legend(loc=0)
    plt.title('predicted time series')
    plt.show()


#读取数据
#data_parse=lambda dates:pd.datetime.strptime(dates,'%m-%d-%Y')
ts_date=pd.read_table('G:\data_operation\python_book\chapter4\\time_series.txt',delimiter='\t',index_col='date')

ts_date=ts_date['number'].astype('float32')
ts_date.index=pd.to_datetime(ts_date.index)
#print(ts_date)
print('data summary')
print(ts_date.describe())
#plt.plot(ts_date)

#稳定性检测
adf,pvalue1,critical_values=adf_val(ts_date,'raw time series','raw acf','raw pacf')#原始序列的稳定性检测
pvalue2=acorr_val(ts_date)#原始序列的白噪声检测
print(pvalue2)
rule1=(adf < critical_values['1%'] and adf < critical_values['5%'] and adf < critical_values['10%'] and pvalue1 <0.01)
rule2=(pvalue2[0,] <0.05)

#对时间序列稳定性处理
log_n,ts_date=get_best_log(ts_date,max_log=5,rule1=rule1,rule2=rule2)


adf,pvalue1,critical_values=adf_val(ts_date,'final time series','final acf','final pacf')
pvalue2=acorr_val(ts_date)

#训练模型
#model_arma=arma_fit(ts_date)
model_arma1=ARMA(ts_date,order=(2,4))
model_arma=model_arma1.fit(disp=-1,method='css')

#模型训练和效果评估
ts_date=train_test(model_arma,ts=ts_date,lon_n=log_n,rule1=rule1,rule2=rule2) #还原后的时间序列,规则原始序列的结果，一般原始的序列，都不满足平稳性

#模型应用
start='1991-07-28'
end='1991-08-02'

predict_data(model_arma,ts_date,log_n,start,end,rule1=rule1,rule2=rule2)

