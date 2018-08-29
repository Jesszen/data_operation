import sys
import pandas as pd
import graphviz
#sys.path.append('G:\data_operation\python_book\chapter4\\apriori.py')
from data_operation.chapter04 import apriori

data_file_path='G:\data_operation\python_book\chapter4\\association.txt'
fileName ='G:\data_operation\python_book\chapter4\\association.txt'

# minS=0.1 #支持度
# minC=0.38 #置信度

data_set=apriori.createData(data_file_path) # 嵌套列表
print(data_set)
"""
L,suppdata=apriori.apriori(data_set,minSupport=minS)   #满足支持度的规则

#rules=apriori.generateRules(fileName=data_file_path,L=L,supportData=suppdata,minConf=minC) #满足置信度的规则

#关联规则结果评估，占位符format格式输出
model_summary='data record :{1} \n association rules count:{0}' #{1}  {0} 表示索引

#print(model_summary.format(len(rules),len(data_set)))

"""
