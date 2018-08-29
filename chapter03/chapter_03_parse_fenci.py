"""
自然文本预处理
文本非结构化的数据
特征抽取出来，也就是把文本量化为特征向量。
1、基本处理
   去除无效标签
   编码转换       #读取文本fn=open('text.txt',mode='r',encoding='utf-8')
   去除停用词
2、分词
   概念：将一连串的字符串，按照一定逻辑分割成单独的词组。
3、文本转向量
   通常使用向量空间模型来描述向量文本【文档特征矩阵】

"""
import pandas as  pd
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


#定义分词器
def jieba_cut(string):
    """
    将一行字符串，分解成词组
    :param string:
    :return:
    """
    word_list=[]
    seq_list=jieba.cut(string)

    for word in seq_list:
        word_list.append(word)
    return word_list

fn=open('G:\data_operation\python_book\chapter3\\text.txt',mode='r',encoding='utf-8')
string_lines=fn.readlines()
#string_lines=fn.read()
fn.close()

#展现分词的过程和呈现的结果
seq_list=[]
for i in string_lines:
    tmp=jieba_cut(i)
    print(tmp)
    seq_list.append(tmp)#嵌套列表结构，有多少行就有多少个列表
print(len(seq_list[4]))

#利用定义的分词器作为TfidfVectorizer模型的分词器
"""
TfidfVectorizer
term frequency inverse document frequency vectorizer
不仅考虑词频，而且考虑词频在文章中的倒数，进行加权限制
文本量多的时候，tfidf可以压制常用词汇对文章分类的干扰

"""
stop_words=['\n','/','\'','”','“',',','，','和','是','随着','对于','对','中','与','在','、','。','；', '）','都','能']
vector_model=TfidfVectorizer(stop_words=stop_words,tokenizer=jieba_cut)#词向量模型,要求分词器可迭代
x=vector_model.fit_transform(string_lines)#稀疏矩阵
vector_value=x.toarray()#稀疏矩阵转化成常规矩阵

v_name=vector_model.get_feature_names()#得到特征维度的含义
vector_pd=pd.DataFrame(vector_value,columns=v_name)
print(len(v_name))
print(vector_pd)
