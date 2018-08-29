import jieba
import matplotlib.pyplot as plt
import re
import numpy as np
import collections
from PIL import Image
import wordcloud
import time
import jieba.analyse
import jieba.posseg
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics

"""
t1=time.time()
strings_data=[]
with open(file='G:\data_operation\python_book\chapter4\\xinsong.txt',encoding='utf-8') as f:
    for i in f:
         strings_data.append(i.strip().rstrip())

strings_data=str(strings_data)
print(strings_data[1:100])


pa=re.compile(r'|\t|\n|\.|，|,|\?|-|--|\"|:|;|\(|\)|\\|\'|“|”')
strings_data=re.sub(pattern=pa,repl='',string=strings_data)

stop_path='G:\data_operation\python_book\stop_words.txt'
def stop_words_list(file,encoding):
    stopwords_list=[line.strip() for line in open(file=file,encoding=encoding).readlines()]
    return stopwords_list

#文本分词
seq_list_exact=jieba.cut(sentence=strings_data,cut_all=False) #精确分词模式，已经把文章分词了,形成的可迭代对象，但不是列表，如果tf则要定义个函数，
print(seq_list_exact)
object_list=[]
#停用词
remove_words=stop_words_list(file=stop_path,encoding='utf-8')
for i in seq_list_exact:
    if i==' ':
        continue
    if not i in remove_words:
        object_list.append(i)

#词频统计
word_counts=collections.Counter(object_list) #词频统计，输出字典格式
word_counts_top=word_counts.most_common(10)#获取前5个高频词


for m,n in word_counts_top:
    print(len(m))
    print('most common string %s,counts %d'%(m,n))


#词频展示
mask=np.array(Image.open('G:\data_operation\python_book\chapter4\\11280.jpg')) #定义词频背景
wc=wordcloud.WordCloud(font_path='C:/Windows/Fonts/STXIHEI.TTF',mask=mask,max_words=20,max_font_size=120)

wc.generate_from_frequencies(word_counts) #生成词云

image_colors=wordcloud.ImageColorGenerator(mask)#从背景图建立颜色方法
wc.recolor(color_func=image_colors)#将词云设置为背景图方案

plt.imshow(wc)
plt.axis('off') #关闭坐标
plt.show()

t2=time.time()
print(t2-t1)

"""

#关键字提取

# fn=open('G:\data_operation\python_book\chapter4\\article1.txt')
# strings_art=fn.read()#整篇文档作为一个字符串读取
# fn.close()
#
# tags_pairs=jieba.analyse.extract_tags(sentence=strings_art,topK=8,withWeight=True,withFlag=True,allowPOS=['ns','n','vn','nr'])#withflag  是否此行标注，允许标注的词性
# #withweight  设置if-idf的权重，关键词的权重跟其出现次数成正比，跟其频率成反比
# #allowPOS只允许提取特定词语类别的标签
# print(tags_pairs)
# tags_list=[]
#
# for i in tags_pairs:
#     print(i)                                     # 打印的结果(pair('产品', 'n'), 0.1389534531062063)
#     tags_list.append((i[0].word,i[0].flag,i[1]))#元组列表，需要字典格式引用把
#
# tags_pd=pd.DataFrame(tags_list,columns=['word','flag','weight'])
# print(tags_pd)

#文本聚类


def jieba_cut(string):
    """
    筛选形容词词性的分词列表
    :param string:
    :return:
    """
    s=[]
    k=jieba.posseg.cut(string)         #区别于jieba.cut()，在于该分词结果，是一个元组列表，包含每个分词和它的词性，可以筛选特定词性的分词
    for i in k:
        if i.flag in ['a','ag','an']:#只选择形容词
           s.append(i.word)
    return s

fn=open('G:\data_operation\python_book\chapter4\\comment.txt',encoding='utf-8')
string_comment=fn.readlines()#读取文件内容为列表
fn.close()

def stop_extract(path,encod):
    with open(file=path,encoding=encod) as f:
        stops=[line.strip() for line in f.readlines()]
    return stops

stop_words=stop_extract(path='G:\data_operation\python_book\stop_words.txt',encod='utf-8')
stop_words= ['…', '。', '，', '？', '！', '+', ' ', '、', '：', '；', '（', '）', '.', '-']
print(stop_words)

#创建词向量模型
vectorizer=TfidfVectorizer(stop_words=stop_words,tokenizer=jieba_cut,use_idf=True)

X=vectorizer.fit_transform(string_comment)
word_value=X.toarray()
print(word_value.shape)
print(len(string_comment))

model_kmean=KMeans(n_clusters=3)
model_kmean.fit(word_value)

cluster_labels=model_kmean.labels_
print(cluster_labels)
columns_names=vectorizer.get_feature_names()
columns_names.append('label')#列数要一个label

k_score=sklearn.metrics.silhouette_score(word_value,cluster_labels)
print('{:*^60}'.format(k_score))
print(k_score)




#合并词向量和标签
labels=np.asarray(cluster_labels).reshape(word_value.shape[0],1)
comment_matrix=np.hstack((word_value,labels))
commen_pd=pd.DataFrame(comment_matrix,columns=columns_names)
#print(commen_pd)

#筛选label=2的数据，然后再删掉label的列，赋值给comment cluster2
comment_cluster2=commen_pd[commen_pd['label'] == 1].drop('label',axis=1)

word_importace=np.sum(comment_cluster2,axis=0)  #多少列多少个列汇总结果,series格式数据

#打印前5个最高频的分词结果
print(word_importace.sort_values(ascending=False)[0:5])


def kk_mean(x):
    s = 0
    for i in range(4):
        model_k = KMeans(n_clusters=i+2)
        model_k.fit(x)
        label = model_k.labels_
        silhouette = sklearn.metrics.silhouette_score(x, label)
        if silhouette > s:
            s = silhouette
            model = model_k
            #label_good = label
            u=i+2
    return s, u,model


k_score, n,model_kmens_good= kk_mean(word_value)
print(k_score,n)

def kk_mean2(x):
    n_clustes=[]
    s_scores=[]
    for i in range(10):
        model_k = KMeans(n_clusters=i+2)
        model_k.fit(x)
        label = model_k.labels_
        n_clustes.append(i+2)
        s_scores.append(sklearn.metrics.silhouette_score(x, label))

    return n_clustes,s_scores

n_clustes,s_scores=kk_mean2(word_value)
plt.figure()
plt.plot(n_clustes,s_scores)
plt.show()