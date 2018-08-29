import  numpy as np
import pandas as pd

"""
1 获取文本对象【针对文本，非结构化数据，推荐python自动的read  readlines readline】
"""

txt_file1 = 'G:\data_operation\python_book\chapter2\\text.txt'
t = open(txt_file1,'r')
#print(t.read())  #txt文档读成一个字符串，适合整体是一个完整的句子
print(t.readlines())#读列表形式，适用每一行是单独一个数据，例如日志
t.close()


"""
2 numpy loadtex(),load(),loadfile 读取数据【针对结构化的，纯数值的数剧】
"""
txt_numpy = 'G:\data_operation\python_book\chapter2\\numpy_data.txt'

n_txt = np.loadtxt(txt_numpy,dtype='float32',delimiter=' ')#numpy.loadtxt(),只能识别一维或者二维数组，格式默认是float32，，，字符串型txt无法读取
print(n_txt)

# 不用close？？？
#==========================================================================================================================================================
write_data_numpy = np.array([[1,2,3,4],[5,6,7,8,],[9,10,11,12]])  #定义要存储的数据

np.save('G:\data_operation\python_book\chapter2\\numpy_load_data',write_data_numpy)#保存到指定文件夹，并命名文件,但不需要.加文件类型，系统自动加.npy类型文件

numpy_load_txt = np.load('G:\data_operation\python_book\chapter2\\numpy_load_data.npy'#numpy.load()读取numpy专用格式的数据
                         )
print(numpy_load_txt)

#==========================================================================================================================================================

tofile_name='G:\data_operation\python_book\chapter2\\numpy_loadfile_data'
n_txt.tofile(tofile_name)# 导出二进制文件，这样会丢失数据形状信息
fromfile_data = np.fromfile(tofile_name,dtype='float32')#数据格式，读出和保存要一致

print(fromfile_data)

"""
3  适用pandas 的read_csv,read_fwt,read_table读取数据【结构化，探索性的，主要因为pandas读取的格式，类似数据框，仿sql式的】
"""

read_csv1 = 'G:\data_operation\python_book\chapter2\csv_data.csv'
csv_data = pd.read_csv(read_csv1,sep=',',names=['col1','col2','col3','clo4','col5'])#names 是给csv新加的列名，要和csv列一一对应，否则NaN值
print(csv_data)

read_table1 = 'G:\data_operation\python_book\chapter2\\table_data.txt'
table_data = pd.read_table(read_table1,sep=';',names=['col1','col2','col3','clo4','col5'])#实际上pandas.read_csv()是readtable的一个特例，因为其seq默认就是,号分隔符
print(table_data)

"""
4 从excel读取数据
pandas  也可以读取excel
"""

import xlrd
read_excel='G:\data_operation\python_book\chapter2\\demo.xlsx'
xlsx_1 = xlrd.open_workbook(filename=read_excel)
print('all_sheet_name %s'% xlsx_1.sheet_names())#查看所有sheet名成
print('====================================')
sheet_1=xlsx_1.sheets()[0]  #获得第一个sheet名称索引从0开始


sheet_1_name= sheet_1.name
sheet_1_cols = sheet_1.ncols
sheet_1_rows=sheet_1.nrows
print('sheet1 name %s cols %d  rows %d'%(sheet_1_name,sheet_1_cols,sheet_1_rows))

sheet_1_nrows4 = sheet_1.row_values(4)#获取第5行
sheet_1_ncls2=sheet_1.col_values(2)#获取第3列
cell23=sheet_1.row(2)[3].value#3行，4列
print('rows 4 :%s \ncol 2 %s \ncell:%s'%(sheet_1_nrows4,sheet_1_ncls2,cell23))
print('====================================')

for i in  range(sheet_1_rows):
    print(sheet_1.row_values(i))
print(sheet_1.cell(1,1))#索引号，从0开始，2行2列







