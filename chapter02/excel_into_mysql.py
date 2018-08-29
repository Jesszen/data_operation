import xlrd
import MySQLdb
import math
import datetime

excel_path = 'G:/data_operation/python_book/chapter2/order.xlsx'
conn = MySQLdb.connect(host='127.0.0.1',user='root',password='123456789', charset='utf8')
#conn.set_character_set('utf8')
mycur = conn.cursor()
mycur.execute("use sales")
#mycur.execute("drop database  if EXISTS `jinlisales`")
#mycur.execute("create database jinlisales")

def large_atr(bb):
    o = 0
    for v in bb:
        if( o <len(str(v).strip())):
            o=len(str(v).strip())
    return o



def create_table(filepath):
    """
    创建一个表：
       以文件名作为表名，以excel第一行为列名，增加一个id自增，作为主键
       返回的结果是一个sql的脚本语句
    :param filepath:
    :return:
    """
    excel_book = xlrd.open_workbook(filepath)
    sheet_names = excel_book.sheet_names()
    name_1 = filepath.split('/')[-1].split('.')[0]
    for sheet_name in sheet_names:
        sheet = excel_book.sheet_by_name(sheet_name)
        nrows=sheet.nrows #多少行
        ncols=sheet.ncols#多少列
        create_table=""
        insert_yuju=""
        for i in range(ncols):
            col_values = sheet.col_values(i)#第i列的所有数据
            h=large_atr(col_values) # 每列最大长度
            insert_yuju = insert_yuju  +  '`' + str(sheet.cell(0,i).value) +'`,'
            create_table = create_table + '`' + str(sheet.cell(0,i).value) + '` VARCHAR('+ str(h)+') null comment \"'  + str(sheet.cell(0,i).value) + '\",' + "\n\r"#换行等转义需要双引号
            # create_table = create_table + '`' + str(sheet.cell(0,i).value) + '` VARCHAR('+ str(h)+') null,' 省掉注释
        t=name_1 + sheet_name#表名后半截
        create_table_0 ='drop table if EXISTS `jess_'
        create_table_1 = ' create table if not exists `jess_'
        create_table_2 = ' id int AUTO_INCREMENT ,primary key (id)) ENGINE=INNODB   DEFAULT CHARSET=utf8;'
        create_table_full = create_table_0 + t + '`;'+create_table_1 + t + '`(' + create_table + create_table_2
        mycur.execute(create_table_full)
        for n in range(1,nrows):
            insert_sql = "insert into `jess_" + t + '`(' + insert_yuju.rstrip(',') + ')values('
            #nrows_vaule =''
            for m in range(ncols):
                if (sheet.cell(n,m).ctype == 3):
                    date_value = datetime.datetime(*xlrd.xldate_as_tuple(sheet.cell(n,m).value,datemode=0))
                    vs =datetime.date.strftime(date_value,'%Y/%d/%m')

                else:
                    vs=str(sheet.cell(n,m).value)
                insert_sql=insert_sql  +'"' + str(vs) + '",'
            insert_sql=insert_sql.rstrip(',') + ');'
            mycur.execute(insert_sql)



create_table(filepath=excel_path)
conn.commit()
mycur.close()
conn.close()



"""
def insert_data(filepath):

    将excel表中的数值，插入已经新建的表中

    :param filepath:
    :return:

    excel_book = xlrd.open_workbook(filepath)
    sheet_names = excel_book.sheet_names()
    name_1 = filepath.split('/')[-1].split('.')[0]
    for sheet_name in sheet_names:
        sheet = excel_book.sheet_by_name(sheet_name)
        ncols= sheet.ncols#多少列
        nrows=sheet.nrows
        create_table=""
        for i in range(ncols):
            create_table = create_table + '`' + str(sheet.cell(0,i).value) +'`,'
        t=name_1 + sheet_name
        insert_sql = "insert into `jess_" + t + '`(' + create_table.rstrip(',') + ')values('
        for n in range(1,nrows):
            insert_sql = "insert into `jess_" + t + '`(' + create_table.rstrip(',') + ')values('
            #nrows_vaule =''
            for m in range(ncols):
                insert_sql=insert_sql  +'"' + str(sheet.cell(n,m).value) + '",'
            insert_sql=insert_sql.rstrip(',') + ');'
            mycur.execute(insert_sql)

"""









"""
        opt_vaules=[]
        for n in (1,nrows):
            for m in range(ncols):
                k=[]
                k = k.append(sheet.cell(n,m))
            u=tuple(k)
            opt_vaules.append(u)
        #d= math.ceil(len(opt_vaules)/3000)
        #for d1 in range(0,d):
        yuju = "insert into %s (%s) VALUES (%s)"
        mycur.executemany(yuju,(t,*field,*opt_vaules))
        mycur.close()
"""



























