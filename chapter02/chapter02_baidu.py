import requests
ak ='ezrEdpmVQ8EAvnUCGSLyPPiZxWXA1oqH'
url='http://api.map.baidu.com/geocoder/v2/?address=%s&output=json&ak=%s'

add='苏州中心'

res= requests.get(url %(add,ak))
add_info=res.text
print(add_info)


import json

add_json= json.loads(add_info)#将json格式数据转化成字典，包含嵌套信息
lat_lng=add_json['result']['location']
print(add_json)
print(lat_lng)
print(add_json.items())
print(type(add_json))


#*****************************************************************************************************
"""
返回xml格式，url由json改为xml即可其他无变化

因为返回值式xml格式，并不是html格式，
所有应用xlml的etree.fromstring()函数，而不是etree.HTML()函数
此外etree.XML()函数也可以

"""

url2='http://api.map.baidu.com/geocoder/v2/?address=%s&output=xml&ak=%s'

add_info2=requests.get(url=url2 % (add,ak))
add2=add_info2.text
print(add2)
import sys

import xml.etree.ElementTree as etree

root = etree.fromstring(add2)
lng=root[1][0][0].text
print(lng)

from lxml import etree

k=etree.fromstring(add2.encode('utf-8')) #设置解码
l=etree.XML(add2.encode('utf-8'))#针对 xml格式，两个函数可以使用
lat=k[1][0][1].text
lng=l[1][0][0].text
print(lat)
print(lng)
