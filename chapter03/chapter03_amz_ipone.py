import requests
from lxml import etree
from urllib.parse import urlencode
import pymongo


m_host='127.0.0.1'
m_port=27017
m_user='Jess'
m_pwd='12345678wu'

m_database='amazon_iphone'
m_collection='htc'
keyword='iphone'

class mongo_j():
    def __init__(self):
        self.client=pymongo.MongoClient('mongodb://{}:{}@{}:{}'.format(m_user,m_pwd,m_host,m_port))
        self.db=self.client[m_database]
        self.collection=self.db[m_collection]
    def add(self,result):
        self.collection.insert(result)

mongo_am = mongo_j()

class amazon():
    def __init__(self):
        self.header={
            'Accept': 'text / html, * / *',
            'Accept - Encoding':'gzip, deflate',
            'Accept - Language': 'zh - CN',
            'Connection': 'keep - alive',
            'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36 Maxthon/5.2.1.6000'
        }
        self.url='https://www.amazon.cn/s/'


    def get_page_numbers(self):
        para={
            'page':1,
            'keywords':keyword
        }
        url=self.url +urlencode(para)
        req=requests.get(url,headers=self.header)
        doc=etree.HTML(req.text)
        page_maz=doc.xpath('//span[@class="pagnDisabled"]/text()')[0]
        return page_maz

    def parse_index(self,url):
        req = requests.get(url, headers=self.header)
        doc=etree.HTML(req.text)
        products=doc.xpath('//div[@id="atfResults"]//ul//li')
        for i in products:
            print(i)
            result={
                'src':i.xpath('.//a[@class="a-link-normal a-text-normal"]//img/@src'),
                'product':i.xpath('.//a[@class="a-link-normal s-access-detail-page  s-color-twister-title-link a-text-normal"]/h2/text()'),
                'price':i.xpath('.//span[@class="a-size-base a-color-price s-price a-text-bold"]/text()')}
            print(len(result['src']))
            print(result)
            if len(result['src']) != 0:
               mongo_am.add(result)

    def schedule(self):
        m=int(self.get_page_numbers())+1
        print(m)
        for n in range(1,m):
            para = {
                'page': n,
                'keywords': keyword
            }
            url = self.url + urlencode(para)
            print(url)
            self.parse_index(url)



par=amazon()
#par.schedule()
print(par.get_page_numbers())

# url='https://www.amazon.cn/s/page=1&keywords=iphone+8'
#
# par.parse_index(url)