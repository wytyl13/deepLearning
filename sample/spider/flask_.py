import flask, json
from flask import request
import re
import time
import requests
from time import sleep

# start the current py file as the server.
server = flask.Flask(__name__)

#获取翻译类型
#判断有没有数字
def getType(content):
    r_sz = len(re.findall('[0-9]', content))
    r_zm = len(re.findall('[a-zA-Z]', content))
    r_hz = len(re.findall('[\u4e00-\u9fa5]', content))
    r_qt = len(re.findall('^[\u4e00-\u9fa5a-zA-Z0-9]+$', content))
    return r_sz, r_zm, r_hz, r_qt


def pdsrlx(content):
    r_sz, r_zm, r_hz, r_qt = getType(content)
    if len(content) == 0:
        return('您未输入任何值')
    elif r_qt == 0:
        return('包含其他字符')
    elif r_sz > 0 and r_zm > 0 and r_hz > 0:
        return('数字、字母、汉字')
    elif r_sz > 0 and r_zm > 0:
        return('数字、字母')
    elif r_sz > 0 and r_hz > 0:
        return('数字、汉字')
    elif r_zm > 0 and r_hz > 0:
        return('字母、汉字')
    elif r_zm > 0:
        return('纯字母类型')
    elif r_hz > 0:
        return('纯汉字类型')
    elif r_sz > 0:
        return('纯数字类型')

# define the follow function as the interface                
@server.route('/', methods=['get', 'post'])
def reg():
    requestContent = request.values.get('content')
    inputType = pdsrlx(requestContent)
    data = {'i': requestContent,
                'from': 'AUTO',
                'to': 'AUTO',
                'smartresult': 'dict',
                'client': 'fanyideskweb',
                'salt': '15884048502311',
                'sign': '02c846c1bc1e4259755c183d77ef31e9',
                'ts': '1588404850231',
                'bv': 'f9c86b1fdf2f53c1fefaef343285247b',
                'doctype': 'json',
                'version': '2.1',
                'keyfrom': 'fanyi.web',
                'action': 'FY_BY_REALTlME'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
                'Referer':'http://fanyi.youdao.com/'}
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule'
    r = requests.post(url,data = data,headers = headers).json()
    r1 = r['translateResult'][0][0]['tgt']
    erroCode = r['errorCode']
    dict = {"select type": inputType, "result": r1, "status": erroCode}
    return json.dumps(dict, ensure_ascii=False)

@server.route('/12315/', methods = ['get', 'post'])
def company():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
                'Referer':'https://zwfw.samr.gov.cn/'}
    url = 'https://zwfw.samr.gov.cn/user-center/publicity/company/info/list?currentNo=1&size=702'
    r = requests.get(url,headers = headers).json()
    result = r["params"]["records"]
    errorCode = r["code"]
    for key in range(len(result)):
        uuid = result[key]["uuid"]
        url_ = 'https://zwfw.samr.gov.cn/user-center/publicity/company/info?id={}'.format(uuid)
        result_ = requests.get(url_,headers = headers).json()
        result__ = result_["params"]["companyInfo"]["content"]
        error_code = result_["code"]
        result[key]["error_code"] = error_code
        result[key]["companyInfo"] = result_
        print(key)
        print(result[key])
        sleep(5)
    return result

if __name__ == "__main__":
    server.run(port=1000, debug=True, host='0.0.0.0')