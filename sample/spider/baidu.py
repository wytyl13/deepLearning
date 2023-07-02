import requests
import re
import urllib
import base64
from time import sleep

def getToken():
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=WzQaXwvpPcXOKdPPCFZ7vY8t&client_secret=FjfzOWOWoqqaGWTtQKnrLhToxzGi0NHT'
    response = requests.get(host)
    if response:
        token = response.json()['access_token']
    else:
        print("error...")
        return
    return token

# encode the binary data to ascii.
def getFileContentAsBase64(path, urlencoded = False):
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content

# decode the ascii to binary data.
# def decodeBase64()

def wordDetect(path):
    token = getToken()
    imageBase64 = getFileContentAsBase64(path)
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={}'.format(token)
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }
    data = {
        'image': imageBase64
    }
    response = requests.request("POST", url, headers=headers, data=data)
    print(response.text)


if __name__ == "__main__":
    content = getFileContentAsBase64("C:/Users/weiyutao/development_code_2023-01-28/vscode/resources/word.jpg")
    print(len(content))
    wordDetect("C:/Users/weiyutao/development_code_2023-01-28/vscode/resources/word.jpg")

