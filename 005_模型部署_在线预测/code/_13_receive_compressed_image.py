# 导入常用的库
import os
import time
# 导入flask库的类和方法
from flask import Flask, render_template, request, jsonify
# 导入加密解密base64数据的库base64
import base64


# 实例化Flask对象
server = Flask(__name__)
server.jinja_env.auto_reload = True
server.config['TEMPLATES_AUTO_RELOAD'] = True


# 根据图片文件路径获取base64编码后内容
def get_imageBase64String(imageFilePath):
    if not os.path.exists(imageFilePath):
        image_base64_string = ''
    else:
        with open(imageFilePath, 'rb') as file:
            image_bytes = file.read()
            image_base64_bytes = base64.b64encode(image_bytes)
            image_base64_string = image_base64_bytes.decode('utf-8')  
    return image_base64_string
    
    
# 访问首页时的回调函数
@server.route('/')
def index():
    htmlFileName = '_13_send_compressed_images.html'
    htmlFileContent = render_template(htmlFileName)
    return htmlFileContent  


# 获取请求中的参数字典 
from urllib.parse import unquote   
def get_dataDict(data_string):
    data_dict = {}
    for text in data_string.split('&'):
        key, value = text.split('=')
        value_1 = unquote(value)
        data_dict[key] = value_1
    return data_dict
    

# post请求的回调函数
from urllib.parse import unquote
@server.route('/send_compressedImage', methods=['POST'])  
def anyname_you_like():
    data_bytes = request.get_data()
    data_string = data_bytes.decode('utf-8')
    data_dict = get_dataDict(data_string)
    image_base64_string = data_dict['image_base64_string']
    image_base64_bytes = image_base64_string.encode('utf-8')
    image_bytes = base64.b64decode(image_base64_bytes)
    imageFilePath = '../resources/temp.jpeg'
    with open(imageFilePath, 'wb') as file:
        file.write(image_bytes)
    image_base64_string = get_imageBase64String(imageFilePath)
    time.sleep(2)
    return jsonify(image_base64_string=image_base64_string)    
                

if __name__ == "__main__":
    server.run('127.0.0.1', port=5000)