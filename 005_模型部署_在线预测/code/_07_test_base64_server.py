# 导入常用的库
import os
# 导入flask库
from flask import Flask, render_template, request, jsonify
# 加载把图片文件转换为字符串的base64库
import base64


# 实例化Flask对象
server = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
server.jinja_env.auto_reload = True
server.config['TEMPLATES_AUTO_RELOAD'] = True


## 根据图片文件路径获取base64编码后内容
def get_imageBase64String(imageFilePath):
    if not os.path.exists(imageFilePath):
        image_base64_string = ''
    else:
        with open(imageFilePath, 'rb') as file:
            image_bytes = file.read()
        image_base64_bytes = base64.b64encode(image_bytes)
        image_base64_string = image_base64_bytes.decode('utf-8')  
    return image_base64_string


# 网络请求'/'的回调函数
@server.route('/')
def index():
    htmlFileName = '_07_test_base64.html'
    htmlFileContent = render_template(htmlFileName)
    print(htmlFileContent)
    return htmlFileContent
    
    
# 网络请求'/get_image'的回调函数，返回图片路径对应的图片经过base64编码后的字符串
@server.route('/get_image', methods=['POST']) 
def anyname_you_like():
    filePath = request.form['filePath']
    image_base64_string = get_imageBase64String(filePath)
    return jsonify(image_base64_string=image_base64_string)
    

# 主函数
if __name__ == '__main__':
    server.run('127.0.0.1', port=5000)