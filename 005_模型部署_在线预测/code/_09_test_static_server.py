# 导入常用的库
import os
# 导入flask库
from flask import Flask, render_template, request, url_for, jsonify
# 加载把图片文件转换为字符串的base64库
import base64


# 实例化Flask对象
server = Flask(
    __name__,
    static_folder="../resources/received_images"
    )
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
server.jinja_env.auto_reload = True
server.config['TEMPLATES_AUTO_RELOAD'] = True    


# 网络请求'/'的回调函数
@server.route('/')
def index():
    htmlFileName = '_09_test_static.html'
    return render_template(htmlFileName)
    
    
# 网络请求'/get_image'的回调函数，返回图片文件的url
@server.route('/get_image', methods=['POST']) 
def anmname_you_like():
    fileName = request.form['fileName']
    image_source_url = url_for('static', filename=fileName)
    print("此图片文件的url为：%s\n" %image_source_url)
    return jsonify(src=image_source_url)
    

# 主函数
if __name__ == '__main__':
    server.run('127.0.0.1', port=5000)