# 导入代码文件_06_yolov3.py的类Detector
from _06_yolov3 import Detector
# 导入常用的库
import numpy as np
import time
import os
from PIL import Image
# 导入flask库
from flask import Flask, render_template, request, jsonify
# 加载把图片文件转换为字符串的base64库
import base64
from yolo3.utils import letterbox_image


# 实例化Flask服务对象，赋值给变量server
server = Flask(__name__)
# 设置开启web服务后，如果更新html文件，可以使更新立即生效
server.jinja_env.auto_reload = True
server.config['TEMPLATES_AUTO_RELOAD'] = True


# 实例化检测器对象
detector = Detector(
    weights_h5FilePath='../resources/yolov3/yolov3_weights.h5',
    anchor_txtFilePath='../resources/yolov3/yolov3_anchors.txt',
    category_txtFilePath='../resources/yolov3/coco.names'
    )
    
    
# 获取当前时间表示的字符串的小数部分，精确到0.1毫秒
def get_secondFloat(timestamp):
    secondFloat = ('%.4f' %(timestamp%1))[1:]
    return secondFloat

 
# 获取当前时间表示的字符串，精确到0.1毫秒    
def get_timeStr():
    now_timestamp = time.time()
    now_structTime = time.localtime(now_timestamp)
    timeStr_pattern = '%Y%m%d_%H%M%S'
    now_timeStr_1 = time.strftime(timeStr_pattern, now_structTime)
    now_timeStr_2 = get_secondFloat(now_timestamp)
    now_timeStr = now_timeStr_1 + now_timeStr_2
    return now_timeStr
    

# 获取使用YOLOv3算法做目标检测的结果
def get_detectResult(image):
    startTime = time.time()
    boxed_image = letterbox_image(image, (416, 416))
    image_data = np.array(boxed_image).astype('float') / 255
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # 模型网络结构运算
    box_ndarray, classId_ndarray, score_ndarray = detector.session.run(
        [detector.boxes, detector.classes, detector.scores],
        feed_dict={
            detector.yolo_model.input: image_data,
            detector.input_image_size: [image.size[1], image.size[0]],
            }
        )
    box_ndarray = box_ndarray[:, [1,0,3,2]]
    return box_ndarray, classId_ndarray, score_ndarray
     

# 获取请求中的参数字典    
def get_dataDict(data):
    data_dict = {}
    for text in data.split('&'):
        key, value = text.split('=')
        data_dict[key] = value
    return data_dict


# '/'的回调函数
@server.route('/')
def index():
    htmlFileName = '_11_yolov3.html'
    return render_template(htmlFileName)    
    

# '/get_detectedResult'的回调函数
from urllib.parse import unquote
@server.route('/get_detectionResult', methods=['POST']) 
def anmname_you_like():
    startTime = time.time()
    post_data = request.get_data()
    data = post_data.decode('utf-8')
    data_dict = get_dataDict(data)
    if 'image_data' in data_dict:
        # 保存接收的图片到指定文件夹
        received_dirPath = '../resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        timeStr = get_timeStr()
        imageFileName = timeStr + '.jpg'
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        try:
            image_base64string = unquote(data_dict['image_data'])
            image_base64bytes = image_base64string.encode('utf-8')
            image_bytes = base64.b64decode(image_base64bytes)
            with open(imageFilePath, 'wb') as file:
                file.write(image_bytes)
            print('接收图片文件保存到此路径：%s' %imageFilePath)
            usedTime = time.time() - startTime
            print('接收图片并保存，总共耗时%.2f秒\n' %usedTime)    
            # 通过图片路径读取图像数据，并对图像数据做目标检测
            image = Image.open(imageFilePath)    
            box_ndarray, classId_ndarray, score_ndarray = get_detectResult(image)
            # 把目标检测结果转化为json格式的字符串
            json_dict = {
                'box_list' : box_ndarray.astype('int').tolist(),
                'classId_list' : classId_ndarray.tolist(),
                'score_list' : score_ndarray.tolist()
                }
            return jsonify(**json_dict)    
        except Exception as e:
            print(e)
        

if __name__ == '__main__':
    server.run('127.0.0.1', port=5000)
