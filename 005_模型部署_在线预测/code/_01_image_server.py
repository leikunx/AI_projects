# -*- coding: utf-8 -*-
# 导入常用的库
import time
import os
import cv2
import numpy as np
# 导入flask库的Flask类和request对象
from flask import request, Flask


app = Flask(__name__)
# 导入pickle，加载图像数据处理减去的像素均值pixel_mean
import pickle
with open('../resources/pixel_mean.pickle', 'rb') as file:
    pixel_mean = pickle.load(file)
# 定义字典id2name_dict，把种类索引转换为种类名称    
className_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
id2name_dict = {a:b for a, b in enumerate(className_list)}
# 加载图像分类模型ResNet56
from keras.models import load_model
from keras.optimizers import Adam
model_filePath = '../resources/cifar10_ResNet56v2_model.162.h5'
model = load_model(model_filePath)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])   
 

# 根据图片文件路径获取图像数据矩阵 
def get_imageNdarray(imageFilePath):
    image_ndarray = cv2.imread(imageFilePath)
    resized_image_ndarray = cv2.resize(image_ndarray,
        (32, 32),
        interpolation=cv2.INTER_AREA)
    return resized_image_ndarray


# 模型预测前必要的图像处理        
def process_imageNdarray(image_ndarray, pixel_mean):
    rgb_image_ndarray = image_ndarray[:, :, ::-1]
    image_ndarray_1 = rgb_image_ndarray / 255
    image_ndarray_2 = image_ndarray_1 - pixel_mean
    return image_ndarray_2


# 使用模型对指定图片文件路径完成图像分类，返回值为预测的种类名称        
def predict_image(model, imageFilePath, id2name_dict):
    image_ndarray = get_imageNdarray(imageFilePath)
    processed_image_ndarray = process_imageNdarray(image_ndarray, pixel_mean)
    inputs = processed_image_ndarray[np.newaxis, ...]
    predict_Y = model.predict(inputs)[0]
    predict_y = np.argmax(predict_Y)
    predict_className = id2name_dict[predict_y]
    print('对此图片路径 %s 的预测结果为 %s' %(imageFilePath, predict_className))
    return predict_className
  

# 定义回调函数，接收来自/的post请求，并返回预测结果
@app.route("/", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    received_file = request.files['file']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = '../resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('图片文件保存到此路径：%s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        predict_className = predict_image(model, imageFilePath, id2name_dict)
        usedTime = time.time() - startTime
        print('完成对接收图片的分类预测，总共耗时%.2f秒' % usedTime)
        return predict_className
    else:
        return 'failed'


# 主函数        
if __name__ == "__main__":
    print('在开启服务前，先测试predict_image函数')
    imageFilePath = '../resources/images/001.jpg'
    predict_className = predict_image(model, imageFilePath, id2name_dict)
    app.run("127.0.0.1", port=5000)
    