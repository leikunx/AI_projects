# -*- coding: utf-8 -*-
from flask import request, Flask, send_file
import time
import os
import cv2
import pickle
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np


app = Flask(__name__)
with open('../resources/pixel_mean.pickle', 'rb') as file:
    pixel_mean = pickle.load(file)
className_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
id2name_dict = {a:b for a, b in enumerate(className_list)}
model_filePath = '../resources/cifar10_ResNet56v2_model.162.h5'
model = load_model(model_filePath)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])   
 
 
def get_imageNdarray(imageFilePath):
    image_ndarray = cv2.imread(imageFilePath)
    resized_image_ndarray = cv2.resize(image_ndarray,
        (32, 32),
        interpolation=cv2.INTER_AREA)
    return resized_image_ndarray

    
def process_imageNdarray(image_ndarray, pixel_mean):
    rgb_image_ndarray = image_ndarray[:, :, ::-1]
    image_ndarray_1 = rgb_image_ndarray / 255
    image_ndarray_2 = image_ndarray_1 - pixel_mean
    return image_ndarray_2

    
def predict_image(model, imageFilePath, id2name_dict):
    image_ndarray = get_imageNdarray(imageFilePath)
    processed_image_ndarray = process_imageNdarray(image_ndarray, pixel_mean)
    inputs = processed_image_ndarray[np.newaxis, ...]
    predict_Y = model.predict(inputs)[0]
    predict_y = np.argmax(predict_Y)
    predict_className = id2name_dict[predict_y]
    print('对此图片路径 %s 的预测结果为 %s' %(imageFilePath, predict_className))
    return predict_className

    
@app.route('/')
def index_page():
    return send_file('_04_web_page.html')
    
  
@app.route("/upload_image", methods=['POST'])
def anyname_you_like():
    startTime = time.time()
    received_file = request.files['input_image']
    imageFileName = received_file.filename
    if received_file:
        received_dirPath = '../resources/received_images'
        if not os.path.isdir(received_dirPath):
            os.makedirs(received_dirPath)
        imageFilePath = os.path.join(received_dirPath, imageFileName)
        received_file.save(imageFilePath)
        print('image file saved to %s' % imageFilePath)
        usedTime = time.time() - startTime
        print('接收图片并保存，总共耗时%.2f秒' % usedTime)
        startTime = time.time()
        predict_className = predict_image(model, imageFilePath, id2name_dict)
        usedTime = time.time() - startTime
        print('完成对接收图片的分类预测，总共耗时%.2f秒' % usedTime)
        return predict_className
    else:
        return 'failed'

        
if __name__ == "__main__":
    print('在开启服务前，先测试predict_image函数')
    imageFilePath = '../resources/images/001.jpg'
    predict_className = predict_image(model, imageFilePath, id2name_dict)
    app.run("127.0.0.1", port=5000)
    