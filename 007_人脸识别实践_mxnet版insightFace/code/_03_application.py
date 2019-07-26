# 导入常用的库
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import cv2
import time
import sys
# 导入深度学习框架库tensorflow
import tensorflow as tf
# 导入代码文件FaceDetection_mtcnn.py
import FaceDetection_mtcnn
# 导入解析传入参数的库argparse
import argparse


# 获取显存动态增长的会话对象session
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


# 获取当前时间表示的字符串，精确到0.1毫秒    
def get_timeString():
    now_timestamp = time.time()
    now_structTime = time.localtime(now_timestamp)
    timeString_pattern = '%Y%m%d_%H%M%S'
    now_timeString_1 = time.strftime(timeString_pattern, now_structTime)
    now_timeString_2 = ('%.4f' %(now_timestamp%1))[2:]
    now_timeString = now_timeString_1 + '_' + now_timeString_2
    return now_timeString
    
    
# 获取增大的新边界框
def get_new_box(box, margin, image_size):
    image_width, image_height = image_size
    x1, y1, x2, y2 = box
    new_x1 = max(0, x1 - margin/2)
    new_y1 = max(0, y1 - margin/2)
    new_x2 = min(image_width, x2 + margin/2)
    new_y2 = min(image_height, y2 + margin/2)
    new_box = new_x1, new_y1, new_x2, new_y2
    return new_box    
    
    
# 定义人脸检测类FaceDetector
class FaceDetector(object):
    # 实例化对象后的初始化方法
    def __init__(self, model_dirPath = '../resources/mtcnn_model'):
        self.session = get_session()
        with self.session.as_default():
            self.pnet, self.rnet, self.onet = FaceDetection_mtcnn.create_mtcnn(
                self.session, model_dirPath)
    
    # 从图像中检测出人脸，并返回检测结果信息
    def detect_image(self, image_3d_array, margin=8):
        min_size = 20
        threshold_list = [0.6, 0.7, 0.7]
        factor = 0.7
        box_2d_array, point_2d_array = FaceDetection_mtcnn.detect_face(
            image_3d_array, min_size, 
            self.pnet, self.rnet, self.onet,
            threshold_list, factor)
        # 模型得出的边界框
        box_2d_array_1 = box_2d_array.reshape(-1, 5)
        # 模型预测出box的4个值、box的置信度，共5个值
        box_2d_array_2 = box_2d_array_1[:, 0:4]
        box_list = []
        image_height, image_width, _ = image_3d_array.shape
        image_size = image_width, image_height
        for box in box_2d_array_2:
            new_box = get_new_box(box, margin, image_size)
            box_list.append(new_box)
        box_2d_array_3 = np.array(box_list).astype('int')
        # 模型得出的人脸5个关键点，即10个值
        if len(point_2d_array) == 0:
            point_2d_array_1 = np.empty((0, 10))
        else:
            point_2d_array_1 = np.transpose(point_2d_array, [1, 0])    
        return box_2d_array_3, point_2d_array_1
        

# 获取仿射变换后的新图像
def get_affine_image_3d_array(original_image_3d_array, box_1d_array, point_1d_array):
    # 左眼、右眼、右嘴角这3个关键点在图像宽高的百分比
    affine_percent_1d_array = np.array([0.3333, 0.3969, 0.7867, 0.4227, 0.7, 0.7835])
    # 获取剪裁图像数据及宽高信息
    x1, y1, x2, y2 = box_1d_array
    clipped_image_3d_array = original_image_3d_array[y1:y2, x1:x2]
    clipped_image_width = x2 - x1
    clipped_image_height = y2 - y1
    clipped_image_size = np.array([clipped_image_width, clipped_image_height])
    # 左眼、右眼、右嘴角这3个关键点在剪裁图中的坐标
    old_point_2d_array = np.float32([
        [point_1d_array[0]-x1, point_1d_array[5]-y1],
        [point_1d_array[1]-x1, point_1d_array[6]-y1],
        [point_1d_array[4]-x1, point_1d_array[9]-y1]
        ])   
    # 左眼、右眼、右嘴角这3个关键点在仿射变换图中的坐标
    new_point_2d_array = (affine_percent_1d_array.reshape(-1, 2)
        * clipped_image_size).astype('float32')
    affine_matrix = cv2.getAffineTransform(old_point_2d_array, new_point_2d_array)    
    # 做仿射变换，并缩小像素至112 * 112
    new_size = (112, 112)
    clipped_image_size = (clipped_image_width, clipped_image_height)
    affine_image_3d_array = cv2.warpAffine(clipped_image_3d_array, affine_matrix, clipped_image_size)
    affine_image_3d_array_1 = cv2.resize(affine_image_3d_array, new_size)
    return affine_image_3d_array_1
    

# 获取人名列表
last_saveTime = time.time()
def get_personName_list(image_3d_array, box_2d_array, point_2d_array, distance_method='euclidean'):
    global last_saveTime
    interval = 1
    is_save_avaiable = (time.time() - last_saveTime >= interval)
    assert box_2d_array.shape[0] == point_2d_array.shape[0]
    personName_list = []
    for box_1d_array, point_1d_array in zip(box_2d_array, point_2d_array):
        affine_image_3d_array = get_affine_image_3d_array(
            image_3d_array, box_1d_array, point_1d_array)
        personName = face_recognizer.get_personName_1(affine_image_3d_array)
        personName_list.append(personName)
        # 保存人脸对齐后的图像数据为图片文件，间隔为1秒
        if is_save_avaiable:
            last_saveTime = time.time()
            dirPath = '../resources/affine_faces/' + personName
            if not os.path.isdir(dirPath):
                os.makedirs(dirPath)
            time_string = get_timeString()
            fileName = time_string + '.jpg'
            filePath = os.path.join(dirPath, fileName)
            image = Image.fromarray(affine_image_3d_array)
            image.save(filePath)
    return personName_list


# 获取绘制人脸检测、人脸识别效果后的图像
import math
def get_drawed_image_3d_array(image_3d_array, box_2d_array, personName_list, point_2d_array,
                    show_box=True, show_personName=True,
                    show_keypoints=False, show_personQuantity=True):
    assert len(box_2d_array) == len(personName_list), '请检查函数的参数'
    # 获取人脸的数量
    person_quantity = len(box_2d_array)
    image_height, image_width, _ = image_3d_array.shape
    if person_quantity != 0:
        # 循环遍历每个实例
        for index in range(person_quantity):
            # 绘制矩形，即检测出的边界框
            if show_box:
                box = box_2d_array[index]
                x1, y1, x2, y2 = box
                leftTop_point = x1, y1
                rightBottom_point = x2, y2
                color = [255, 0, 0]
                thickness = math.ceil((image_width + image_height) / 500)
                cv2.rectangle(image_3d_array, leftTop_point, rightBottom_point, color, thickness)
            # 绘制文字，文字内容为边界框上边界上方的实例种类名称
            if show_personName:  
                text = personName_list[index]
                image = Image.fromarray(image_3d_array)
                imageDraw = ImageDraw.Draw(image)
                font_filePath = 'C:/Windows/Fonts/STLITI.TTF'
                fontSize = math.ceil((image_width + image_height) / 35)
                font = ImageFont.truetype(font_filePath, fontSize, encoding='utf-8')
                textRegionLeftTop = (x1+5, y1)
                color = (255, 0, 0)
                imageDraw.text(textRegionLeftTop, text, color, font=font)
                image_3d_array = np.array(image)
            if show_keypoints:
                point_1d_array = point_2d_array[index]
                for i in range(5):
                    point = point_1d_array[i], point_1d_array[i+5]
                    radius = math.ceil((image_width + image_height) / 300)
                    color = (0, 255, 0)
                    thickness = -1
                    cv2.circle(image_3d_array, point, radius, color, thickness)
    # 绘制文字，文字内容为图片中总共检测出多少个实例物体
    if show_personQuantity:
        text = '此图片中总共检测出的人脸数量：%d' %person_quantity
        image = Image.fromarray(image_3d_array)
        imageDraw = ImageDraw.Draw(image)
        font_filePath = 'C:/Windows/Fonts/STLITI.TTF'
        fontSize = math.ceil((image_width + image_height) / 50)
        # 字体文件在Windows10系统可以找到此路径，在Ubuntu系统需要更改路径
        font = ImageFont.truetype(font_filePath, fontSize, encoding='utf-8')
        textRegionLeftTop = (20, 20)
        imageDraw.text(textRegionLeftTop, text, (34, 139, 34), font=font)
        image_3d_array = np.array(image)
    return image_3d_array               
        

# 解析代码文件运行时传入的参数，argument中文叫做参数            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_keypoints', action='store_true', default=False)
    parser.add_argument('--show_personQuantity', action='store_true', default=True)
    parser.add_argument('--video_aviFilePath', default=None)
    parser.add_argument('--margin', type=int, default=8)
    parser.add_argument('--distance_method', type=str, default='euclidean')
    argument_namespace = parser.parse_args()
    return argument_namespace          
        
        
# 主函数
if __name__ == '__main__':
    # 解析传入的参数
    argument_namespace = parse_args()
    show_keypoints = argument_namespace.show_keypoints
    show_personQuantity = argument_namespace.show_personQuantity
    video_aviFilePath = argument_namespace.video_aviFilePath
    margin = argument_namespace.margin
    distance_method = argument_namespace.distance_method
    distance_method_list = ['euclidean', 'cosine']
    assert distance_method in distance_method_list, 'distance_method must be in %s' %(
        ' or'.join(distance_method_list))
    # 实例化相机对象
    cameraIndex = 0
    camera = cv2.VideoCapture(cameraIndex)
    windowName = "faceDetection_demo"
    is_successful, image_3d_array = camera.read()
    if is_successful:
        # 导入代码文件FaceRecognizer.py中的类FaceRecognizer
        from FaceRecognizer_2 import FaceRecognizer
        print('成功加载人脸识别器类FaceRecognizer')
        face_detector = FaceDetector()
        face_recognizer = FaceRecognizer(distance_method=distance_method)
        print('成功实例化人脸检测对象、人脸识别对象')
    else:
        print('未成功调用相机，请检查相机是否已连接电脑')
        sys.exit()
    # 实例视频流写入对象
    if video_aviFilePath != None:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        image_height, image_width, _ = image_3d_array.shape
        image_size = image_width, image_height
        videoWriter = cv2.VideoWriter(video_aviFilePath, fourcc, 6, image_size)
    while is_successful:
        startTime = time.time()
        # 使用cv2库从相机读取的图像数据均为blue、green、red通道顺序，与rgb顺序相反
        is_successful, bgr_3d_array = camera.read()
        rgb_3d_array = cv2.cvtColor(bgr_3d_array, cv2.COLOR_BGR2RGB)
        box_2d_array, point_2d_array = face_detector.detect_image(rgb_3d_array, margin)
        # 根据人脸检测结果，获取人脸识别结果、绘制检测效果图
        if box_2d_array.shape[0] != 0:
            # 获取每个边界框对应的人名
            personName_list = get_personName_list(rgb_3d_array, box_2d_array, point_2d_array)
            # 绘制检测效果图s
            show_box=True
            show_personName=True
            drawed_image_3d_array = get_drawed_image_3d_array(rgb_3d_array, box_2d_array, 
                personName_list, point_2d_array, show_box, show_personName,
                show_keypoints, show_personQuantity)
            bgr_3d_array = cv2.cvtColor(drawed_image_3d_array, cv2.COLOR_RGB2BGR)
            usedTime = time.time() - startTime
            print('人脸检测和人脸识别并绘制检测效果图，用时%.4f秒' %usedTime)
        cv2.imshow(windowName, bgr_3d_array)
        # 往视频流文件写入图像数据
        if video_aviFilePath != None:
            videoWriter.write(bgr_3d_array)
        # 按Esc键或者q键退出    
        pressKey = cv2.waitKey(10)
        if 27 == pressKey or ord('q') == pressKey:
            cv2.destroyAllWindows()
            sys.exit()      