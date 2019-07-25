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


# 定义人脸检测类FaceDetection
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
    affine_image_3d_array = cv2.warpAffine(clipped_image_3d_array, 
        affine_matrix, clipped_image_size)
    affine_image_3d_array_1 = cv2.resize(affine_image_3d_array, new_size)
    return affine_image_3d_array


# 解析代码文件运行时传入的参数，argument中文叫做参数            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dirPath', type=str, default='../resources/person_database')
    parser.add_argument('--out_dirPath', type=str, default='../resources/face_database')
    parser.add_argument('--margin', type=int, default=8)
    argument_namespace = parser.parse_args()
    return argument_namespace    
    
        
# 主函数
if __name__ == '__main__':
    # 解析传入的参数
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dirPath
    out_dirPath = argument_namespace.out_dirPath
    margin = argument_namespace.margin
    # 实例化人脸检测类FaceDetection的对象
    face_detetor = FaceDetector()
    # 遍历输入文件夹中的每一个图片文件
    sub_dirName_list = next(os.walk(in_dirPath))[1]
    for sub_dirName in sub_dirName_list:
        in_sub_dirPath = os.path.join(in_dirPath, sub_dirName)
        out_sub_dirPath = os.path.join(out_dirPath, sub_dirName)
        fileName_list = next(os.walk(in_sub_dirPath))[2]
        for fileName in fileName_list:
            in_filePath = os.path.join(in_sub_dirPath, fileName)
            out_filePath = os.path.join(out_sub_dirPath, fileName)
            image_3d_array = np.array(Image.open(in_filePath))
            box_2d_array, point_2d_array = face_detetor.detect_image(image_3d_array, margin)
            face_quantity = len(box_2d_array)
            if face_quantity > 1 or face_quantity == 0:
                print('文件路径为%s的图片检测出的人脸数目等于%d，' %(in_filePath, face_quantity))
                continue
            box_1d_array = box_2d_array[0]
            point_1d_array = point_2d_array[0]
            affine_image_3d_array = get_affine_image_3d_array(image_3d_array, 
                box_1d_array, point_1d_array)
            affine_image = Image.fromarray(affine_image_3d_array)    
            if not os.path.isdir(out_sub_dirPath):
                os.makedirs(out_sub_dirPath)
            affine_image.save(out_filePath)
    print('人脸剪裁和对齐程序运行完成!!')        
    