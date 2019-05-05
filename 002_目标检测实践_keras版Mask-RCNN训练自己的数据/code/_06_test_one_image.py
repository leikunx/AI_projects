# 导入常用的库
import os
import sys
import cv2
import time
import numpy as np
import json
# 工程的根目录
ROOT_DIR = os.path.abspath("../resources/")
# 导入Mask RCNN库
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import visualize

# 获取文件夹中的文件路径
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = next(os.walk(dirPath))[2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list 
    
# 根据配置json文件路径解析出字典
def get_jsonDict(config_jsonFilePath):
    with open(config_jsonFilePath, 'r', encoding='utf8') as file:
        fileContent = file.read()
    json_dict = json.loads(fileContent)
    className_list = json_dict['className_list']
    className_list = [k.strip() for k in className_list]
    className_list = sorted(className_list, reverse=False)
    json_dict['className_list'] = className_list
    return json_dict

# 模型测试配置类
class InferenceConfig(Config):
    def __init__(self, config_dict):
        super(InferenceConfig, self).__init__()
        self.NAME = config_dict['source']
        self.BACKBONE = config_dict['backbone']
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.BATCH_SIZE =1
        self.NUM_CLASSES = 1 + len(config_dict['className_list'])
        self.IMAGE_MIN_DIM = min(config_dict['image_width'], config_dict['image_height'])
        self.IMAGE_MAX_DIM = max(config_dict['image_width'], config_dict['image_height'])
        self.IMAGE_SHAPE = np.array([config_dict['image_height'], config_dict['image_width'], 3])
        self.IMAGE_META_SIZE = 15
        self.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        self.TRAIN_ROIS_PER_IMAGE = 32
        self.STEPS_PER_EPOCH = config_dict['steps_per_epoch']
        self.VALIDATION_STEPS = 20
        self.LEARNING_RATE = 0.001

# 获取模型对象    
def get_model(model_dirPath, config_dict):
    model = modellib.MaskRCNN(mode="inference", 
        config=InferenceConfig(config_dict),
        model_dir=model_dirPath)
    weights_path = model.find_last()
    print('模型加载权重文件，权重文件的路径：%s' %weights_path)
    model.load_weights(weights_path, by_name=True)
    return model

# 定义检测函数，并将检测结果用matplotlib库绘制出来    
def detect_image(model, imageFilePath, config_dict):
    image_ndarray = cv2.imread(imageFilePath)[:, :, ::-1]
    results = model.detect([image_ndarray], verbose=0)
    r = results[0]
    className_list = ['BG'] + config_dict['className_list']
    visualize.display_instances(image_ndarray, r['rois'], r['masks'], r['class_ids'], 
                            className_list, r['scores'], figsize=(8, 8))

# 解析调用代码文件时传入的参数    
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_dirPath',
        type=str,
        help='模型权重文件夹路径',
        default='../download_resources/logs')  
    parser.add_argument('-i', '--image_dirPath',
        type=str,
        help='图片文件夹路径',
        default='../download_resources/n01440764')     
    parser.add_argument('--image_suffix',
        type=str,
        help='图片文件的后缀',
        default='.jpg')
    parser.add_argument('-c', '--config',
        type=str,
        help='模型配置json文件路径',
        default='../resources/model_config.json')
    argument_namespace = parser.parse_args()
    return argument_namespace 
    
if __name__ == '__main__':
    # 解析传入的参数
    argument_namespace = parse_args()
    model_dirPath = argument_namespace.model_dirPath
    image_dirPath = argument_namespace.image_dirPath
    image_suffix = argument_namespace.image_suffix
    config_jsonFilePath = argument_namespace.config
    # 获取模型配置字典，并实例化模型对象 
    config_dict = get_jsonDict(config_jsonFilePath)
    model = get_model(model_dirPath, config_dict)
    # 获取图片文件路径
    imageFilePath_list = get_filePathList(image_dirPath, image_suffix)
    assert len(imageFilePath_list), 'no image in image directory path, please check your input parameters: image_dirPath , image_suffix'
    imageFilePath = np.random.choice(imageFilePath_list, 1)[0]
    print('对此文件路径的图片做检测：%s'%imageFilePath)
    # 对单张图片做检测
    detect_image(model, imageFilePath, config_dict)