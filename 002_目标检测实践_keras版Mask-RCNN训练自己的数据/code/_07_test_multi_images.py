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
from mrcnn.visualize import random_colors, apply_mask

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
    
# 定义绘制函数，将检测结果在原图上绘制出来，并返回绘制后的图
def display_instances(image_ndarray, boxes, masks, class_ids, 
                      class_names, scores=None,
                      show_mask=True, show_bbox=True,
                      show_title=True):
    # 获取实例的数量
    N = boxes.shape[0]
    # 生成随机的颜色
    colors = random_colors(N)
    # 循环遍历每个实例
    for i in range(N):
        color = colors[i]
        color = tuple([int(255 * k) for k in color])
        y1, x1, y2, x2 = boxes[i]
        # 绘制边界框
        if show_bbox:
            thickness = 3
            leftTop_point = x1, y1
            rightDown_point = x2, y2
            cv2.rectangle(image_ndarray, leftTop_point, rightDown_point, color, thickness)
        # 绘制边界框上面的标题
        if show_title:    
            class_id = class_ids[i]
            title = '%s %.3f' %(class_names[class_id], scores[i])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.7
            title_color = (0, 0, 255)
            thickness = 2
            cv2.putText(image_ndarray, title, leftTop_point, font, font_size, title_color, thickness)
        # 绘制掩码   
        if show_mask:
            mask = masks[:, :, i]
            color = tuple([float(k / 255) for k in color])
            image_ndarray = apply_mask(image_ndarray, mask, color)
    return image_ndarray     
    
# 定义检测函数，并将检测结果用cv2库显示    
def detect_multi_images(model, imageFilePath_list, config_dict, video_aviFilePath=None):
    width = 640
    height = 640
    size = (width, height)
    windowName = 'detect_multi_images'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    display_width = 1000
    display_height = 618
    display_size = (display_width, display_height)
    cv2.resizeWindow(windowName, display_width, display_height)
    if video_aviFilePath != None:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        videoWriter = cv2.VideoWriter(video_aviFilePath, fourcc, 2, display_size)
    for imageFilePath in imageFilePath_list:
        # 根据图片文件路径读取图片
        # [:, :, ::-1]表示第3维取反，训练模型时图片的通道顺序为RGB，所以测试模型时图片的通道顺序也必须为RGB   
        image_ndarray = cv2.imread(imageFilePath)[:, :, ::-1]
        # 改变图片大小，利于模型检出实例
        resized_image_ndarray = cv2.resize(image_ndarray, size,
            interpolation=cv2.INTER_LANCZOS4)
        # 调用模型对象的detect方法，得到检测结果    
        results = model.detect([resized_image_ndarray], verbose=0)
        r = results[0]
        if r['rois'].shape[0] == 0:
            print('此图片路径没有检测出实例：%s' %imageFilePath)
        # 使用检测结果，调用display_instantces方法画图
        className_list = ['BG'] + config_dict['className_list']
        drawed_image_ndarray = display_instances(resized_image_ndarray, r['rois'], r['masks'], r['class_ids'], 
                                className_list, r['scores'])
        # 将检测结果图改变大小后显示
        drawed_image_ndarray = cv2.resize(drawed_image_ndarray, display_size,
            interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow(windowName, drawed_image_ndarray[:, :, ::-1])
        # 如果视频文件路径不为空，则当前展示画面作为一帧存入视频    
        if video_aviFilePath != None:
            videoWriter.write(drawed_image_ndarray[:, :, ::-1])  
        # 第1次按空格键可以暂停检测，第2次按空格键继续检测
        pressKey = cv2.waitKey(400)
        if ord(' ') == pressKey:
            cv2.waitKey(0)
        # 按Esc键或者q键可以退出循环
        if 27 == pressKey or ord('q') == pressKey:
            break
    videoWriter.release()
    cv2.destroyAllWindows()

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
    parser.add_argument('-c', '--config_jsonFilePath',
        type=str,
        help='模型配置json文件路径',
        default='../resources/model_config.json')
    parser.add_argument('-v', '--video_aviFilePath',
        type=str,
        help='多张图片的检测结果保存为视频文件的 路径',
        default='../resources/detect_result.avi')
    argument_namespace = parser.parse_args()
    return argument_namespace 

# 主函数    
if __name__ == '__main__':
    # 解析传入的参数
    argument_namespace = parse_args()
    model_dirPath = argument_namespace.model_dirPath.strip()
    image_dirPath = argument_namespace.image_dirPath.strip()
    image_suffix = argument_namespace.image_suffix.strip()
    config_jsonFilePath = argument_namespace.config_jsonFilePath.strip()
    video_aviFilePath = argument_namespace.video_aviFilePath.strip()
    # 获取模型配置字典，并实例化模型对象 
    config_dict = get_jsonDict(config_jsonFilePath)
    model = get_model(model_dirPath, config_dict)
    # 获取图片文件路径
    imageFilePath_list = get_filePathList(image_dirPath, image_suffix)
    assert len(imageFilePath_list), 'no image in image directory path, please check your input parameters: image_dirPath , image_suffix'
    print('对此文件夹路径的图片做检测：%s'%image_dirPath)
    # 对多张图片做检测
    detect_multi_images(model, imageFilePath_list, config_dict, video_aviFilePath)
    print('多张图片的检测结果录制为视频，保存在此路径:%s' %os.path.abspath(video_aviFilePath))