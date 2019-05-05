# 导入常用的库
import os
import sys
import numpy as np
import json
import cv2
# 工程的根目录
ROOT_DIR = os.path.abspath("../resources/")
# 导入Mask RCNN库
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
# 定义保存文件夹，用来训练日志和训练好的权重文件
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# 定义COCO数据集预训练权重文件的路径
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# 如果此路径不存在，则报错
assert os.path.exists(COCO_MODEL_PATH), 'you need read resources/readme.md guide file and download mask_rcnn_coco.h5'
    
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

# 模型配置类
class ModelConfig(Config): 
    def __init__(self, json_dict):
        super(ModelConfig, self).__init__()
        self.NAME = json_dict['source']
        self.BACKBONE = json_dict['backbone']
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = json_dict['batch_size']
        self.NUM_CLASSES = 1 + len(json_dict['className_list'])
        self.IMAGE_MIN_DIM = min(json_dict['image_width'], json_dict['image_height'])
        self.IMAGE_MAX_DIM = max(json_dict['image_width'], json_dict['image_height'])
        self.IMAGE_SHAPE = np.array([json_dict['image_height'], json_dict['image_width'], 3])
        self.IMAGE_META_SIZE = 15
        self.RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
        self.TRAIN_ROIS_PER_IMAGE = 32
        self.STEPS_PER_EPOCH = json_dict['steps_per_epoch']
        self.VALIDATION_STEPS = 20
        self.LEARNING_RATE = 0.001

# 数据集类
class ShapesDataset(utils.Dataset):
    def __init__(self, imageFilePath_list, json_dict):
        super(ShapesDataset, self).__init__()
        self.className_list = json_dict['className_list']
        # Add classes
        for i, className in enumerate(self.className_list, 1):
            self.add_class(source=json_dict['source'], class_id=i, class_name=className)
        # Add images
        for i, imageFilePath in enumerate(imageFilePath_list):
            jsonFilePath = os.path.splitext(imageFilePath)[0] + '.json'
            self.add_image(source=json_dict['source'], image_id=i,
                           path=imageFilePath, jsonFilePath=jsonFilePath)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = cv2.imread(info['path'])[:,:,::-1]
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        jsonFilePath = info['jsonFilePath']
        with open(jsonFilePath, 'r', encoding='utf8') as file:
            fileContent = file.read()
        json_dict = json.loads(fileContent)
        shapes = json_dict['shapes']
        shape_number = len(shapes)
        image_ndarray = cv2.imread(info['path'])
        height, width, _ = image_ndarray.shape
        mask_ndarray = np.zeros((height, width, shape_number), np.uint8)
        label_list = []
        for i, shape in enumerate(shapes):
            self.draw_mask(mask_ndarray, i, shape, label_list)
        # 把物体种类名称转换为物体种类Id
        classId_list = [self.class_names.index(k) for k in label_list]
        return mask_ndarray.astype(np.bool), np.array(classId_list)
    
    def draw_mask(self, mask_ndarray, i, shape, label_list):
        if 'shape_type' not in shape:
            shapeType = 'polygon'
        else:
            shapeType = shape['shape_type']
        shapeType_list = ['polygon', 'rectangle', 'circle', 'line', 'point', 'linestrip']
        label = shape['label']
        label_list.append(label)
        point_list = shape['points']
        if shapeType not in shapeType_list:
            print('shape have wrong shape_type! please check json file.')
            return 
        elif shapeType == 'polygon' or shapeType == 'line' or shapeType == 'linestrip':
            point_ndarray = np.array(point_list)[np.newaxis, ...]
            mask_ndarray[:, :, i:i+1] = self.draw_fillPoly(mask_ndarray[:, :, i:i+1].copy(), point_ndarray, 128)
        elif shapeType == 'rectangle':
            leftTop_point, rightDown_point = point_list
            mask_ndarray[:, :, i:i+1] = self.draw_rectangle(mask_ndarray[:, :, i:i+1].copy(), leftTop_point, rightDown_point, 128, -1)
        elif shapeType == 'circle':    
            center_point, contour_point = point_list
            center_x ,center_y = center_point
            contour_x, contour_y = contour_point
            radius = int(((center_x - contour_x) ** 2 + (center_y - contour_y) ** 2) ** 0.5)
            mask_ndarray[:, :, i:i+1] = self.draw_circle(mask_ndarray[:, :, i].copy(), tuple(center_point), radius, 128, -1)
        elif shape_type == 'point':
            center_point = point_list[0]
            radius = 3
            mask_ndarray[:, :, i:i+1] = self.draw_circle(mask_ndarray[:, :, i].copy(), tuple(center_point), radius, 128, -1)
    
    def draw_fillPoly(self, mask, point_ndarray, color):
        cv2.fillPoly(mask, point_ndarray, 128)
        return mask
    
    def draw_rectangle(self, mask, leftTop_point, rightDown_point, color, thickness):
        cv2.rectangle(mask, leftTop_point, rightDown_point, color, thickness)
        return mask
    
    def draw_circle(self, mask, center_point, radius, color, thickness):
        cv2.circle(mask, center_point, radius, color, thickness)
        return mask

# 随机划分为3个数据集：训练集，验证集，测试集
def get_train_val_test(dirPath, image_suffix, json_dict):
    imageFilePath_list = get_filePathList(dirPath, image_suffix)
    np.random.seed(2019)
    np.random.shuffle(imageFilePath_list)
    val_percent, test_percent = 0.05, 0.10
    sample_number = len(imageFilePath_list)
    val_number = max(int(sample_number * val_percent), 1)
    test_number = max(int(sample_number * test_percent), 1)
    train_number = sample_number - val_number - test_number
    print('训练集样本数量:%d，验证集样本数量:%d，测试集样本数量:%d' %(train_number, val_number, test_number))
    train_imageFilePath_list = imageFilePath_list[-train_number : ]
    val_imageFilePath_list = imageFilePath_list[: val_number]
    test_imageFilePath_list = imageFilePath_list[val_number : val_number+test_number]
    train_dataset = ShapesDataset(train_imageFilePath_list, json_dict)
    val_dataset = ShapesDataset(val_imageFilePath_list, json_dict)
    test_dataset = ShapesDataset(test_imageFilePath_list, json_dict)
    train_dataset.prepare()
    val_dataset.prepare()
    test_dataset.prepare()
    return train_dataset, val_dataset, test_dataset

# 解析调用代码文件时传入的参数    
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dirPath',
        type=str,
        help='输入图片文件夹路径',
        default='../download_resources/resized_images_640x640')  
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
    # 解析出调用代码文件时传入的参数
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dirPath
    image_suffix = argument_namespace.image_suffix
    config_jsonFilePath = argument_namespace.config
    # 根据配置文件和配置类ModelConfig实例化模型配置对象
    json_dict = get_jsonDict(config_jsonFilePath)
    modelConfig = ModelConfig(json_dict)
    modelConfig.display()
    # 随机划分训练集、验证集、测试集，在模型训练中，测试集没有被用到
    train_dataset, val_dataset, _ = get_train_val_test(in_dirPath, image_suffix, json_dict)
    # 实例化模型对象
    model = modellib.MaskRCNN(mode="training",
        config=modelConfig,
        model_dir=MODEL_DIR)
    # 模型加载COCO数据集的预训练权重
    model.load_weights(COCO_MODEL_PATH,
        by_name=True,
        exclude=["mrcnn_class_logits", 
            "mrcnn_bbox_fc", 
            "mrcnn_bbox", 
            "mrcnn_mask"])
    # 模型fine-tune第1步：训练卷积核以外的层  
    print('模型fine-tune第1步：训练卷积核以外的层')
    model.train(train_dataset, 
        val_dataset, 
        learning_rate=modelConfig.LEARNING_RATE, 
        epochs=1, 
        layers='heads')
    # 模型fine-tune第2步：训练全部层         
    print('模型fine-tune第2步：训练全部层')
    model.train(train_dataset,
        val_dataset, 
        learning_rate=modelConfig.LEARNING_RATE / 10,
        epochs=20, 
        layers="all")            
    