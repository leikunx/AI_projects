# 获取文件夹中的文件路径
import os
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = next(os.walk(dirPath))[2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list
    
# 此段代码删除不对应的图片文件或xml文件
def check_1(dirPath, image_suffix, label_suffix):
    # 检查标记好的文件夹是否有图片漏标，并删除漏标的图片 
    imageFilePath_list = get_filePathList(dirPath, image_suffix)
    allFileMarked = True
    for imageFilePath in imageFilePath_list:
        labelFilePath = imageFilePath[:-len(image_suffix)] + label_suffix
        if not os.path.exists(labelFilePath):
            print('%s 此图片文件没有对应的标注文件，将被删除' %imageFilePath)
            os.remove(imageFilePath)
            allFileMarked = False
    if allFileMarked:
        print('祝贺你! 所有图片文件都被标注了。')
    # 检查有标注文件却没有图片的情况，删除多余的标注文件    
    labelFilePath_list = get_filePathList(dirPath, label_suffix)
    for labelFilePath in labelFilePath_list:
        imageFilePath = labelFilePath[:-len(label_suffix)] + image_suffix
        if  not os.path.exists(imageFilePath):
            print('%s 此标注文件没有对应的图片文件，将被删除' %labelFilePath)
            os.remove(labelFilePath)
        
# 此段代码检查标记的xml文件中是否有物体标记类别拼写错误        
import json
def check_2(dirPath, label_suffix, className_list):
    className_set = set(className_list)
    labelFilePath_list = get_filePathList(dirPath, label_suffix)
    allFileCorrect = True
    for labelFilePath in labelFilePath_list:
        with open(labelFilePath) as file:
            fileContent = file.read()
        json_dict = json.loads(fileContent)
        shapes = json_dict['shapes']
        for shape in shapes:
            className = shape['label']
            if className not in className_set:
                print('%s 这个标注文件中有错误的种类名称 "%s" ' %(labelFilePath, className))
                allFileCorrect = False
    if allFileCorrect:
        print('祝贺你! 已经通过检验，所有标注文件中的标注都有正确的种类名称')
 
# 此段代码检测标记的多边形路点是否超过图片的边界
# 如果有此类型的多边形路点，则直接删除与多边形路点相关的标注文件和图片文件
from PIL import Image
def check_3(dirPath, image_suffix, label_suffix):
    labelFilePath_list = get_filePathList(dirPath, label_suffix)
    allFileCorrect = True
    for labelFilePath in labelFilePath_list:
        imageFilePath = labelFilePath[:-len(label_suffix)] + image_suffix
        image = Image.open(imageFilePath)
        width, height = image.size
        image.close()
        with open(labelFilePath) as file:
            fileContent = file.read()
        json_dict = json.loads(fileContent)
        shapes = json_dict['shapes']
        for i, shape in enumerate(shapes, 1):
            shapeCorrect = True
            points = shape['points']
            for j, point in enumerate(points, 1):
                x, y = point
                if x>=0 and y>=0 and x<width and y<height:
                    continue
                else:
                    print('%s此标注文件存在问题，第%s个shape中的第%d个point有越界值(%d, %d)' 
                        %(labelFilePath, i, j, x, y))
                    #os.remove(labelFilePath)
                    #os.remove(imageFilePath)
                    allFileCorrect = False
                    shapeCorrect = False
                    break
            if not shapeCorrect:
                break
    if allFileCorrect:
        print('祝贺你! 已经通过检验，所有标注文件中的多边形路点都没有越界')

# 如果图片使用PIL和cv2这两个库读出的宽高不同，则删除此图片
import cv2
def check_4(dirPath, image_suffix, label_suffix):
    imageFilePath_list = get_filePathList(dirPath, image_suffix)
    for imageFilePath in imageFilePath_list:
        image = Image.open(imageFilePath)
        pil_width, pil_height = image.size
        image.close()
        image_ndarray = cv2.imread(imageFilePath)
        cv2_height, cv2_width, _ = image_ndarray.shape
        if pil_width != cv2_width or pil_height != cv2_height:
            print('%s此图片文件使用PIL和cv2这两个库读出的宽高不同，将被删除' %imageFilePath)
            os.remove(imageFilePath)
            labelFilePath = imageFilePath[:-len(image_suffix)] + label_suffix
            print('%s此标注文件将被删除' %labelFilePath)
            os.remove(labelFilePath)
    
# 从json文件中解析出物体种类列表className_list，要求每个种类占一行
def get_classNameList(config_jsonFilePath):
    with open(config_jsonFilePath, 'r', encoding='utf8') as file:
        fileContent = file.read()
    json_dict = json.loads(fileContent)
    className_list = json_dict['className_list']
    className_list = [k.strip() for k in className_list]
    className_list= sorted(className_list, reverse=False)
    return className_list
    
# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', 
        type=str, 
        help='文件夹路径',
        default='./selected_images')
    parser.add_argument('--image_suffix',
        type=str,
        help='图片文件的后缀',
        default='.jpg')
    parser.add_argument('--label_suffix',
        type=str,
        help='标注文件的后缀',
        default='.json')
    parser.add_argument('-c', '--config_jsonFilePath',
        type=str,
        help='模型配置json文件路径',
        default='../resources/model_config.json')
    argument_namespace = parser.parse_args()
    return argument_namespace      
    
# 主函数    
if __name__ == '__main__':
    argument_namespace = parse_args()
    dirPath = argument_namespace.dirPath
    assert os.path.exists(dirPath), 'not exists this path: %s' %dirPath
    config_jsonFilePath = argument_namespace.config_jsonFilePath
    className_list = get_classNameList(config_jsonFilePath)
    image_suffix = argument_namespace.image_suffix
    label_suffix = argument_namespace.label_suffix
    image_suffix = '.' + image_suffix.lstrip('.')
    label_suffix = '.' + label_suffix.lstrip('.')
    check_1(dirPath, image_suffix, label_suffix)
    check_2(dirPath, label_suffix, className_list)
    check_3(dirPath, image_suffix, label_suffix)
    check_4(dirPath, image_suffix, label_suffix)