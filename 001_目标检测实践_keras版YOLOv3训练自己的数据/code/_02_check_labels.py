# 获取文件夹中的文件路径
import os
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = next(os.walk(dirPath))[2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 删除文件前检查文件是否存在，如果不存在则报告此文件不存在，存在则报告此文件会被删除
def delete_file(filePath):
    if not os.path.exists(filePath):
        print('%s 这个文件路径不存在，请检查一下' %filePath)
    else:
        print('%s 这个路径的文件将被删除' %filePath)
    
# 此段代码删除不对应的图片文件或xml文件
def check_1(dirPath, suffix):
    # 检查标记好的文件夹是否有图片漏标，并删除漏标的图片 
    imageFilePath_list = get_filePathList(dirPath, suffix)
    allFileMarked = True
    for imageFilePath in imageFilePath_list:
        xmlFilePath = imageFilePath[:-4] + '.xml'
        if not os.path.exists(xmlFilePath):
            delete_file(imageFilePath)
            allFileMarked = False
    if allFileMarked:
        print('祝贺你!所有图片文件都被标注了。')
    # 检查有xml标注文件却没有图片的情况，删除多余的xml标注文件    
    xmlFilePath_list = get_filePathList(dirPath, '.xml')
    xmlFilePathPrefix_list = [k[:-4] for k in xmlFilePath_list]
    xmlFilePathPrefix_set = set(xmlFilePathPrefix_list)
    imageFilePath_list = get_filePathList(dirPath, suffix)
    imageFilePathPrefix_list = [k[:-4] for k in imageFilePath_list]
    imageFilePathPrefix_set = set(imageFilePathPrefix_list)
    redundant_xmlFilePathPrefix_list = list(xmlFilePathPrefix_set - imageFilePathPrefix_set)
    redundant_xmlFilePath_list = [k+'.xml' for k in redundant_xmlFilePathPrefix_list]
    for xmlFilePath in redundant_xmlFilePath_list:
        delete_file(xmlFilePath)
        
# 此段代码检查标记的xml文件中是否有物体标记类别拼写错误        
import xml.etree.ElementTree as ET
def check_2(dirPath, className_list):
    className_set = set(className_list)
    xmlFilePath_list = get_filePathList(dirPath, '.xml')
    allFileCorrect = True
    for xmlFilePath in xmlFilePath_list:
        with open(xmlFilePath) as file:
            fileContent = file.read()
        root = ET.XML(fileContent)
        object_list = root.findall('object')
        for object_item in object_list:
            name = object_item.find('name')
            className = name.text
            if className not in className_set:
                print('%s 这个xml文件中有错误的种类名称 "%s" ' %(xmlFilePath, className))
                allFileCorrect = False
    if allFileCorrect:
        print('祝贺你! 已经通过检验，所有xml文件中的标注都有正确的种类名称')
 
# 此段代码检测标记的box是否超过图片的边界
# 如果有此类型的box，则直接删除与box相关的xml文件和图片文件
from PIL import Image
def check_3(dirPath, suffix):
    xmlFilePath_list = get_filePathList(dirPath, '.xml')
    allFileCorrect = True
    for xmlFilePath in xmlFilePath_list:
        imageFilePath = xmlFilePath[:-4] + '.' + suffix.strip('.')
        image = Image.open(imageFilePath)
        width, height = image.size
        with open(xmlFilePath) as file:
            fileContent = file.read()
        root = ET.XML(fileContent)
        object_list = root.findall('object')
        for object_item in object_list:
            bndbox = object_item.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            if xmin>=1 and ymin>=1 and xmax<=width and ymax<=height:
                continue
            else:
                delete_file(xmlFilePath)
                delete_file(imageFilePath)
                allFileCorrect = False
                break
    if allFileCorrect:
        print('祝贺你! 已经通过检验，所有xml文件中的标注框都没有越界')

# 从文本文件中解析出物体种类列表className_list，要求每个种类占一行
def get_classNameList(txtFilePath):
    with open(txtFilePath, 'r', encoding='utf8') as file:
        fileContent = file.read()
        line_list = [k.strip() for k in fileContent.split('\n') if k.strip()!='']
        className_list= sorted(line_list, reverse=False)
    return className_list
    
# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str, help='文件夹路径', default='../resources/selected_images')
    parser.add_argument('-s', '--suffix', type=str, default='.jpg')
    parser.add_argument('-c', '--class_txtFilePath', type=str, default='../resources/className_list.txt')
    argument_namespace = parser.parse_args()
    return argument_namespace      
    
# 主函数    
if __name__ == '__main__':
    argument_namespace = parse_args()
    dirPath = argument_namespace.dirPath
    assert os.path.exists(dirPath), 'not exists this path: %s' %dirPath
    class_txtFilePath = argument_namespace.class_txtFilePath
    className_list = get_classNameList(class_txtFilePath)
    suffix = argument_namespace.suffix
    check_1(dirPath, suffix)
    check_2(dirPath, className_list)
    check_3(dirPath, suffix)
