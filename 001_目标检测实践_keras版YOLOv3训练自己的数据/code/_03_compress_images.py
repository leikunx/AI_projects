# 获取文件夹中的文件路径
import os
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 修改文件夹中的单个xml文件
import xml.etree.ElementTree as ET
def single_xmlCompress(old_xmlFilePath, new_xmlFilePath, new_size):
    new_width, new_height = new_size
    with open(old_xmlFilePath) as file:
        fileContent = file.read()
    root = ET.XML(fileContent)
    # 获得图片宽度变化倍数，并改变xml文件中width节点的值
    width = root.find('size').find('width')
    old_width = int(width.text)
    width_times = new_width / old_width
    width.text = str(new_width)
    # 获得图片高度变化倍数，并改变xml文件中height节点的值
    height = root.find('size').find('height')
    old_height = int(height.text)
    height_times = new_height / old_height
    height.text = str(new_height)
    # 获取标记物体的列表，修改其中xmin,ymin,xmax,ymax这4个节点的值
    object_list = root.findall('object')
    for object_item in object_list:
        bndbox = object_item.find('bndbox')
        xmin = bndbox.find('xmin')
        xminValue = int(xmin.text)
        xmin.text = str(int(xminValue * width_times))
        ymin = bndbox.find('ymin')
        yminValue = int(ymin.text)
        ymin.text = str(int(yminValue * height_times))
        xmax = bndbox.find('xmax')
        xmaxValue = int(xmax.text)
        xmax.text = str(int(xmaxValue * width_times))
        ymax = bndbox.find('ymax')
        ymaxValue = int(ymax.text)
        ymax.text = str(int(ymaxValue * height_times))
    tree = ET.ElementTree(root)
    tree.write(new_xmlFilePath)
    
# 修改文件夹中的若干xml文件
def batch_xmlCompress(old_dirPath, new_dirPath, new_size):
    xmlFilePath_list = get_filePathList(old_dirPath, '.xml')
    for xmlFilePath in xmlFilePath_list:
        old_xmlFilePath = xmlFilePath
        xmlFileName = os.path.split(old_xmlFilePath)[1]
        new_xmlFilePath = os.path.join(new_dirPath, xmlFileName)
        single_xmlCompress(xmlFilePath, new_xmlFilePath, new_size)
        
#修改文件夹中的单个jpg文件
from PIL import Image
def single_imageCompress(old_imageFilePath, new_imageFilePath, new_size):
    old_image = Image.open(old_imageFilePath)
    new_image = old_image.resize(new_size, Image.ANTIALIAS)
    new_image.save(new_imageFilePath)
    
# 修改文件夹中的若干jpg文件
def batch_imageCompress(old_dirPath, new_dirPath, new_size, suffix):
    if not os.path.isdir(new_dirPath):
        os.makedirs(new_dirPath)
    imageFilePath_list = get_filePathList(old_dirPath, suffix)
    for imageFilePath in imageFilePath_list:
        old_imageFilePath = imageFilePath
        jpgFileName = os.path.split(old_imageFilePath)[1]
        new_imageFilePath = os.path.join(new_dirPath, jpgFileName)
        single_imageCompress(old_imageFilePath, new_imageFilePath, new_size)

# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str, help='文件夹路径', default='../download_resources/selected_images')    
    parser.add_argument('-w', '--width', type=int, default=416)
    parser.add_argument('-he', '--height', type=int, default=416)
    parser.add_argument('-s', '--suffix', type=str, default='.jpg')
    argument_namespace = parser.parse_args()
    return argument_namespace  

# 主函数    
if __name__ == '__main__':
    argument_namespace = parse_args()
    old_dirPath = argument_namespace.dirPath
    assert os.path.exists(old_dirPath), 'not exists this path: %s' %old_dirPath
    width = argument_namespace.width
    height = argument_namespace.height
    new_size = (width, height)
    new_dirPath = '../download_resources/images_%sx%s' %(str(width), str(height))
    suffix = argument_namespace.suffix
    batch_imageCompress(old_dirPath, new_dirPath, new_size, suffix)
    print('所有图片文件都已经完成压缩')
    batch_xmlCompress(old_dirPath, new_dirPath, new_size)
    print('所有xml文件都已经完成压缩')