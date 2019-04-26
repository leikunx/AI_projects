import os
import random
from PIL import Image
import shutil
import argparse

# 获取文件夹中的文件路径
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = next(os.walk(dirPath))[2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list
 
# 选出一部分像素足够，即长，宽都大于指定数值的图片
def select_qualifiedImages(in_dirPath, sample_number, suffix, out_dirPath, required_width, required_height):
    imageFilePath_list = get_filePathList(in_dirPath, suffix)
    random.shuffle(imageFilePath_list)
    if not os.path.isdir(out_dirPath):
        os.makedirs(out_dirPath)
    count = 0
    for imageFilePath in imageFilePath_list:
        image = Image.open(imageFilePath)
        image_width, image_height = image.size
        if image_width >= required_width and image_height >= required_height:
            out_imageFilePath = os.path.join(new_dirPath, '%03d.jpg' %count)
            shutil.copy(imageFilePath, out_imageFilePath)
            count += 1
        if count == sample_number:
            break

# 解析运行代码文件时传入的参数            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='../download_resources/n01440764', help='输入文件夹')
    parser.add_argument('-o', '--out_dir', type=str, default='../download_resources/selected_images', help='输出文件夹')
    parser.add_argument('-n', '--number', type=int, default=200)
    parser.add_argument('-s', '--suffix', type=str, default='.JPEG')
    parser.add_argument('-w', '--width', type=int, default=416)
    parser.add_argument('-he', '--height', type=int, default=416)
    argument_namespace = parser.parse_args()
    return argument_namespace
            
# 获取数量为200的合格样本存放到selected_images文件夹中
if __name__ == "__main__":
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dir
    assert os.path.exists(in_dirPath), 'not exists this path: %s' %in_dirPath
    out_dirPath = argument_namespace.out_dir
    sample_number = argument_namespace.number
    suffix = argument_namespace.suffix
    width = argument_namespace.width
    height = argument_namespace.height
    select_qualifiedImages(in_dirPath, sample_number, suffix, out_dirPath, required_width, required_height)
    out_dirPath = os.path.abspath(out_dirPath)
    print('选出的图片文件保存到文件夹：%s' %out_dirPath)