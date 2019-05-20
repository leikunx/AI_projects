import os
import random
from PIL import Image
import cv2
import argparse

# 获取文件夹中的文件路径
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = next(os.walk(dirPath))[2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list
 
# 选出一部分像素足够，即长，宽都大于指定数值的图片
def select_qualifiedImages(in_dirPath, out_dirPath, in_suffix, out_suffix, sample_number, required_width, required_height):
    imageFilePath_list = get_filePathList(in_dirPath, in_suffix)
    random.shuffle(imageFilePath_list)
    if not os.path.isdir(out_dirPath):
        os.makedirs(out_dirPath)
    count = 0
    for i, imageFilePath in enumerate(imageFilePath_list):
        image = Image.open(imageFilePath)
        image_width, image_height = image.size
        if image_width >= required_width and image_height >= required_height:
            count += 1 
            out_imageFilePath = os.path.join(out_dirPath  , '%03d%s' %(count, out_suffix))
            image_ndarray = cv2.imread(imageFilePath)
            cv2.imwrite(out_imageFilePath, image_ndarray)
        if count == sample_number:
            break

# 解析运行代码文件时传入的参数            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='../resources/n01440764', help='输入文件夹')
    parser.add_argument('-o', '--out_dir', type=str, default='../resources/selected_images', help='输出文件夹')
    parser.add_argument('--in_suffix', type=str, default='.JPEG')
    parser.add_argument('--out_suffix', type=str, default='.jpg')
    parser.add_argument('-n', '--number', type=int, default=200)
    parser.add_argument('-w', '--width', type=int, default=416)
    parser.add_argument('-he', '--height', type=int, default=416)
    argument_namespace = parser.parse_args()
    return argument_namespace
            
# 获取数量为200的合格样本存放到selected_images文件夹中
if __name__ == "__main__":
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dir.strip()
    assert os.path.exists(in_dirPath), 'not exists this path: %s' %in_dirPath
    out_dirPath = argument_namespace.out_dir.strip()
    sample_number = argument_namespace.number
    in_suffix = argument_namespace.in_suffix.strip()
    in_suffix = '.' + in_suffix.lstrip('.')
    out_suffix = argument_namespace.out_suffix.strip()
    out_suffix = '.' + out_suffix.lstrip('.')
    required_width = argument_namespace.width
    required_height = argument_namespace.height
    select_qualifiedImages(in_dirPath, out_dirPath, in_suffix, out_suffix, sample_number, required_width, required_height)
    out_dirPath = os.path.abspath(out_dirPath)
    print('选出的图片文件保存到文件夹：%s' %out_dirPath)
    imageFilePath_list = get_filePathList(out_dirPath, out_suffix)
    selectedImages_number = len(imageFilePath_list)
    print('总共选出%d张图片' %selectedImages_number)
    if selectedImages_number < sample_number:
        print('选出的样本数量没有达到%d张，需要减小传入的参数width和height!')