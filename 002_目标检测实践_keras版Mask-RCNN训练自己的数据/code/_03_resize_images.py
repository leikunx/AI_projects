# 获取文件夹中的文件路径
import os
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 修改文件夹中的单个标注文件
import json
import base64
def single_resizeLabel(old_labelFilePath, new_labelFilePath, old_size, new_size, new_imageFilePath):
    with open(old_labelFilePath) as file:
        fileContent = file.read()
    json_dict = json.loads(fileContent)
    # 获得图片宽度变化倍数，并改变标注文件中imageHeight节点的值
    # 获得图片高度变化倍数，并改变标注文件中imageWidth节点的值
    old_width, old_height = old_size
    new_width, new_height = new_size
    json_dict['imageWidth'] = new_width
    json_dict['imageHeight'] = new_height
    width_times = new_width / old_width
    height_times = new_height / old_height
    # 获取shape的列表，根据图片宽高变化比修改路点 的值
    shape_list = json_dict['shapes']
    for shape in shape_list:
        points = shape['points']
        shape['points'] = [[int(old_x * width_times), int(old_y * height_times)] for (old_x, old_y) in points]
    with open(new_imageFilePath, 'rb') as file:
        fileContent_bytes = file.read()
        imageData_bytes = base64.b64encode(fileContent_bytes)
        imageData_str = imageData_bytes.decode('utf8')
    json_dict['imageData'] = imageData_str
    new_fileContent = json.dumps(json_dict, indent=2)
    with open(new_labelFilePath, 'w', encoding='utf8') as file:
        file.write(new_fileContent)
    
    
# 修改文件夹中的若干label文件
from PIL import Image
def batch_resizeLabel(in_dirPath, out_dirPath, new_size, image_suffix, label_suffix):
    labelFilePath_list = get_filePathList(in_dirPath, label_suffix)
    for labelFilePath in labelFilePath_list:
        old_labelFilePath = labelFilePath
        old_imageFilePath = old_labelFilePath[:-len(label_suffix)] + image_suffix
        image = Image.open(old_imageFilePath)
        old_size = image.size
        labelFileName = os.path.split(old_labelFilePath)[1]
        new_labelFilePath = os.path.join(out_dirPath, labelFileName)
        new_imageFilePath = new_labelFilePath[:-len(label_suffix)] + image_suffix
        single_resizeLabel(old_labelFilePath, new_labelFilePath, old_size, new_size, new_imageFilePath)
        
#修改文件夹中的单个jpg文件
def single_resizeImage(old_imageFilePath, new_imageFilePath, new_size):
    old_image = Image.open(old_imageFilePath)
    new_image = old_image.resize(new_size, Image.ANTIALIAS)
    new_image.save(new_imageFilePath)
    
# 修改文件夹中的若干jpg文件
def batch_resizeImage(in_dirPath, out_dirPath, new_size, image_suffix, label_suffix):
    if not os.path.isdir(out_dirPath):
        os.makedirs(out_dirPath)
    imageFilePath_list = get_filePathList(in_dirPath, image_suffix)
    for imageFilePath in imageFilePath_list:
        old_imageFilePath = imageFilePath
        imageFileName = os.path.split(old_imageFilePath)[1]
        new_imageFilePath = os.path.join(out_dirPath, imageFileName)
        single_resizeImage(old_imageFilePath, new_imageFilePath, new_size)

# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dirPath',
        type=str,
        help='输入文件夹路径',
        default='../download_resources/selected_images')    
    parser.add_argument('-o', '--out_dirPath',
        type=str,
        help='输出文件夹路径',
        default='../download_resources/resized_images') 
    parser.add_argument('-w', '--width', 
        type=int,
        help='改变图片大小后图片的宽',
        default=512)
    parser.add_argument('-he', '--height',
        type=int,
        help='改变图片大小后图片的高',
        default=512)
    parser.add_argument('--image_suffix',
        type=str,
        help='图片文件的后缀名',
        default='.jpg')
    parser.add_argument('--label_suffix',
        type=str,
        help='标注文件的后缀名',
        default='.json'
        )
    argument_namespace = parser.parse_args()
    return argument_namespace  

# 主函数    
if __name__ == '__main__':
    argument_namespace = parse_args()
    in_dirPath = argument_namespace.in_dirPath.strip()
    assert os.path.exists(in_dirPath), 'not exists this path: %s' %in_dirPath
    width = argument_namespace.width
    height = argument_namespace.height
    new_size = (width, height)
    out_dirPath = argument_namespace.out_dirPath.strip()
    image_suffix = argument_namespace.image_suffix.strip()
    label_suffix = argument_namespace.label_suffix.strip()
    image_suffix = '.' + image_suffix.lstrip('.')
    label_suffix = '.' + label_suffix.lstrip('.')
    batch_resizeImage(in_dirPath, out_dirPath, new_size, image_suffix, label_suffix)
    print('所有图片文件都已经完成压缩')
    batch_resizeLabel(in_dirPath, out_dirPath, new_size, image_suffix, label_suffix)
    print('所有标注文件都已经完成压缩')