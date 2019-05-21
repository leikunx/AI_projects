from _06_yolo import YOLO
from PIL import Image
import os
import cv2
import time
import numpy as np

# 获取文件夹中的文件路径
def get_filePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 对多张图片做检测，并保存为
def detect_multi_images(modelFilePath, jpgFilePath_list, out_videoFilePath=None):
    yolo_model = YOLO(weightsFilePath=weightsFilePath)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    width = 1000
    height = 618
    size = (width, height)
    cv2.resizeWindow('result', width, height)
    if out_videoFilePath is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        videoWriter = cv2.VideoWriter(out_videoFilePath, fourcc, 2, size)
    for jpgFilePath in jpgFilePath_list:
        image = Image.open(jpgFilePath)
        out_image = yolo_model.detect_image(image)
        resized_image = out_image.resize(size, Image.ANTIALIAS)
        resized_image_ndarray = np.array(resized_image)
        #图片第1维是宽，第2维是高，第3维是RGB
        #PIL库图片第三维是RGB，cv2库图片第三维正好相反，是BGR
        cv2.imshow('result', resized_image_ndarray[..., ::-1])
        if out_videoFilePath is not None:
            videoWriter.write(resized_image_ndarray[..., ::-1])
        # 第1次按空格键可以暂停检测，第2次按空格键继续检测
        pressKey = cv2.waitKey(400)
        if ord(' ') == pressKey:
            cv2.waitKey(0)
        # 按Esc键或者q键可以退出循环
        if 27 == pressKey or ord('q') == pressKey:
            break
    yolo_model.close_session()
    videoWriter.release()
    cv2.destroyAllWindows()
        
# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str, help='directory path', default='../resources/n01440764')
    parser.add_argument('--image_suffix', type=str, default='.JPEG')
    parser.add_argument('-m', '--modelFilePath', type=str, default='../resources/trained_weights.h5')
    argument_namespace = parser.parse_args()
    return argument_namespace     

if __name__ == '__main__':
    argument_namespace = parse_args()
    dirPath = argument_namespace.dirPath
    image_suffix = argument_namespace.image_suffix
    modelFilePath = argument_namespace.modelFilePath
    jpgFilePath_list = get_filePathList(dirPath, image_suffix)
    out_videoFilePath = 'fish_output_2.avi'
    detect_multi_images(modelFilePath, jpgFilePath_list, out_videoFilePath)
