from _06_yolo import YOLO
from PIL import Image
import os
import cv2
import time
import numpy as np

# 获取文件夹中的文件路径
def getFilePathList(dirPath, partOfFileName=''):
    allFileName_list = list(os.walk(dirPath))[0][2]
    fileName_list = [k for k in allFileName_list if partOfFileName in k]
    filePath_list = [os.path.join(dirPath, k) for k in fileName_list]
    return filePath_list

# 对多张图片做检测，并保存为
def detectMultiImages(modelFilePath, jpgFilePath_list, out_mp4FilePath=None):
    yolo_model = YOLO(model_path=modelFilePath)
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    width = 1000
    height = 618
    size = (width, height)
    cv2.resizeWindow('result', width, height)
    if out_mp4FilePath is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
        videoWriter = cv2.VideoWriter(out_mp4FilePath, fourcc, 1.7, size)
    for jpgFilePath in jpgFilePath_list:
        image = Image.open(jpgFilePath)
        out_image = yolo_model.detect_image(image)
        resized_image = out_image.resize(size, Image.ANTIALIAS)
        resized_image_ndarray = np.array(resized_image)
        #图片第1维是宽，第2维是高，第3维是RGB
        #PIL库图片第三维是RGB，cv2库图片第三维正好相反，是BGR
        cv2.imshow('result', resized_image_ndarray[...,::-1])
        time.sleep(0.3)
        if out_mp4FilePath is not None:
            videoWriter.write(resized_image_ndarray[...,::-1])
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
    yolo_model.close_session()
    cv2.destroyAllWindows()
        
# 解析运行代码文件时传入的参数
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirPath', type=str, help='directory path', default='../download_resources/n01440764')
    parser.add_argument('-w', '--width', type=int, default=416)
    parser.add_argument('-he', '--height', type=int, default=416)
    parser.add_argument('-c', '--class_txtFilePath', type=str, default='../download_resources/className_list.txt')
    parser.add_argument('-a', '--anchorFilePath', type=str, default='./model_data/yolo_anchors.txt')
    argument_namespace = parser.parse_args()
    return argument_namespace     

if __name__ == '__main__':
    modelFilePath = 'saved_model/trained_weights.h5'
    dirPath = '../n01440764'
    out_mp4FilePath = 'fish_output_2.mp4'
    jpgFilePath_list = getFilePathList(dirPath, '.JPEG')
    detectMultiImages(modelFilePath, jpgFilePath_list, out_mp4FilePath)
