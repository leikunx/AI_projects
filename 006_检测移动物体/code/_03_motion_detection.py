# 导入常用的库
import os
import cv2
# 导入MindVision品牌的相机库mvsdk
import mvsdk


# 相机持续拍摄并保存照片
# 如果没有使用MindVision品牌相机，需要修改此函数内容
def get_capturedImage(cameraIndex):
    imageFilePath = '../resources/temp.jpg'
    capability = mvsdk.CameraGetCapability(cameraIndex)
    mvsdk.CameraSetTriggerMode(cameraIndex, 0)
    # 加载相机配置文件
    configFilePath = '../resources/camera.Config'
    assert os.path.exists(configFilePath), 'please check if exists %s'%configFilePath
    mvsdk.CameraReadParameterFromFile(cameraIndex, configFilePath)
    # 获取相机拍摄照片的预处理
    mvsdk.CameraPlay(cameraIndex)
    frameBufferSize = capability.sResolutionRange.iWidthMax * capability.sResolutionRange.iHeightMax * 3
    frameBufferAddress = mvsdk.CameraAlignMalloc(frameBufferSize, 16)
    rawData, frameHead = mvsdk.CameraGetImageBuffer(cameraIndex, 2000)
    mvsdk.CameraImageProcess(cameraIndex, rawData, frameBufferAddress, frameHead)
    mvsdk.CameraReleaseImageBuffer(cameraIndex, rawData)
    # 把文件路径转换为绝对文件路径
    imageFilePath_1 = os.path.abspath(imageFilePath)
    status = mvsdk.CameraSaveImage(cameraIndex, imageFilePath_1, frameBufferAddress, frameHead, mvsdk.FILE_JPG, 100)
    if status != mvsdk.CAMERA_STATUS_SUCCESS:
        print('ID为%d的相机拍摄并保存照片失败!!!')
        is_successful = False
    else:
        is_successful = True
    image_ndarray = cv2.imread(imageFilePath) if is_successful else None
    return image_ndarray
    
    
# 对拍摄图像进行图像处理，先转灰度图，再进行高斯滤波。
def get_processedImage(image_ndarray):
    image_ndarray_1 = cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2GRAY)
    # 用高斯滤波对图像处理，避免亮度、震动等参数微小变化影响效果
    filter_size = 15
    image_ndarray_2 = cv2.GaussianBlur(image_ndarray_1, (filter_size, filter_size), 0)
    return image_ndarray_2
    
    
# 获取表示当前时间的字符串
import time
def get_timeString():
    now_timestamp = time.time()
    now_structTime = time.localtime(now_timestamp)
    timeString_pattern = '%Y %m %d %H:%M:%S'
    now_timeString = time.strftime(timeString_pattern, now_structTime)
    return now_timeString
    
    
# 根据两张图片的不同，在第2张图上绘制不同位置的方框、日期时间    
def get_drawedDetectedImage(first_image_ndarray, second_image_ndarray):
    if second_image_ndarray is None or first_image_ndarray is None:
        return None
    first_image_ndarray_2 = get_processedImage(first_image_ndarray)
    second_image_ndarray_2 = get_processedImage(second_image_ndarray)
    # cv2.absdiff表示计算2个图像差值的绝对值
    absdiff_ndarray = cv2.absdiff(first_image_ndarray_2, second_image_ndarray_2)
    # cv2.threshold表示设定阈值做图像二值化
    threshold_ndarray = cv2.threshold(absdiff_ndarray, 25, 255, cv2.THRESH_BINARY)[1]
    # cv2.dilate表示图像膨胀
    dilate_ndarray = cv2.dilate(threshold_ndarray, None, iterations=2)
    # cv2.findContours表示找出图像中的轮廓
    contour_list = cv2.findContours(threshold_ndarray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    copy_image_ndarray = second_image_ndarray.copy()
    for contour in contour_list:
        if cv2.contourArea(contour) < 2000:
            continue
        else:
            x1, y1, w, h = cv2.boundingRect(contour)
            x2, y2 = x1 + w, y1 + h
            leftTop_coordinate = x1, y1
            rightBottom_coordinate = x2, y2
            bgr_color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(copy_image_ndarray, leftTop_coordinate, rightBottom_coordinate, bgr_color, thickness)
            time_string = get_timeString()
            text = '在时刻%s 发现运动物体! x=%d, y=%d' %(time_string, x1, y1)
            print(text)
    time_string = get_timeString()
    bgr_color = (0, 0, 255)
    thickness = 2
    cv2.putText(copy_image_ndarray, time_string, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr_color, thickness)
    return copy_image_ndarray


# 使用cv2库展示图片
def show_image(image_ndarray):  
    windowName = 'display'
    cv2.imshow(windowName, image_ndarray)


# 主函数
from sys import exit
if __name__ == '__main__':
    # 获取相机设备的信息
    device_list = mvsdk.CameraEnumerateDevice()
    if len(device_list) == 0:
        print('没有连接MindVision品牌的相机设备')
        exit()
    device_info = device_list[0]
    cameraIndex = mvsdk.CameraInit(device_info, -1, -1)
    # 开始用相机监控
    first_image_ndarray = None
    while True:
        second_image_ndarray = get_capturedImage(cameraIndex)
        drawed_image_ndarray = get_drawedDetectedImage(first_image_ndarray, second_image_ndarray)
        if drawed_image_ndarray is not None:
            show_image(drawed_image_ndarray)
        # 在展示图片后，等待1秒，接收按键
        pressKey = cv2.waitKey(1)
        # 按Esc键或者q键可以退出循环
        if 27 == pressKey or ord('q') == pressKey:
            cv2.destroyAllWindows()  
            break
        # 随着时间推移，当前帧作为下一帧的前一帧
        first_image_ndarray = second_image_ndarray
    # 关闭相机
    mvsdk.CameraUnInit(cameraIndex)
    