# 导入常用的库
import os
import numpy as np
import cv2
from PIL import Image
# 导入发起网络请求的库requests
import requests
# 导入加密图片文件为base64数据的库base64
import base64

# 根据图片文件路径获取base64编码后内容
def get_imageBase64String(imageFilePath):
    assert os.path.exists(imageFilePath), "此图片路径不存在: %" %imageFilePath
    with open(imageFilePath, 'rb') as file:
        image_bytes = file.read()
        image_base64_bytes = base64.b64encode(image_bytes)
        image_base64_string = image_base64_bytes.decode('utf-8')  
    return image_base64_string
    
     
# 使用cv2库显示图片
def cv2_display(image_ndarray):
    windowName = "object_detection_result"
    # cv2设置窗口可以变化
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, image_ndarray)
    while True:
        pressKey = cv2.waitKey()
        # 按Esc键或者q键可以关闭显示窗口
        if 27 == pressKey or ord('q') == pressKey:
            cv2.destroyAllWindows()
            break


# 根据图片文件路径获取图像数据，图像缩小后，保存到新图片文件，返回新图片文件路径
import math
def resize_image(imageFilePath, max_height=416, max_width=416):
    image_ndarray = cv2.imread(imageFilePath)
    old_height, old_width, _ = image_ndarray.shape
    if old_width > max_width or old_height > max_height:
        if old_width / old_height >= max_width / max_height:
            new_width = max_width
            resized_multiple = new_width / old_width
            new_height = math.ceil(old_height * resized_multiple)
        else:
            new_height = max_height
            resized_multiple = new_height / old_height
            new_width = math.ceil(old_width * resized_multiple)
    else:
        resized_multiple = 1
        new_width = old_width
        new_height = old_height            
    resized_image_ndarray = cv2.resize(
        image_ndarray,
        (new_width, new_height),
        )    
    image_dirPath, imageFileName = os.path.split(imageFilePath)
    resized_imageFileName = 'resized_' + imageFileName
    resized_imageFilePath = os.path.join(image_dirPath, resized_imageFileName)
    cv2.imwrite(resized_imageFilePath, resized_image_ndarray)
    return resized_imageFilePath, resized_multiple
    

# 通过种类的数量，每个种类对应的颜色，颜色变量color为rgb这3个数值组成的元祖
import colorsys    
def get_colorList(category_quantity):
    hsv_list = []
    for i in range(category_quantity):
        hue = i / category_quantity
        saturation = 1
        value = 1
        hsv = (hue, saturation, value)
        hsv_list.append(hsv)
    colorFloat_list = [colorsys.hsv_to_rgb(*k) for k in hsv_list]
    color_list = [tuple([int(x * 255) for x in k]) for k in colorFloat_list]
    return color_list
    

# 从文本文件中解析出物体种类列表category_list，要求每个种类占一行
def get_categoryList(category_txtFilePath):
    with open(category_txtFilePath, 'r', encoding='utf8') as file:
        fileContent = file.read()
    line_list = [k.strip() for k in fileContent.split('\n') if k.strip()!='']
    category_list = line_list
    return category_list 

    
# 获取绘制检测效果之后的图片
from PIL import Image, ImageDraw, ImageFont    
def get_drawedImage(image, box_list,  
                    classId_list, score_list, category_list=None,
                    show_bbox=True, show_class=True, 
                    show_score=True, show_instanceQuantity=True):
    if category_list == None:
        category_txtFilePath = '../resources/yolov3/coco.names'
        category_list = get_categoryList(category_txtFilePath)
    # 复制原图像数据，赋值给表示绘画图像数据的变量drawed_image，这样可以避免修改原图像数据
    drawed_image = image.copy()
    # 获取图像的宽、高
    image_width, image_height = image.size
    # 获取实例的数量
    box_ndarray = np.array(box_list).astype('int')
    instance_quantity = box_ndarray.shape[0]
    # 生成随机的颜色
    category_quantity = len(category_list)
    color_list = get_colorList(category_quantity)
    # 循环遍历每个实例
    for index in range(len(box_list)):
        classId = classId_list[index]
        className = category_list[classId]
        color = color_list[classId]
        x1, y1, x2, y2 = box_ndarray[index]
        # 增强绘图功能的健壮性
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)
        # 方框的左上角坐标、右上角坐标
        box_leftTop = x1, y1
        box_rightBottom = x2, y2
        # 绘制矩形，即检测出的边界框
        if show_bbox:
            drawed_image_ndarray = np.array(drawed_image)
            thickness = max(1, (image_width + image_height) // 300)
            cv2.rectangle(drawed_image_ndarray, box_leftTop, box_rightBottom, color, thickness)
            drawed_image = Image.fromarray(drawed_image_ndarray)
        # 绘制文字，文字内容为
        if show_class:  
            # 实例化图像画图对象、图像字体对象
            imageDraw = ImageDraw.Draw(drawed_image)
            fontSize = max(1, int(0.02 * image_height + 0.5))
            imageFont = ImageFont.truetype(
                font='../resources/yolov3/FiraMono-Medium.otf',
                size= fontSize
                )
            # 定义文本区域显示的内容    
            if show_score:
                score = score_list[index]
                text = '%s %.2f' %(className, score)
            else:
                text = "%s" %className
            # 根据字体种类和文字内容动态调整绘制    
            textRegion_size = imageDraw.textsize(text, imageFont)    
            if y1 < 10:
                textRegion_leftTop = (x1, y1)
                textRegion_rightBottom = (x1 + textRegion_size[0], y1 + textRegion_size[1])
            else:
                textRegion_leftTop = (x1, y1 - textRegion_size[1])
                textRegion_rightBottom = (x1 + textRegion_size[0], y1)
            # 绘制与边界框颜色相同的文字背景    
            imageDraw.rectangle(
                [textRegion_leftTop, textRegion_rightBottom],
                fill=color
                )
            # 绘制表示种类名称、置信概率的文字    
            imageDraw.text(textRegion_leftTop, text, fill=(0, 0, 0), font=imageFont)
            del imageDraw
        # 绘制文字，文字内容为图片中总共检测出多少个实例物体
        if show_instanceQuantity:
            imageDraw = ImageDraw.Draw(drawed_image)
            text = '此图片中总共检测出的物体数量：%02d' %instance_quantity
            fontSize = max(1, int(0.05 * image_height + 0.5))
            font = ImageFont.truetype('C:/Windows/Font/STLITI.TTF', fontSize, encoding='utf-8')
            textRegion_leftTop = (3, 3)
            textColor = (34, 139, 34)
            imageDraw.text(textRegion_leftTop, text, textColor, font=font)
    return drawed_image
 
    
# 主函数    
if __name__ == '__main__':
    url = "http://127.0.0.1:5000/get_detectionResult"
    while True:
        input_content = input('输入图片路径，输入-1退出，默认值(../resources/images/person.jpg): ')
        if input_content.strip() == "":
            input_content = '../resources/images/person.jpg'
        if input_content.strip() == "-1":
            break
        elif not os.path.exists(input_content.strip()):
            print('输入图片路径不正确，请重新输入')
        else:
            imageFilePath = input_content.strip()
            resized_imageFilePath, resized_multiple = resize_image(imageFilePath)
            image_base64_string = get_imageBase64String(resized_imageFilePath)
            data_dict = {'image_base64_string' : image_base64_string}
            # 调用request.post方法发起post请求，并接收返回结果
            response = requests.post(url, data=data_dict)
            # 处理返回的json格式数据，准备好传入get_drawedImageNdarray函数的参数
            responseJson_dict = response.json()
            image = Image.open(imageFilePath)
            box_ndarray = np.array(responseJson_dict['box_list']) / resized_multiple
            box_list = box_ndarray.astype('int').tolist()
            classId_list = responseJson_dict['classId_list']
            score_list = responseJson_dict['score_list']
            # 根据目标检测结果获取画框图像数据
            drawed_image = get_drawedImage(
                image,
                box_list,
                classId_list,
                score_list
                )
            rgb_image_ndarray = np.array(drawed_image)    
            bgr_image_ndarray = rgb_image_ndarray[..., ::-1]
            cv2_display(bgr_image_ndarray)
            