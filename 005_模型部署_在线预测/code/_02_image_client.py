import requests
import os


# 主函数 
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    while True:
        input_content = input('输入图片路径，输入-1退出，默认值(../resources/images/001.jpg): ')
        if input_content.strip() == "":
            input_content = '../resources/images/001.jpg'
        if input_content.strip() == "-1":
            break
        elif not os.path.exists(input_content.strip()):
            print('输入图片路径不正确，请重新输入')
        else:
            imageFilePath = input_content.strip()
            imageFileName = os.path.split(imageFilePath)[1]
            file_dict = {
                'file':(imageFileName,
                    open(imageFilePath,'rb'),
                    'image/jpg')}
            result = requests.post(url, files=file_dict)
            predict_className = result.text
            print('图片路径:%s 预测结果为:%s\n' %(imageFilePath, predict_className))