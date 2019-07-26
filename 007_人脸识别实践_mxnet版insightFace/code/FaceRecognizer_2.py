# 加载常用的库
import numpy as np
import os
import cv2
from PIL import Image
import time
# 加载mxnet库
import mxnet as mx
import mxnet.ndarray as nd
# 加载用于标准化的sklearn.preprocessing
from sklearn import preprocessing
from sklearn.model_selection import KFold
# 加载用于开启多进程的库multiprocessing
from multiprocessing import Process, Manager


# 加载人脸向量化模型
def load_model(prefix = '../resources/insightFace_model/model', epoch = 0, batch_size=10):
    symbol, arg_params, auxiliary_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = symbol.get_internals()
    output_layer = all_layers['fc1_output']
    context = mx.gpu(0)
    model = mx.mod.Module(symbol=output_layer, context=context, label_names=None)
    model.bind(data_shapes=[('data', (batch_size, 3, 112, 112))])
    model.set_params(arg_params, auxiliary_params)
    return model


# 定义类 人脸向量化器FaceVectorizer
class FaceVectorizer(object):
    # 实例化对象后的初始化方法
    def __init__(self):
        self.batch_size = 1
        self.model = None
     
    # 获取输入数据
    def get_feedData(self, image_4d_array):
        image_list = []
        for image_3d_array in image_4d_array:
            height, width, _ = image_3d_array.shape
            if height!= 112 or width != 112:
                image_3d_array = cv2.resize(image_3d_array, (112, 112))
            image_list.append(image_3d_array)
        image_4d_array_1 = np.array(image_list)
        image_4d_array_2 = np.transpose(image_4d_array_1, [0, 3, 1, 2])
        image_4D_Array = nd.array(image_4d_array_2)
        image_quantity = len(image_list)
        label_1D_Array = nd.ones((image_quantity, ))
        feed_data = mx.io.DataBatch(data=(image_4D_Array,), label=(label_1D_Array,))
        return feed_data
        
    # 使用insightFace模型，把多张人脸的图像数据转换多张人脸的特征向量
    # 返回结果为2维数组
    def get_feature_2d_array(self, image_4d_array):
        if len(image_4d_array.shape) ==  3:
            image_4d_array = np.expand_dims(image_4d_array, 0)
        assert len(image_4d_array.shape) == 4, 'image_ndarray shape length is not 4'
        image_quantity = len(image_4d_array)
        if image_quantity != self.batch_size or not self.model:
            self.batch_size = image_quantity
            self.model = load_model(batch_size=self.batch_size)
        feed_data = self.get_feedData(image_4d_array)
        self.model.forward(feed_data, is_train=False)
        outputs = self.model.get_outputs()
        output_2D_Array = outputs[0]
        output_2d_array = output_2D_Array.asnumpy()
        feature_2d_array = preprocessing.normalize(output_2d_array)
        return feature_2d_array
 
 
# 计算准确率
def get_accuracy(distance_1d_array, actual_isSame_1d_array, threshold):
    predict_isSame_1d_ndarray = np.less(distance_1d_array, threshold)
    true_positive_quantity = np.sum(np.logical_and(
        predict_isSame_1d_ndarray, actual_isSame_1d_array))
    true_negetive_quantity = np.sum(np.logical_and(
        np.logical_not(predict_isSame_1d_ndarray), np.logical_not(actual_isSame_1d_array)))
    accuracy = float(true_positive_quantity+true_negetive_quantity) / len(distance_1d_array)
    return accuracy


# 获取2个特征2维数组间的距离
import math
def get_distance_1d_array(feature_2d_array_1, feature_2d_array_2, distance_method='euclidean'):
    # 计算欧几里得距离
    if distance_method == 'euclidean':
        diffValue_2d_array = np.subtract(feature_2d_array_1, feature_2d_array_2)
        distance_1d_array = np.sum(np.square(diffValue_2d_array),1)
    # 计算余弦相似度    
    elif distance_method == 'cosine':
        numerator_1d_array  = np.sum(np.multiply(feature_2d_array_1, feature_2d_array_2), axis=1)
        denominator_1d_array = np.linalg.norm(feature_2d_array_1, axis=1) * np.linalg.norm(feature_2d_array_2, axis=1)
        similarity_1d_array = numerator_1d_array / denominator_1d_array
        distance_1d_array = np.arccos(similarity_1d_array) / math.pi
    return distance_1d_array    


# 获取人脸数据库的字典，键为人脸文件路径、值为人脸512维向量
def get_database_dict(old_database_dict, new_filePath_list, face_vectorizer):
    startTime = time.time()
    new_database_dict = {}
    for new_filePath in new_filePath_list:
        if new_filePath in old_database_dict:
            feature_1d_array = old_database_dict[new_filePath].flatten()
        else:  
            image_3d_array = np.array(Image.open(new_filePath))
            image_3d_array = cv2.resize(image_3d_array, (112, 112))
            image_4d_array = np.expand_dims(image_3d_array, 0)
            feature_2d_array = face_vectorizer.get_feature_2d_array(image_4d_array)
            feature_1d_array = feature_2d_array.flatten()
        new_database_dict[new_filePath] = feature_1d_array
    usedTime = time.time() - startTime  
    print('获取人脸数据库的字典，用时%.4f秒' %usedTime)
    return new_database_dict

 
# 多进程，自动更新人脸数据库
import pickle
def update_database(face_dirPath, face_vectorizer, manager_dict, pickleFilePath='../resources/database_dict.pickle'):
    if not os.path.exists(pickleFilePath):
        old_database_dict = {}
    else:    
        with open(pickleFilePath, 'rb') as file:
            old_database_dict = pickle.load(file)
    while True:
        old_filePath_list = list(old_database_dict.keys())
        personName_list, personId_list, new_filePath_list = check_database(face_dirPath)
        if old_filePath_list != new_filePath_list:
            new_database_dict = get_database_dict(old_database_dict, new_filePath_list, face_vectorizer)
            with open(pickleFilePath, 'wb') as file:
                manager_dict['dumpFile_finished'] = False
                pickle.dump(new_database_dict, file)
                manager_dict['share_finished'] = False
                manager_dict['feature_list'] = [k.tolist() for k in new_database_dict.values()]
                manager_dict['personName_list'] = personName_list
                manager_dict['personId_list'] = personId_list
                manager_dict['dumpFile_finished'] = True
            old_database_dict = new_database_dict
        elif not manager_dict['share_finished']:
            manager_dict['feature_list'] = [k.tolist() for k in old_database_dict.values()]
            manager_dict['personName_list'] = personName_list
            manager_dict['personId_list'] = personId_list
            manager_dict['dumpFile_finished'] = True
        time.sleep(1)
    

# 检查人脸数据库，返回结果为人名列表、人ID列表、文件路径列表
def check_database(face_dirPath):
    startTime = time.time()
    fileSuffix_set = set(['jpg', 'bmp', 'png'])
    personName_list = next(os.walk(face_dirPath))[1]
    personId_list = []
    filePath_list = []
    for i, personName in enumerate(personName_list):
        dirPath = os.path.join(face_dirPath, personName)
        fileName_list = next(os.walk(dirPath))[2]
        for fileName in fileName_list:
            fileSuffix = os.path.splitext(fileName)[1][1:]
            if fileSuffix in fileSuffix_set:
                filePath = os.path.join(dirPath, fileName)
                filePath_list.append(filePath)
                personId_list.append(i)
    usedTime = time.time() - startTime  
    print('检查人脸数据库，用时%.4f秒' %usedTime)            
    return personName_list, personId_list, filePath_list
    
    
# 定义类 人脸识别器FaceRecognizer
class FaceRecognizer(object):
    # 实例化对象后的初始化方法
    def __init__(self, face_dirPath='../resources/face_database', distance_method='euclidean'):
        self.feature_dimension = 512
        self.distance_method = distance_method
        self.face_vectorizer = FaceVectorizer()
        self.manager_dict = Manager().dict()
        self.manager_dict['share_finished'] = False
        self.manager_dict['dumpFile_finished'] = False
        self.process = Process(target=update_database, args=(
            face_dirPath, self.face_vectorizer, self.manager_dict))
        self.process.start()
        self.load_database()
    
    # 把多进程中管道的共享变量赋值给 人脸识别器对象的属性
    def share_variable(self):
        if not self.manager_dict['share_finished'] and self.manager_dict['dumpFile_finished']:
            self.personName_list = self.manager_dict['personName_list']    
            self.personId_list = self.manager_dict['personId_list']
            self.database_2d_array = np.array(self.manager_dict['feature_list'])
            self.person_quantity = len(self.personName_list)
            self.face_quantity = len(self.personId_list)
            self.bincount_1d_array = np.bincount(self.personId_list)
            self.personId_1d_array = np.array(self.personId_list)
            self.manager_dict['share_finished'] = True
    
    # 加载人脸数据库
    def load_database(self):
        startTime = time.time()
        # 统计人脸数据库的人数，人脸图像数量
        while not self.manager_dict['dumpFile_finished']:
            time.sleep(1)
        self.share_variable()
        print('人脸数据库中总共有%d个人, %d个人脸图像' %(self.person_quantity, self.face_quantity))
        # 打印加载人脸图像数据花费的时间
        usedTime = time.time() - startTime
        print('加载%d张人脸图像，总共用时 %.4f秒' %(self.face_quantity, usedTime))
        # 计算得出最佳阈值
        self.make_bestThreshold()
    
    # 通过生成随机的数据集，使用10折交叉验证得出最佳阈值
    def make_bestThreshold(self):
        self.make_randomDataSet()
        startTime = time.time()
        k_fold = KFold(n_splits=10, shuffle=False)
        sample_quantity = len(self.distance_1d_array)
        index_1d_array = np.arange(sample_quantity)
        # 在200个阈值中找出最佳阈值
        bestThreshold_list = []
        for fold_index, (train_1d_array, test_1d_array) in enumerate(k_fold.split(index_1d_array)):
            train_distance_1d_array = self.distance_1d_array[train_1d_array]
            train_isSame_1d_array = self.isSame_1d_array[train_1d_array]
            test_distance_1d_array = self.distance_1d_array[test_1d_array]
            test_isSame_1d_array = self.isSame_1d_array[test_1d_array]
            accuracy_list = []
            threshold_1d_array = np.arange(0, 2.0, 0.01)
            for threshold in threshold_1d_array:
                train_accuracy = get_accuracy(train_distance_1d_array, train_isSame_1d_array, threshold)
                test_accuracy = get_accuracy(test_distance_1d_array, test_isSame_1d_array, threshold)
                # 训练集权重0.4，测试集权重0.6
                accuracy = 0.4 * train_accuracy + 0.6 * test_accuracy
                accuracy_list.append(accuracy)
            bestThreshold_index = np.argmax(accuracy_list) 
            bestThreshold = threshold_1d_array[bestThreshold_index]
            max_accuracy = np.max(accuracy_list)
            print('第%d次计算，使用判断阈值%.2f获得最大准确率%.4f' %(fold_index+1, bestThreshold, max_accuracy))
            bestThreshold_list.append(bestThreshold)
        self.bestThreshold = np.mean(bestThreshold_list)
        print('经过10折交叉验证，获得最佳判断阈值%.4f' %self.bestThreshold)
        usedTime = time.time() - startTime
        print('获取最佳判断阈值，用时%.2f秒' %(usedTime))
        
    # 生成随机的数据集
    def make_randomDataSet(self):
        startTime = time.time()
        sample_quantity = 32 * int(self.face_quantity ** 0.58)
        feature_2d_array_1 = np.empty((sample_quantity, self.feature_dimension))
        feature_2d_array_2 = np.empty((sample_quantity, self.feature_dimension))
        self.isSame_1d_array = np.empty((sample_quantity))
        # 数据集前半部分为相同人脸
        same_sample_quantity = int(sample_quantity / 2)
        selected_personId_1d_array = np.where(self.bincount_1d_array >= 2)[0]
        same_personId_1d_array = np.random.choice(selected_personId_1d_array, same_sample_quantity)
        for i, personId in enumerate(same_personId_1d_array):
            selected_index = np.where(self.personId_1d_array==personId)[0]
            index_1, index_2 = np.random.choice(selected_index, 2, replace=False)
            feature_2d_array_1[i] = self.database_2d_array[index_1]
            feature_2d_array_2[i] = self.database_2d_array[index_2]
        self.isSame_1d_array[:same_sample_quantity] = True    
        # 数据集后半部分为不同人脸
        difference_sample_quantity = int(sample_quantity / 2)
        index_1d_array = np.arange(self.face_quantity)
        for i in range(same_sample_quantity, same_sample_quantity+difference_sample_quantity):
            index_1, index_2 = np.random.choice(index_1d_array, 2)
            personId_1, personId_2 = self.personId_1d_array[[index_1, index_2]]
            # 通过循环判断，使personId_1不等于personId_2
            while personId_1 == personId_2:
                index_1, index_2 = np.random.choice(index_1d_array, 2)
                personId_1, personId_2 = self.personId_1d_array[[index_1, index_2]]
            feature_2d_array_1[i] = self.database_2d_array[index_1]
            feature_2d_array_2[i] = self.database_2d_array[index_2]    
        self.isSame_1d_array[same_sample_quantity:] = False
        # 打印随机生成数据集花费的时间
        usedTime = time.time() - startTime
        print('随机生成数据集，总共%d组人脸，用时%.2f秒' %(sample_quantity, usedTime))
        # 得出2个特征2维数组间的距离
        self.distance_1d_array = get_distance_1d_array(feature_2d_array_1, feature_2d_array_2, self.distance_method)
    
    # 获取人脸对应的人名
    def get_personName(self, imageFilePath):
        fileSuffix_set = set(['jpg', 'bmp', 'png'])
        imageFileSuffix = os.path.splitext(imageFilePath)[1][1:]
        assert imageFileSuffix in fileSuffix_set, 'get_personName function has wrong argument, check imageFilePath suffix'
        image_3d_array = np.array(Image.open(imageFilePath))
        personName = self.get_personName_1(image_3d_array)
        return personName
        
    # 获取人脸对应的人名
    def get_personName_1(self, image_3d_array):
        self.share_variable()
        if len(image_3d_array.shape) == 3:
            image_4d_array = np.expand_dims(image_3d_array, 0)
        elif len(image_3d_array.shape) == 4:
            image_4d_array = image_3d_array
        else:
            raise ValueError('传入图像数据的维度既不是3维也不是4维')
        feature_2d_array = self.face_vectorizer.get_feature_2d_array(image_4d_array)
        distance_1d_array = get_distance_1d_array(self.database_2d_array, feature_2d_array, self.distance_method)
        isSame_1d_array = np.less(distance_1d_array, self.bestThreshold)
        predict_personId_1d_array = self.personId_1d_array[isSame_1d_array]
        if len(predict_personId_1d_array) == 0:
            personName = '无效人脸'
        else:
            min_distance_index = np.argmin(distance_1d_array)
            similar_personId = self.personId_1d_array[min_distance_index]
            predict_bincount_1d_array = np.bincount(predict_personId_1d_array)
            similar_percent = predict_bincount_1d_array[similar_personId] / self.bincount_1d_array[similar_personId]
            if similar_percent >= 0.7:
                personName = self.personName_list[similar_personId]
            else:
                personName = '无效人脸'
        return personName
        