# 复写代码_keras版ResNet

## 下载资源
阅读[resources/readme.md](resources/)，并完成其中内容。
根据讲解视频学习知识点，并且完成实践内容。
学习第5个知识点：ResNet20_v2网络架构的梳理，即查看文件[第5题_ResNet20v2网络结构.txt](第5题_ResNet20v2网络结构.txt)

## 
知识点列举如下，主要是代码细节：
1. keras如何加载cifar10数据集，即下面3个代码细节：
    * 1.1 keras.dataset.cifar10代码文件中的load_data；
    * 1.2 keras.datasets.cifar代码文件中的load_batch
    * 1.3 keras.utils.data_utils代码文件中的get_file
    
2. keras如何对图像分类原有的类别ID做One-Hot编码，即下面1个代码细节：
    * 2.1 keras.utils.__init__代码文件中的to_categorical
    * 2.2 keras.utils.np_utils代码文件中的to_categorical

3. keras搭建神经网络后，如何打印神经网络的架构信息：
    * 3.1 keras.models代码文件中的Model
    * 3.2 keras.engine.training代码文件中的Model类，Model类继承keras.engine.network代码文件中的Network类
    * 3.3 Network类中的summary方法，summary方法中调用了keras.utils.layer_utils代码文件中的print_summary方法
    * 3.4 演示keras.utils.layer_utils代码文件中的print_summary方法

4. 打印神经网络的架构信息后，如何理解表格中的4个字段：
* 首先定义数据层、处理层：输入数据层和输出数据层都属于数据层，卷积层、BN层、Activation层都属于处理层
    * 4.1 Layer字段中的值，即处理层的名字。命名规则：如果不做限制，每次运行结果不同
        * 处理层的命名和模型保存、模型加载的关系
    * 4.2 Output Shape字段表示输入数据层经过处理层后，输出数据层矩阵的形状，相关知识点包括：下采样、通道扩展
    * 4.3 Param字段中的数值如何计算得来，即W矩阵或者Conv矩阵、偏置biase矩阵这2个矩阵的数值个数相加的结果
    * 4.3 Connected to字段的值，即Layer字段对应处理层的输入数据层

5. 图片分类模型ResNet20_v2的网络架构梳理，20这个值是如何计算得出的：
* 出1个选择题如下：
    * a. Conv+BN+Activation+Dense总共20层
    * b. Conv共19层+Dense1层
    * c. Conv共18层+Dense2层
    
6. ResNet_v2与ResNet_v1的区别
    * 6.1 朗读ResNet_v1论文部分内容的翻译结果
    * 6.2 朗读ResNet_v2论文部分内容的翻译结构
    * 6.3 查看网上别人对于ResNet_v1和ResNet_v2区别的见解
    
7. python的生成器与keras.preprocessing.image文件的ImageDataGenerator类的关系：
    * 7.1 举例python代码实现最简单的生成器
    * 7.2 生成器的作用
    * 7.3 快速浏览ImageDataGenerator类的__init__方法、fit方法
    
8. 在深度学习框架Keras中如何实现多GPU并行计算 
    * 8.1 理解多GPU与batch_size的关系
    * 8.2 多GPU实际加速效果演示、多GPU运算时显卡使用率分析
    * 8.3 在夏天如何做好GPU的散热工作
    
9. 理解keras.models代码文件中的Model类
    * 9.1 通过查看代码，Model类指向keras.engine.training代码文件的Model类，Model类继承于keras.engine.network代码文件的Network类
    * 9.2 查看Network类的初始化方法__init__
    
10. 演示在Google云计算平台Colab，使用TPU完成对ResNet20、ResNet56的模型训练 
    * 10.1 网络端口转发工具ngrok，使用它可以转发Colab中的tensorboard的信息
    * 10.2 在使用TPU前，获取TPU设备的信息，确保机器能够使用TPU
    * 10.3 通过5行代码完成将CPU或GPU上的模型转换为TPU上的模型

11. 实现模型训练完成后保存模型文件、权重文件，并对这2者做区分：
    * 11.1 模型文件与权重文件大小的比较
    * 11.2 入门的简单网络因为文件占用硬盘空间不大，使用保存模型文件的方法即可，做演示。
    * 11.3 实际工作应用中，保存权重文件、加载权重文件更为常用
    
12. 如何用matplotlib库展示模型测试结果   
    * 12.1 随机选取100张图片并展示识别结果。
    * 12.1 随机选取100张图片的同时，要求10个类别，每个类别取10张。
    
