# coding: utf-8

from __future__ import print_function
import keras
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model
import glob
import numpy as np

import random
import os
import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
# # 将显存容量调到只会使用30%
config.gpu_options.per_process_gpu_memory_fraction = 0.3
# # 使用设置好的配置
set_session(tf.Session(config=config))

# 超参数
width = 224     # 图片宽度
height = 224    # 图片高度
channel = 3     # 图片通道
train_ratio = 0.8   # 训练比例
crop_fix_size = (220, 220)      # 裁剪图片大小
crop_ratio = 0.5        # 裁剪比例
lr = 0.1        # 学习率
batch = 10      # batch_size数量  几张照片
epoch = 5   # 总训练次数
patienceEpoch = 3       # 
size = width, height


# 文件读取
def CountFiles(path):
    files = []
    labels = []
    subdirs = os.listdir(path)
    print(subdirs)
    subdirs.sort()
    for index in range(len(subdirs)):
        subdir = os.path.join(path, subdirs[index])
        sys.stdout.flush()    # 刷新缓存区
        print("label --> dir : {} --> {}".format(index, subdir))
        for image_path in glob.glob("{}/*.jpg".format(subdir)):    
        # glob.glob()匹配所有的符合条件的文件，并将其以list的形式返回
            files.append(image_path)
            labels.append(index)

    return files, labels, len(subdirs)


# 图片随机排序再进行zip
files, labels, clazz = CountFiles(r"AI/flower/flower_photos")     # 图片路径,图片label(数字代表即为label),类别个数
c = list(zip(files, labels))
random.shuffle(c)    # 将序列的所有元素随机排序
files, labels = zip(*c)    # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
# labels进行one-hot编码
labels = np.array(labels)
labels = keras.utils.to_categorical(labels, clazz)  # 将整型的类别标签转为onehot编码
# 训练图片总数
train_num = int(train_ratio * len(files))

# 划分数据集
train_x, train_y = files[:train_num], labels[:train_num]
test_x, test_y = files[train_num:], labels[train_num:]

# 图片处理方式 镜像 旋转 位移 白噪声
def LoadImage(image_path):
    img = cv2.imread(image_path)	# 读取图片
    # resize图片缩放  img输入图片  disze输出图片尺寸  interpolation插入方式  INTER_AREA使用像素区域关系进行重采样
    img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32")    # 转换数据类型
    img /= 255.
    # 图片预处理
    # 裁剪
    if random.random() < crop_ratio:
        im1 = img.copy()
        # 将图像以设定的阈值进行剪裁
        x = random.randint(0, img.shape[0] - crop_fix_size[0] - 1)
        y = random.randint(0, img.shape[1] - crop_fix_size[1] - 1)
        im1 = im1[x:x+crop_fix_size[0], y:y+crop_fix_size[1], :]     # 图像裁剪
        im1 = cv2.resize(im1, dsize = size, interpolation = cv2.INTER_AREA)
        img = im1
    # 镜像
    if random.random() < crop_ratio:
        im1 = img.copy()

        if random.random()<0.3:
            im1 = cv2.flip(im1, 1, dst=None)  # 水平镜像
        elif random.random()<0.6:
            im1 = cv2.flip(im1, 0, dst=None)  # 垂直镜像
        elif random.random() < 0.1:
            im1 = cv2.flip(im1, -1, dst=None)  # 对角镜像
        # 将图片缩放回原来的尺寸
        im1 = cv2.resize(im1, dsize=size, interpolation=cv2.INTER_AREA)
        img = im1
# 旋转
#    if random.random() < crop_ratio:
#        im1 = img.copy()
#        height = im1.shape[0]
#        width = im1.shape[1]
#        deep = im1.shape[2]
#        rotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), 45, 0.7)
#        dst = cv2.warpAffine(im1, rotate, (height, width))
#        
#        img = dst

    return np.array(img)


def LoadImageGen(files_r, labels_r, batch=32):
    start = 0
    while start < len(files_r):
        stop = start + batch
        if stop > len(files_r):
            stop = len(files_r)
        imgs = []
        lbs = []
        for i in range(start, stop):
            imgs.append(LoadImage(files_r[i]))
            lbs.append(labels_r[i])
        yield np.array(imgs), np.array(lbs)
        if start + batch < len(files_r):
            start += batch
        else:
            c = list(zip(files_r, labels_r))
            random.shuffle(c)
            files_r, labels_r = zip(*c)
            start = 0


# 模型InceptionV3
# 参数 weight 权重  include_top是否包含完全连接网络顶部的网络层  pooling池化  input_shape指定形状元组  classes图片分类的类别数
model = InceptionV3(weights=None, include_top=True, pooling='avg', input_shape=(width, height, channel), classes=clazz)
# 对model进行编译  loss损失  optimizer优化器  metrics指标为准确率
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=lr, decay=0.),  # 可选SGD  Adagrad  Adam
              metrics=['accuracy'])
# 每个epoch运行的步骤数
steps_per_epoch = int(len(train_x)/batch) - 1,
validation_steps = int(len(test_x) / batch) - 1,
# tensorboard跟踪训练模型参数  训练过程可视化
tensorBoardCallBack = keras.callbacks.TensorBoard(log_dir="./tensorboard",  # 保存分析日志文件的文件名
                                                  histogram_freq=0, write_graph=True,  # 0为直方图不会被计算  设置True为可视化图像
                                                  write_grads=True, batch_size=batch,  # 可视化梯度值直方图  传入神经元网络输入批的大小
                                                  write_images=True)  # 是否将模型权重以图片可视化，若True则日志文件会很大
# 模型权重文件
modelCheckpoint = ModelCheckpoint(filepath="./model_{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5", # 保存模型到filepath
                                  verbose=0, save_best_only=False)  # verbose设置0为不输出模型保存信息
                                  # save_best_only 设置为True则只保存验证集上性能最好的模型
print("class num : {},  train num: {}, test num: {}, batch : {}, train steps: {}, validation steps : {}".format(clazz, len(train_x), len(test_x), batch, steps_per_epoch, validation_steps))
model.fit_generator(
    LoadImageGen(train_x, train_y, batch=batch),    #生成器函数
    steps_per_epoch=int(len(train_x)/batch),    # 当生成器返回steps_per_epoch次数据时计一次epoch结束，执行下一个epoch
    epochs=epoch,    # 数据迭代的轮数
    verbose=1,    # 日志显示 0不输出 1进度条 2每个epoch输出一行
    validation_data=LoadImageGen(test_x, test_y, batch=batch),     # 生成验证集的生成器
    validation_steps=int(len(test_x) / batch),     # 当validation_data为生成器时，本参数指定验证集的生成器返回次数
    callbacks=[(EarlyStopping(monitor='val_acc', patience=patienceEpoch)), tensorBoardCallBack, modelCheckpoint],    # 训练过程中会调用lsit中的回调函数
    # EarlyStopping当被监测的数量不再提升则停止训练（被监视的值，当EarlyStopping被激活的时候经过patienceEpoch后停止）
)
# 评估训练模型
score = model.evaluate_generator(
    LoadImageGen(test_x, test_y, batch=batch),    #生成器函数
    steps=int(len(test_x) / batch))    # 测试数据
print("Test loss: {}, Test accuracy: {}".format(score[0], score[1]))

