import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 2
# 加载MNIST数据集
(train_images, train_labels), (test_images,
                               test_labels) = keras.datasets.mnist.load_data()

# 数据格式化(60000,28,28,1) 样本数量、图像宽度、图像高度、信道数
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# 数据格式化0-1之间


def preprocess_iamges(imgs):
    sample_img = imgs if len(imgs.shape) == 2 else imgs[0]
    assert sample_img.shape in [(28, 28, 1), (28, 28)], sample_img.shape
    return imgs / 255.0


train_images = preprocess_iamges(train_images)
test_images = preprocess_iamges(test_images)

# 查看五个样本图
plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
    plt.show

# 搭建模型 keras构建模型方法2
model = keras.Sequential()

# Conv2D:卷积操作
# MaxPooling2D:池化操作
# Dropout:去掉某些连接。避免过拟合和参数过多
# Flatten:reshape成一维数据便于接下来全连接操作
# Dense:全连接层
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 模型编译（传入优化器、损失函数、评价函数）
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型（传入训练数据，训练label，epoch）
history = model.fit(train_images, train_labels, epochs=5)
print(test_images.shape)

# 验证数据（传入测试数据、测试label）返回 loss、acc
test_loss, test_acc = model.evaluate(test_images, test_labels)
# 打印测试集正确率
print("测试集的 正确率{}".format(test_acc))