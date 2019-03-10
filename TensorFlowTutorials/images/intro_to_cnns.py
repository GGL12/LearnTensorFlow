# 本教程演示如何训练一个简单的卷积神经网络(CNN)对MNIST数字进行分类。这个简单的网络在MNIST测试集中可以达到99%以上的准确率。由于本教程使用Keras顺序API，创建和训练我们的模型只需要几行代码。
# 导入tensorflow
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.keras import datasets, layers, models

# 下载并准备MNIST数据集
(train_images, train_labels), (test_images,
                               test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 0 1 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

# 创建卷积层
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

# 添加全连接层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(train_images, train_labels)

# 验证模型
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
