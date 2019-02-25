from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 加载fashion_mnist数据集
fashion_mnist = keras.datasets.fashion_mnist

'''
返回值是元组形式:
    Returns:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)
'''
# help(fashion_mnist.load_data)
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# 对应标签的具体类别
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 查看数据是否有误 训练集:(6000,28,28) 测试集:(10000,28,28)
train_images.shape
len(train_labels)
test_images.shape
len(test_labels)

# 查看样例
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 数据预处理 将RGB数据缩放0-1之间
train_images = train_images / 255.0
test_images = test_images / 255.0


# 显示钱二十五张样例，观察我们的图片是否和对应类别匹配
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''
搭建模型：
        Fc1:128个节点、rule激活函数；
        fc2:10个节点、softmax输出
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
编译模型:
    优化器:adam,
    损失函数:sparse_categorical_crossentropy,
    评价函数:accuracy
'''
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 测试集验证
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# 预测样例
predictions = model.predict(test_images)
# 预测输出为每个类别的概率值(softmax)
predictions[0]
# 获取最大概率类别索引
pred_index = np.argmax(predictions[0])

# 查看类别名
class_names[pred_index]
# 绘画
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
