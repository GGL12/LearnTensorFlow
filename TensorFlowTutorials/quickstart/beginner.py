# 首先，将TensorFlow库导入程序:
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python import keras

# 加载并准备MNIST数据集。将样本从整数转换为浮点数:
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 使用tf.keras.Sequential搭建模型。选择优化器和损失函数
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
