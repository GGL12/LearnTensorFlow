#!/usr/bin/env python
# coding=UTF-8
'''
@Description: Python Python lalalala
@Author: GGL12
@Github: https://github.com/GGL12
@LastEditors: Please set LastEditors
@Date: 2019-03-05 18:59:13
@LastEditTime: 2019-03-05 19:58:53
'''
import tensorflow.keras.backend as K
import PIL.Image as Image
import numpy as np
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pylab as plt
from __future__ import absolute_import, division, print_function

# TensorFlow Hub是一种共享经过预处理的模型组件的方法。
'''
本教程演示了：
    1：如何使用TensorFlow Hub与tf.keras。
    2：如何使用TensorFlow Hub进行图像分类。
    3：如何进行简单的迁移学习。
'''
# 导入相关的包
tf.VERSION

# 下载数据集
# 对于这个例子，我们将使用TensorFlow flowers数据集:
data_root = tf.keras.utils.get_file(
    'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)

# 将这些数据加载到模型中的最简单方法是使用 tf.keras.preprocessing.image.ImageDataGenerator:
# TensorFlow Hub的所有图像模块都需要[0,1]范围内的浮动输入。使用ImageDataGenerator的rescale参数来实现这一点。
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

# 得到的对象是一个迭代器
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

# 一个ImageNet分类器

# 下载分类器
# 使用hub.module加载一个mobilenet,，和tf.keras.layers将它包装成一个keras层。
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"


def classifier(x):
    classifier_module = hub.Module(classifier_url)
    return classifier_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))

classifier_layer = layers.Lambda(classifier, input_shape=IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()

# 重新构建数据生成器，将输出大小设置为与model期望的大小匹配。
image_data = image_generator.flow_from_directory(
    str(data_root), target_size=IMAGE_SIZE)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

# 在使用Keras时，需要手动初始化TFHub模块。
sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)

# 在单个图像运行
grace_hopper = tf.keras.utils.get_file(
    'image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SIZE)
grace_hopper
grace_hopper = np.array(grace_hopper) / 255
grace_hopper.shape

# 添加批处理维度，并将图像传递给模型
result = classifier_model.predict(grace_hopper[np.newaxis, ...])
result.shape

# 得到所预测的类别
predicted_class = np.argmax(result[0], axis=-1)
predicted_class

# 我们有预测的类ID，获取ImageNet标签，并解码预测
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# 显示照片
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name)
plt.show()

# batch运行模型
result_batch = classifier_model.predict(image_batch)
labels_batch = imagenet_labels[np.argmax(result_batch, axis=-1)]
labels_batch

# 现在看看这些预测是如何与图片相符的:
plt.figure(figsize=(10, 9))
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.imshow(image_batch[n])
    plt.title(labels_batch[n])
    plt.axis('off')
    _ = plt.suptitle("ImageNet predictions")
    plt.show()


# 学习简单的转移

# 使用tfhub可以很容易地重新训练模型的顶层来识别数据集中的类。
# 下载模型
# @param {type:"string"}
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

# 创建模块，并检查期望输入的图像大小:


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

# 确保数据生成器正在生成预期大小的图像:
image_data = image_generator.flow_from_directory(
    str(data_root), target_size=IMAGE_SIZE)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Labe batch shape: ", label_batch.shape)
    break

# 将模块包装在keras层中。
features_extractor_layer = layers.Lambda(
    feature_extractor, input_shape=IMAGE_SIZE+[3])

# 冻结特征提取层中的变量，使训练只修改新的分类器层。
features_extractor_layer.trainable = False

# 现在将hub层包装在tf.keras中。顺序模型，并添加一个新的分类层。
model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

# 初始化TFHub模块。
init = tf.global_variables_initializer()
sess.run(init)

# 测试运行单个批处理，以查看结果是否具有预期的形状。
result = model.predict(image_batch)
result.shape

# 训练模型
# 编译模型
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Callback


class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])


steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=1,
          steps_per_epoch=steps_per_epoch,
          callbacks=[batch_stats])

# 现在即使只是几个训练迭代之后，我们已经可以看到模型在任务上取得了进展。
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats.batch_acc)

# 检查预测
# 要重做之前的绘图，首先获取类名的有序列表:
label_names = sorted(image_data.class_indices.items(),
                     key=lambda pair: pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names

# 通过模型运行图像批处理，并将索引转换为类名。
result_batch = model.predict(image_batch)
labels_batch = label_names[np.argmax(result_batch, axis=-1)]
labels_batch

# 绘制结果
plt.figure(figsize=(10, 9))
for n in range(30):
    plt.subplot(6, 5, n+1)
    plt.imshow(image_batch[n])
    plt.title(labels_batch[n])
    plt.axis('off')
_ = plt.suptitle("Model predictions")

# 导出模型
export_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
export_path
