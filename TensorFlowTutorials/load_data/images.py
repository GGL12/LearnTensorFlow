# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import IPython.display as display
import os
import random
import pathlib
"""
Created on Sun Mar  3 09:48:06 2019

@author: Administrator
"""
# 本教程提供了一个如何使用加载图像数据集的简单示例:tf.data

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE

# 下载并检查数据集
'''
在开始任何培训之前，您需要一组图像来向网络教授您想要识别的新类。我们创建了一个
creative-commons授权的花卉照片存档，最初可以使用。
'''
data_root_orig = tf.keras.utils.get_file('flower_photos',
                                         'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         untar=True)
data_root = pathlib.Path(data_root_orig)
print(data_root)

# 下载218MB后，你现在应该有一份花卉照片的副本:
for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count

all_image_paths[:10]

# 检查图片 现在让我们快速看几张图片，这样我们就知道我们在处理什么了:
attributions = (data_root/"LICENSE.txt").open(encoding='utf-8').readlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)


def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


# 确定每个图像的标签
# 列出可用标签:
label_names = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())
label_names

# 为每个标签分配索引:
label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index

# 创建每个文件及其标签索引的列表
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("前十个标签: ", all_image_labels[:10])

# 加载并格式化图像
'''
TensorFlow包括所有你需要加载和处理图像的工具:
'''
img_path = all_image_paths[0]
img_path

# 以下是原始数据:
img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

# 解码成图像张量:
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)

# 为您的模型调整大小:
img_final = tf.image.resize(img_tensor, [192, 192])
img_final = img_final/255.0
print(img_final.shape)
print(img_final.numpy().min())
print(img_final.numpy().max())

# 在后面的简单函数中总结这些。


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # [0,1]

    return image


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


image_path = all_image_paths[0]
label = all_image_labels[0]
plt.imshow(load_and_preprocess_image(img_path))
plt.grid(False)
plt.xlabel(caption_image(img_path).encode('utf-8'))
plt.title(label_names[label].title())
print()

# 建立一个tf.data.Dataset
'''
图像集
构建tf.data的最简单方法。Dataset使用from_tensor_sections方法。
将字符串数组切片，得到一个字符串数据集:
'''

# output_shapes和output_types字段描述数据集中每个项的内容。在本例中，它是一组标量二进制字符串
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)

# 现在创建一个新的数据集，通过在路径数据集上映射preprocess_image动态加载和格式化图像。
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)


plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))

# (图像、标签)对的数据集
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
    print(label_names[label.numpy()])

# 由于数据集的顺序是相同的，我们可以将它们压缩在一起，以获得(图像、标签)对的数据集。
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

# 新数据集的形状和类型也是形状和类型的元组，描述每个字段:
print(image_label_ds)

# 意:当您有all_image_tags和all_image_paths这样的数组时，tf.data.dataset.Dataset.zip的另一种选择是切片这对数组。
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# 元组被解压缩到映射函数的位置参数中


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

# 训练的基本方法
'''
要使用此数据集训练模型，您将需要以下数据:
    1:打乱数据
    2:批处理。
    3:重复训练
    4:尽可能的批量
使用tf.data api.可以很容易地添加这些特性。
'''
BATCH_SIZE = 32

# 设置与数据集一样大的洗牌缓冲区大小可以确保数据的大小完全打乱
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# prefetch”允许数据集在模型训练时在后台获取批数据。
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds
