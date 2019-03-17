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

'''
这里有几件事需要注意:
    1:顺序很重要。
        a.repeat之前的A.shuffle将会跨历历元边界对项进行洗牌(有些项将会在其他项被看到之前被看到两次)。
        在.batch之后的.shuffle将会打乱批次的顺序，但是不会在批次之间打乱项目。
    2:们使用与数据集大小相同的buffer_size进行完全的洗牌。对于数据集大小，较大的值提供更好的随机化，但使用更多的内存。
    3:在从洗牌缓冲区中提取任何元素之前填充它。因此，当数据集启动时，较大的buffer_size可能会导致延迟。
    4:改组后的数据集在改组缓冲区完全为空之前不会报告数据集的结束。数据集由.repeat重新启动，导致另一个等待shuffle-buffer被填充。

最后一点可以通过使用tf.data.Dataset来解决。采用融合tf.数据的方法进行实验shuffle_and_repeat功能:
'''
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_add_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

#将数据集导入模型
'''
从tf.keras.applications获取MobileNet v2的副本。
这将用于一个简单的迁移学习示例。
将MobileNet权重设置为不可训练:
'''
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3),include_top=False)
mobile_net.trainable=False

#该模型期望其输入归一化为[-1,1]范围:
'''
help(keras_applications.mobilenet_v2.preprocess_input)
这个函数应用“Inception”预处理，它将RGB值从[0,255]转换为[- 1,1]
'''

#因此，在将其传递到MobilNet模型之前，我们需要将输入范围从[0,1]转换为[-1,1]。
def change_range(image,label):
    return 2*image-1,label

keras_ds = ds.map(change_range)

#MobileNet为每个图像返回一个6x6的特征空间网格。给它传一组图片看看:

#据集可能需要几秒钟启动，因为它填充了它的shuffle缓冲区。
image_batch,label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

'''
因此，构建一个围绕MobileNet的模型，并使用tf.keras.layers。在输出tf.keras.layers之前，
GlobalAveragePooling2D对这些空间维度进行平均致密层:
'''
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names))
])

#现在它产生预期形状的输出:
logit_batch = model(image_batch).numpy()

print("min logit:",logit_batch.min())
print("max logit:",logit_batch.max())
print()

print("shape:",logit_batch.barch)

#编译模型
model.compile(
    optimizer=tf.train.losses.sparse_categorical_crossentropy,
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)

#有2个可训练变量:weights and bias:
len(model.trainable_variables)
model.summary()

#训练模型。
'''
通常，您会指定每个epoch的实际步骤数，但是出于演示目的，只运行3个步骤。
'''
steps_per_epoch = tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds,epochs=1,steps_per_epoch=3)

#性能
'''
注意:本节只展示了一些可能有助于提高性能的简单技巧。
上面使用的简单管道在每个历元上分别读取每个文件。这对于CPU上的本地训练是可以的，
但是对于GPU的训练可能是不够的，对于任何类型的分布式训练都是完全不合适的。
'''

#要进行研究，首先构建一个简单的函数来检查我们的数据集的性能:
import time
def timeit(ds,batches=2*steps_per_epoch+1):
    overall_start = time.time()
    #取一个批次来启动管道(填充洗牌缓冲区)，
    #开始计时前
    it = iter(ds.take(batches+1))
    next(it)

    start = time.time()
    for i,(image,label) in enumerate(it):
        if i%10 == 0:
            print(".",end="")
    print()
    end = time.time()

    duration = end- start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
    print("Total time: {}s".format(end-overall_start))
        
#当前数据集的性能为:
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_add_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds
timeit(ds)

#缓存
'''
使用tf.data.Dataset。缓存以方便跨时代缓存计算。如果dataq能够装入内存，这将特别具有性能。
这里的图像缓存后，预先预制(解码和调整大小):
'''
ds = image_label_ds.cache()
ds = ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count)
)
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
ds
timeit(ds)

#TFRecord文件
'''
原始图像数据
TFRecord文件是一种存储二进制块序列的简单格式。通过将多个示例打包到同一个文件中，TensorFlow能够同时读取多个示例，这对于使用诸如GCS之类的远程存储服务时的性能尤为重要。
首先，从原始图像数据构建一个TFRecord文件:
'''
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

#接下来，构建一个数据集，它从TFRecord文件中读取数据，并使用前面定义的preprocess_image函数对图像进行解码/重新格式化。
image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)

#将其与我们前面定义的标签数据集一起压缩，以获得预期的(图像、标签)对。
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds
#这比缓存版本慢，因为我们没有缓存预处理。
timeit(ds)

#序列化的张量
'''
为了将一些预处理保存到TFRecord文件中，首先将处理后的图像做成数据集，如下图所示:
'''
paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = paths_ds.map(load_and_preprocess_image)
image_ds

#现在不是.jpeg字符串的数据集，而是张量的数据集。要将其序列化为TFRecord文件，首先要将张量数据集转换为字符串数据集。

ds = image_ds.map(tf.io.serialize_tensor)
ds

tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(ds)

#通过缓存预处理，可以非常有效地从TFrecord文件加载数据。只要记住在使用它之前先去序列化张量。
ds = tf.data.TFRecordDataset('images.tfrec')

def parse(x):
    result = tf.io.parse_tensor(x, out_type=tf.float32)
    result = tf.reshape(result, [192, 192, 3])
    return result

ds = ds.map(parse, num_parallel_calls=AUTOTUNE)
ds

#现在，添加标签并应用与之前相同的标准操作:
ds = tf.data.Dataset.zip((ds, label_ds))
ds = ds.apply(
  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
ds

timeit(ds)