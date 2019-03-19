# 图像字幕
'''
    给定如下的图片，我们的目标是生成一个标题，例如“一个冲浪运动员骑在波浪上”。
    这里，我们将使用一个基于注意力的模型。这使我们能够看到模型在生成标题时所关注的图像的哪些部分。
'''
from __future__ import division, absolute_import, print_function
import tensorflow as tf
# 我们将生成注意力图，以查看图像的哪些部分
# 我们的模型专注于标题期间
import matplotlib.pyplot as plt
# Scikit-learn包括许多有用的实用程序
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import numpy as np
import os
import json
from glob import glob
from PIL import Image
import pickle

# 下载并准备MS-COCO数据集
'''
我们将使用MS-COCO数据集来训练我们的模型。这个数据集包含了>82,000个图像，每个图像都至少有5个不同的注释。下面的代码将自动下载并提取数据集。
注意:前面有大量的下载。我们将使用训练集，它是一个13GB的文件。
'''
annotation_zip = tf.keras.utils.get_file('captions.zip',
                                         cache_subdir=os.path.abspath('.'),
                                         origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                         extract=True)
annotation_file = os.path.dirname(
    annotation_zip)+'/annotations/captions_train2014.json'
name_of_zip = 'train2014.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
    image_zip = tf.keras.utils.get_file(name_of_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    PATH = os.path.dirname(image_zip)+'/train2014/'
else:
    PATH = os.path.abspath('.')+'/train2014/'


# 可以选择限制训练集的大小，以便更快地进行训练
'''
对于本例，我们将选择30,000个标题的子集，并使用这些标题和相应的图像来训练我们的模型。
与往常一样，如果您选择使用更多的数据，标题质量将会提高。
'''
# 读取json的文件
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# 在向量中存储标题和图像名称
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = "<start>" + annot['caption'] + '<end>'
    image_id = annot['image_id']
    full_coco_iamge_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_iamge_path)
    all_captions.append(caption)

# captions和image_names混在一起设置随机状态
train_captions, img_name_vector = shuffle(
    all_captions,
    all_img_name_vector,
    random_state=1
)
# 从洗牌集中选择前30000个标题
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

len(train_captions), len(all_captions)

# 使用InceptionV3对图像进行预处理
'''
接下来，我们将使用InceptionV3(在Imagenet上预先训练)对每个图像进行分类。我们将从最后一个卷积层中提取特征。
首先，我们需要将图像转换成inceptionV3期望的格式:
    1:将映像大小调整为(299,299)
    2:使用preprocess_input方法将像素放置在-1到1的范围内(以匹配用于训练InceptionV3的图像的格式)。
'''


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.Inception_v3.preprocess_input(img)
    return img, image_path


# 初始化InceptionV3并加载预训练的Imagenet权重
'''
    为此，我们将创建一个tf。keras模型，其中输出层是InceptionV3体系结构中的最后一个卷积层。
每个图像都通过网络转发，最后得到的向量存储在字典中(image_name—> feature_vector)。
我们使用最后一个卷积层因为我们在这个例子中使用了注意力。该层输出的形状为8x8x2048。
我们在训练中避免这样做，这样就不会成为瓶颈。
在所有图像通过网络之后，我们对字典进行pickle并将其保存到磁盘。
'''
image_model = tf.keras.applications.Inception_v3(
    include_top=False,
    weights='imagenet'
)
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

iamge_features_extract_model = tf.keras.Model(
    new_input,
    hidden_layer
)

# 缓存从InceptionV3中提取的特性
'''
    我们将使用InceptionV3对每个图像进行预处理，并将输出缓存到磁盘。将输出缓存到RAM中会
更快，但需要占用内存，每个映像需要8 * 8 * 2048个浮点数。在编写本文时，这将超过Colab的内存限制
(尽管这些限制可能会发生变化，但是一个实例目前似乎有大约12GB的内存)。
使用更复杂的缓存策略(例如，通过分片图像来减少随机访问磁盘I/O)可以提高性能，但代价是需要
更多的代码。
'''

# 得到唯一的图像
encode_train = sorted(set(img_name_vector))
# 根据您的系统配置随意更改batch_size
iamge_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
iamge_dataset = iamge_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE.batch(16)
)


for img, path in iamge_dataset:
    batch_features = iamge_features_extract_model(img)
    batch_features = tf.reshape(
        batch_features,
        (batch_features.shape[0], -1, batch_features.shape[3])
    )
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

# 对标题进行预处理和标记
'''
首先，我们将标记标题(例如，通过分隔空格)。这将为我们提供数据中所有独特单词的词汇表(如“surfing”、“football”等)。
接下来，我们将把词汇量限制在前5000个单词以内，以节省内存。我们将用标记“UNK”(表示未知)替换所有其他单词。
最后，我们创建一个单词——>索引映射，反之亦然。
然后我们将所有序列填充为与最长序列相同的长度。
'''
