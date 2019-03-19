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
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path
