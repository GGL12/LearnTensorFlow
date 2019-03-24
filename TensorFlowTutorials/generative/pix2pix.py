# Pix2Pix
'''
    本笔记本演示了使用条件GAN的图像到图像的翻译，如条件对抗性网络的图像到图像的翻译中所述。
利用这项技术，我们可以为黑白照片上色，将谷歌地图转换为谷歌地球，等等。在这里，我们将建筑
立面转换为真实的建筑。
    例如，我们将使用位于布拉格的捷克技术大学机器感知中心提供的CMP外观数据库。为了保持示例
简短，我们将使用这个数据集的预处理副本，该副本由上述文章的作者创建。
在一个P100 GPU上，每个epoch大约需要58秒。
'''
# 导入TensorFlow和其他库
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 加载数据
'''
    您可以从这里下载这个数据集和类似的数据集。如本文所述，我们将随机抖动和镜像应用于训练数据集。
1:在随机抖动中，图像大小调整为286 x 286，然后随机裁剪为256 x 256
2:在随机镜像中，图像水平随机翻转i。e从左到右。
'''
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 256
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


inp, re = load(PATH + 'train/100.jpg')
# 为matplotlib转换为int以显示图像
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    strcked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        strcked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]

##将图像归一化为[- 1,1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image
