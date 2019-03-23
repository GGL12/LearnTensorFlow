# 深卷积生成对抗性网络
'''
    本教程演示如何使用深度卷积生成对抗性网络(DCGAN)生成手写数字的图像。代码是使用带有tf的
Keras序列API编写的。GradientTape训练循环。
    GANs是什么?   
    生成对抗网络(GANs)是当今计算机科学中最有趣的概念之一。两个模型通过对抗性过程同时
训练。生成器(“艺术家”)学会创建看起来真实的图像，而鉴别器(“艺术评论家”)学会区分真实图像和赝品。
    在训练过程中，生成器逐渐变得更擅长创建看起来真实的图像，而鉴别器则变得更擅长区分它们。
当鉴别器无法分辨真伪图像时，该过程达到平衡。 
    本笔记本在MNIST数据集中演示了这个过程。下面的动画展示了生成器在经过50个时代的训练后生
成的一系列图像。这些图像一开始是随机噪声，随着时间的推移越来越像手写数字。
'''

# 导入TensorFlow和其他库
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow.python.keras.layers as layers
import time
from IPython import display

# 加载和准备数据集
'''
您将使用MNIST数据集来训练生成器和鉴别器。生成器将生成类似MNIST数据的手写数字。
'''
(train_images, train_labels), (_, _) = tf.python.keras.datasets.mnist.load_data()
train_images = train_images.reshpe(
    train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 将图像归一化为[- 1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# 创建模型
'''
生成器和鉴别器都是使用Keras顺序API定义的。
    生成器使用tf.keras.layers。conv2d转置(上采样)层从种子(随机噪声)生成图像。
从一个以该种子为输入的密集层开始，然后向上采样几次，直到达到所需的图像大小28x28x1。
注意到tf.keras.layers。每个层的LeakyReLU激活，但使用tanh的输出层除外。
'''


def make_generator_model():
    model = tf.python.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 255)

    model.add(layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 255)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                                     padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model
