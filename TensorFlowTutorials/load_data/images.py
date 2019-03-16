# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:48:06 2019

@author: Administrator
"""
#本教程提供了一个如何使用加载图像数据集的简单示例:tf.data

from __future__ import absolute_import, division, print_function

import tensorflow as tf
tf.enable_eager_execution()
tf.VERSION

AUTOTUNE = tf.data.experimental.AUTOTUNE