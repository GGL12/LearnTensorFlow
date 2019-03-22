# 变化的卷积自编码
'''
本笔记本演示了如何通过训练变分自动编码器(1,2)生成手写数字的图像。
'''

# 导入tensorflow和其他第三方库
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

# 加载MNIST数据集
'''
    每个MNIST图像最初都是一个784个整数的向量，每个整数都在0-255之间，表示像素的强度。
我们用模型中的伯努利分布为每个像素建模，并静态地对数据集进行二值化。
'''
(trian_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
trian_images = trian_images.reshape(
    trian_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(
    test_images.shape[0], 28, 28, 1).astype('float32')

# 0-1归一化
trian_images = trian_images / 255.
test_images = test_images / 255.

# 0-1灰度化
trian_images[trian_images >= .5] = 1
trian_images[trian_images < .5] = 0

test_images[test_images >= .5] = 1
test_images[test_images < .5] = 0

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

# 使用tf.data用于创建批处理和洗牌数据集

train_dataset = tf.data.Dataset.from_tensor_slices(
    trian_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

# 用tf.keras.Sequential连接生成网络和推理网络
'''
    在VAE的例子中，我们使用了两个小的对流作为生成网络和推理网络。由于这些神经网络很小，
我们使用tf.keras。顺序来简化我们的代码。设x和z分别表示观测值和潜在变量，如下所示。

    生成网络:
    这就定义了生成模型，生成模型以一个潜在的编码作为输入，输出观测条件分布的参数，
即p(x|z)。此外，我们使用单位高斯先验p(z)作为潜在变量。

    推理网络:
    这定义了一个近似的后验分布q(z|x)，它以一个观测值作为输入，输出一组参数，用于潜在表示
的条件分布。在这个例子中，我们简单地将这个分布建模为对角高斯分布。在这种情况下，推理网络
输出一个因式高斯函数的均值和对数方差参数(对数方差代替方差直接用于数值稳定性)。

    重新参数化:
    在优化过程中，我们可以对q(z|x)进行采样，首先对一个单位高斯函数进行采样，然后乘以
标准差，再加上平均值。这确保梯度可以通过样本传递到推理网络参数。

    网络体系结构:
    对于推理网络，我们使用了两个卷积层和一个全连接层。在生成网络中，我们通过使用一个
完全连接的层和三个卷积转置层(在某些上下文中也称为反卷积层)来镜像这个架构。注意，在训练
VAEs时通常避免使用批处理标准化，因为使用小批处理带来的额外随机性可能加剧抽样随机性之上
的不稳定性。
'''


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'
            ),
            tf.keras.layers.Flatten(),
            # 没有激活函数
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'
            ),
            # 没有激活函数
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"
            )
        ])

    def sample(self, eps):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(
            x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(slef, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


# 定义损失函数和优化器
'''
    VAEs通过最大化边际对数可能性的证据下界(ELBO)来训练:
logp(x)≥ELBO=Eq(z|x)[logp(x,z)q(z|x)].
    在实践中，我们对该期望的单样本蒙特卡罗估计进行了优化:
logp(x|z)+logp(z)−logq(z|x),
    其中z是从q(z|x)中采样的。
注意:我们也可以分析计算KL项，但是为了简单起见，这里我们将所有这三个项都包含在蒙特卡罗估计器中。
'''
optimizer = tf.keras.optimizer.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis
    )


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))
