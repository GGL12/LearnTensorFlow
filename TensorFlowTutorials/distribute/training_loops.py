'''
    本教程演示如何使用tf.distribute.Strategy自定义训练循环的策略。我们将在fashion MNIST数
据集中训练一个简单的CNN模型。fashion MNIST数据集包含60000张尺寸为28×28的训练图像和10000张尺寸为28×28的测试图像。
    我们使用定制的训练循环来训练我们的模型，因为它们给了我们灵活性和对训练更大的控制。此外，它更容易调试模型和训练循环。
'''
from __future__ import absolute_import,division,print_function

import tensorflow as tf 
import numpy as np 
import os

#下载数据
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#数组添加维度——> new shape == (28,28,1)我们这样做是因为我们模型的第一层是卷积的
#它需要4D输入(batch_size, height, width, channels)。
# batch_size维度将在稍后添加。

train_images = train_images[...,None]
test_images = test_images[...,None]

#0 1 初始化
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

#创建一个分配变量和图表的策略
'''
1:所有的变量和模型图都被复制到副本上。
2:输入均匀地分布在各个副本上。
3:每个副本计算它接收到的输入的损失和梯度。
4:梯度通过对所有副本求和来同步。
5:同步之后，对每个副本上的变量副本进行相同的更新。
'''
#方法中未指定设备,在tf.distribute.MirroredStrategy，它将被自动检测到。
strategy = tf.distribute.MirroredStrategy()

print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#设置输入管道
'''
如果一个模型是在多个gpu上训练的，那么就应该相应地增加批处理大小，以便有效地利用额外的计算能力。此外，学习速度应该相应地调整。
'''

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
train_steps_per_epoch = len(train_images) // BATCH_SIZE
test_steps_per_epoch = len(test_images) // BATCH_SIZE

with strategy.scope():
    train_iterator = strategy.experimental_make_numpy_iterator(
        (train_images,train_labels),BATCH_SIZE,shuffle=BUFFER_SIZE
    )
    test_iterator = strategy.experimental_make_numpy_iterator(
        (test_images,test_labels),BATCH_SIZE,shuffle=BUFFER_SIZE
    )

#创建模型
def create_model():
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

#创建检查点目录来存储检查点。
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#定义损失函数
'''
    通常，在一台拥有1个GPU/CPU的机器上，损失除以批量输入的示例数。
那么，当使用tf.distribute.Strategy时，如何计算损失呢?
    1:例如，假设您有4个GPU，批处理大小为64。一批输入分布在多个副本(4个gpu)上，每个副本的输入大小为16。
    2:每个副本上的模型使用其各自的输入进行转发并计算损失。现在，不再用损失除以其各自输入(16)中的示例数，而是用损失除以全局输入大小(64)。
为什么要这样做?
    1:这样做是因为在计算每个副本上的梯度之后，通过对它们求和，可以跨副本同步梯度。
如何在TensorFlow中处理这个?
    1:tf.keras.losses会自动处理。
    2:如果分发自定义丢失函数，不要使用tf.reduce_mean(除以本地批量大小)实现他，将总和除以全局批量大小:cale_loss = tf.reduce_sum(loss) * (1. / global_batch_size)
'''
with strategy.scope():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

#定义度量来跟踪损失和准确性
'''
这些指标跟踪损失和准确性。您可以使用.result()在任何时候获取累积的统计信息。
'''
with strategy.scope():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#训练

#模型和优化器必须在“strategy.scope”下创建。
with strategy.scope():
    model = create_model()

    optimizer = tf.keras.optimizer.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model)

with strategy.scope():
    #训练
    def train_step(inputs):
        images,labels = input

        with tf.GradientTape() as tape:
            predictions = model(images,training=True)
            loss = loss_object(labels,predictions)

        gradients = tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradient(zip(gradients,model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels,predictions)

    #测试
    def test_step(inputs):
        images ,labels = inputs

        predictions = model(images,training=False)
        t_loss = loss_object(labels,predictions)

        test_loss(t_loss)
        test_accuracy(labels,predictions)