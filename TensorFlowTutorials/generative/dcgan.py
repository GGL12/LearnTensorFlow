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

#使用(尚未经过训练的)生成器创建图像。
generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise,training=False)

plt.imshow(generated_image[0,:,:,0],cmap='gray')

#鉴别器是一种基于cnn的图像分类器。
def make_discriminator_model():
    model = tf.python.keras.Sequential()
    model.add(layers.Conv2D(64,(5,5),strides=(1,1),padding='same',input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout())
    model.add(layers.Flatten())
    model.add(layers.Dense())

    return model

#使用(尚未经过训练的)鉴别器将生成的图像分为真图像和假图像。该模型将被训练为对真实图像输出正值，对假图像输出负值。
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

#定义损失和优化器
'''
为这两个模型定义损失函数和优化器。
'''
#这个方法返回一个辅助函数来计算交叉熵损失
cross_entropy = tf.python.keras.losses.BinaryCrossentropy(from_logits=True)

#鉴频器的损失
'''
该方法量化了鉴别器对真伪图像的识别能力。它将disciminator对真实图像的预测与1数组进行比较，
将disciminator对伪造(生成)图像的预测与0数组进行比较。
'''
def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_loss),fake_loss)
    total_loss = real_loss + fake_loss
    return total_loss

#generator的损耗量化了它欺骗鉴别器的能力。直观地说，如果生成器运行良好，鉴别器将把假图像分类为真实的(或1)。
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

#由于我们将分别训练两个网络，因此鉴别器和生成器优化器是不同的。
generator_optimizer = tf.python.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.python.keras.optimizers.Adam(1e-4)

#保存检查点
'''
本笔记本还演示了如何保存和恢复模型，这在长时间运行的训练任务被中断时是很有帮助的。
'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
#定义训练循环
EPOCHS = 50
noise_dim = 100
num_examples_to_genetatr = 16
##我们将在以后重用这个种子(这样会更容易)
#在动画GIF中可视化进度)
seed = tf.random.normal([num_examples_to_genetatr,noise_dim])

'''
训练循环从生成器接收随机种子作为输入开始。种子是用来产生图像的。然后使用鉴别器对真实
图像(来自训练集)和伪造图像(由生成器生成)进行分类。计算了每一种模型的损耗，并利用梯度对
产生器和鉴别器进行了更新。
'''

#注意“tf.function”的用法
#这个注释导致函数被“编译”。
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE,noise_dim])

    with tf.GradientTape() as gen_tape,tf.GradientTape() as dics_tape:
        generated_images = generator(noise,training=True)

        real_output = discriminator(images,training=True)
        fake_output = discriminator(generated_image,training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss,discriminator.trainable_variables)
    gradients_of_discriminator = dics_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset,epochs):
    for epoch in epochs:
        start = time.time

        for image_batch in dataset:
            train_step(image_batch)
        #为GIF生成图像
        display.clear_output(wait=True)
        generate_and_save_images(
            generator,
            epoch + 1,
            seed
        )
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
    
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start)) 

    display.clear_output(wait=True)
    generate_and_save_images(generator,
                           epochs,
                           seed)

#生成和保存图像
def generate_and_save_images(model,epoch,test_input):
    #意“training”被设置为False。所以所有层都在推理模式(batchnorm)下运行。
    predictions = model(test_input,training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0] * 127.5 + 127.5,cmap='gray')
        plt.axis('off')
    plt.savefig('iamge_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

#训练模型
'''
    调用上面定义的train()方法来同时训练生成器和鉴别器。注意，甘斯的训练是很棘手的。重要的是，
生成器和鉴别器不能互相压倒对方(例如，它们的训练速度相似)。
在训练开始时，生成的图像看起来像随机噪声。随着训练的进展，生成的数字将越来越真实。
在大约50个时代之后，它们就像MNIST数字了。使用Colab上的默认设置，这可能需要大约一分钟/历元。
'''

train(train_dataset,EPOCHS)

#恢复最新的检查点。
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#创建一个GIF
'''
使用epoch号显示单个图像
'''

def display_image(epoch_no):
    return PIL.Image.open("image_at_epoch_{:04d}.png".format(epoch_no))

display_image(EPOCHS)

#使用imageio创建一个动画gif使用在培训期间保存的图像。
with imageio.get_writer('dcgan.gif',mode="I") as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)

    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    os.rename('dcgan.gif','dcgan.gif.png')

display.Image(filename="dcgan.gif.png")