import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
'''
    在本教程中，我们将讨论如何使用从一个预先训练的网络传输学习来对猫狗图像进行分类。
比我们从零开始训练的模型会得到更高的准确性。
    预训练模型是一个保存的网络，它以前在大型数据集上训练过，通常是在大型图像分类任务上。
我们可以直接使用预训练模型也可以使用预训练的模型进行迁移学习。转移学习背后的原理是，
如果这个模型在一个足够大和通用的数据集上训练，这个模型将有效地作为一个视觉世界的通用模型。
我们可以利用这些学习到的特性映射，而不必在大型数据集上训练大型模型，使用这些模型作
为特定于我们任务的模型的基础：
    1：特征提取——使用以前网络学习的表示形式从新样本中提取有意义的特征。我们只需要在预训练
的模型上添加一个新的分类器，它将从头开始训练，这样我们就可以为我们的数据集重新使用前面学到的特征映射。
    2：微调-解冻用于特征提取的冻结模型库的顶层，并联合训练新添加的分类器层和冻结模型的最后一层。
这允许我们在最终分类器之外“微调”更高阶的特征表示，以使它们与所涉及的特定任务更加相关。
    我们将遵循一般的机器学习流程:
    1：检查和理解数据
    2：构建一个输入管道——使用Keras ImageDataGenerator，就像我们在图像分类教程中所做的那样
    3：组成我们的模型：
        3.1：在我们的预训练模型中加载(和预训练权重)
        3.2：把我们的分类层组合起来
    4：训练我们的模型
    5：验证我们的模型
'''
# 我们将看到一个使用预先训练的卷积网络作为特征提取的示例，然后对其进行微调，以训练基本模型的最后几层。
from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version is ", tf.__version__)

# 数据预处理
# 下载猫狗数据
zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
                                   fname="cats_and_dogs_filtered.zip", extract=True)
base_dir, _ = os.path.splitext(zip_file)
# 准备训练和验证猫狗数据集
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
# 样本猫的训练数据目录
train_cats_dir = os.path.join(train_dir, 'cats')
print('Total training cat images:', len(os.listdir(train_cats_dir)))
# 样本狗的训练数据目录
train_dogs_dir = os.path.join(train_dir, 'dogs')
print('Total training dog images:', len(os.listdir(train_dogs_dir)))
# 样本猫的验证数据目录
validation_cats_dir = os.path.join(validation_dir, 'cats')
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
# 样本狗的验证数据目录
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

# 创建具有图像增强功能的图像数据生成器
image_size = 160  # 所以图片shape(160,160)
batch_size = 32

# 将所有图像缩放1/255并应用图像增强
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
# 使用train_datagen生成器以batch20的方式传输训练图像
train_generator = train_datagen.flow_from_directory(
    train_dir,  # 训练图像的源目录
    target_size=(image_size, image_size),
    batch_size=batch_size,
    # 因为我们使用binary_crossentropy loss，所以我们需要二进制标签
    class_mode='binary')
# 使用train_datagen生成器以batch20的方式传输验证图像
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,  # 验证图像的源目录
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='binary')

# 根据预先训练的卷积网络创建基础模型
'''
    我们将从谷歌开发的MobileNet V2模型中创建基础模型，并对ImageNet数据集进行预处理，这是一个
包含1.4M图像和1000类图像的大型数据集。这是一个强大的模型。让我们看看它学到了什么特性可以
解决猫和狗的问题。
首先，我们需要选择使用MobileNet V2的哪个中间层进行特征提取。一种常见的做法是在压平操作之
前使用最后一层的输出，即所谓的“瓶颈层”。这里的原因是，以下完全连接的层对于网络所训练的任务
来说过于专门化，因此这些层所学习的特性对于新任务来说不是很有用。然而，瓶颈特性保留了很多
通用性。
    我们实例化一个预先加载了ImageNet上训练的权重的MobileNet V2模型。通过指定
include_top=False参数，我们加载了一个不包含顶部分类层的网络，这对于特征提取非常理想。
'''
IMG_SHAPE = (image_size, image_size, 3)
# 从预先训练的模型MobileNet V2创建基础模型
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# 特征抽取
'''
我们将冻结上一步创建的卷积库，并将其用作一个特征提取器，在其上添加一个分类器并训练顶级分类器。
在编译和训练模型之前，冻结基于卷积的模型非常重要。通过冻结(或凝固层)。，我们阻止在训练过程中更新这些层中的权重。
'''
base_model.trainable = False
# 让我们看看基本模型体系结构
base_model.summary()
# 添加分类层
model = tf.keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(1, activation='sigmoid')
])
# 编译模型
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# 训练模型
# 经过10次训练，准确率达到94%左右。
epochs = 10
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size
history = model.fit_generator(train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps)
# 学习曲线
'''
我们看看使用MobileNet V2 base模型作为固定特征提取器时，训练和验证精度/损失的学习曲线。
'''
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

# 微调模型
'''
    在我们的特征提取实验中，我们只在MobileNet V2基础模型上训练了几个层。训练过程中未更新训练
前网络的权值。进一步提高性能的一种方法是在对顶级分类器进行训练的同时，“微调”预训练模型顶层
的权重。训练过程将强制从通用特性映射到与数据集相关联的特性来调整权重。
    注意:只有在您将预先训练的模型设置为不可训练的顶级分类器进行了训练之后，才应该尝试这样
做。如果你在一个预先训练的模型上添加一个随机初始化的分类器，并尝试联合训练所有的层，
梯度更新的幅度将会太大(由于来自分类器的随机权重)，你的预先训练的模型将会忘记它所学的一切。
    此外，对预训练模型的顶层进行微调而不是对预训练模型的所有层进行微调的原因如下:
在卷积网络中，后面的层越高，它的专门化程度就越高。卷积层的前几层学习了非常简单和通用的特性，
这些特性几乎适用于所有类型的图像。但是当你往高处走的时候，这些特征对于模型所训练的数据集来
说变得越来越具体。微调的目标是使这些专门的特性适应新的数据集。
'''
# 解冻模型的顶层
# 我们需要做的就是解冻base_model，并设置底层不可训练。然后，重新编译模型(这些更改生效所必需的)，并恢复训练。
base_model.trainable = True
# 我们看看在基本模型中有多少层
print("Number of layers in the base model: ", len(base_model.layers))
# 从这一层开始进行微调
fine_tune_at = 100
# 冻结' fine_tune_at '层之前的所有层
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()
# 继续训练模型
history_fine = model.fit_generator(train_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   workers=4,
                                   validation_data=validation_generator,
                                   validation_steps=validation_steps)
# 学习曲线
acc += history_fine.history['acc']
val_acc += history_fine.history['val_acc']
loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.9, 1])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 0.2])
plt.plot([epochs-1, epochs-1], plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
