# 使用tf.kears 进行图像分类
'''
    使用tf.kears.Sequential搭建模型
    使用tf.kears.preprocessing.image.ImageDataGenerator来加载数据

    一些问题:
        1:数据通过API tf.keras.preprocessing.image.ImageDataGenerator进行输入，我们如何有效地处理磁盘上的数据以与我们的模型进行接口?
        2:过度拟合——什么是过度拟合?如何识别过度拟合?如何预防过度拟合?
        3:数据增强和dropout-关键技术，以降低过度拟合的计算机视觉任务，我们将纳入我们本次的图像分类器模型当中。

    我们将遵循一般的机器学习流程:
        1:检查和理解数据
        2:构建输入管道
        3:搭建模型
        4:训练模型
        5:测试模型
        6:不断改进我们的模型
'''

#导入相关的包
import os
import numpy as np 
import matplotlib.pyplot as plt 
#解压数据
import zipfile

#使用tensorflow as tf 和 tf.keras，将采用不同的方法搭建我们的模型.
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

#加载数据 我们使用的数据集是Kaggle中狗对猫数据集 https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \-O /tmp/cats_and_dogs_filtered.zip

#下载的数据集路径
local_zip = '/tmp/cats_and_dogs_filtered.zip' 
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp') # contents are extracted to '/tmp' folder
zip_ref.close()

#设置训练集、验证集的路径
base_dir = '/tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#查看数据基本信息

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))
num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))
total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print('猫的图片总计有：:', num_cats_tr)
print('狗的图片总计有：', num_dogs_tr)

print('验证集猫的图片总计有：', num_cats_val)
print('验证集狗的图片总计有：', num_dogs_val)
print("--")
print("训练集图片总计有：", total_train)
print("验证集图片总计有：", total_val)

#设置模型的参数
batch_size = 100
epochs = 15
IMG_SHAPE = 150#我们的训练数据由150像素宽和150像素高的图像组成

'''
数据准备:
    1:从磁盘中读取数据
    2:解码数据并转化为RGB格式
    3:将它们转换为浮点张量
    4:将张量从0到255之间的值调整为0到1之间的值，因为神经网络更喜欢处理较小的输入值。
幸运的是，所有这些任务都可以使用tf中提供的一个类来完成。keras预处理模块，称为
ImageDataGenerator。不仅可以从磁盘读取的图像预处理和图像到适当的张量,但它也
将这些图像转变成批量的张量,这将是非常有用的在训练我们的网络在我们需
要通过网络的形式批量的输入。
'''

#生成我们的训练数据
train_image_generator = ImageDataGenerator(rescale=1./255) 
#生成我们的验证数据
validation_image_generator = ImageDataGenerator(rescale=1./255)

#在定义了用于训练和验证图像的生成器之后，flow_from_directory
#api将从磁盘加载图像，并将应用调整大小，并使用单行代码将其调整为所需的大小。
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, 
                                                     directory=train_dir, 
                                                     # 通常最好的做法是打乱数据
                                                     shuffle=True, 
                                                     target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150) 
                                                     class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, 
                                                              directory=validation_dir, 
                                                              target_size=(IMG_SHAPE,IMG_SHAPE), #(150,150)
                                                              class_mode='binary')

#查看样例图片
sample_training_images, _ = next(train_data_gen)

#这个函数将以1行5列的网格形式绘制图像，每个列中放置图像。
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()
plotImages(sample_training_images[:5])


#搭建模型
#该模型由3个最大池层的卷积块组成。我们有一个完全连接的层，上面有512个单元，它被relu激活函数激活。模型将输出基于二分类的类概率，二分类由sigmoid激活函数完成
model = Sequential()
model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_SHAPE,IMG_SHAPE, 3,))) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#编译模型
'''
我们将使用ADAM优化器作为我们的选择优化这项任务和二元交叉熵函数作为损失函数。
我们还希望在训练我们的网络时查看每个epoch的培训和验证准确性，因为我们是在
metrics参数中传递它的。
'''
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy']
             )
#训练模型
#我们将使用fit_generator函数来训练我们的网络，而不是使用fit函数，因为我们使用ImageDataGenerator类来为我们的网络生成批量的训练和验证数据。
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(batch_size)))
)

#可视化训练结果
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#过拟合
'''
如果我们看一下上面的图表，我们可以看到，随着时间的推移，训练精度呈线性增长，而在我们的训练
过程中，验证精度在一段时间后会停滞在70%左右。此外，训练和验证的准确性之间的差异是显著的。
这是过度契合的表现。

我们有少量的训练样本时，我们的模型有时会从我们的训练样本中学习噪音或不需要的细节，这在一
定程度上会对新样本的模型性能产生负面影响。这种现象被称为过拟合。这仅仅意味着我们的模型很
难很好地在新数据集上推广。

在我们的训练过程中，有多种方法可以对抗过度适应。通过Data Augmentation和添加dropout来
使模型更加健壮。让我们从数据扩充开始，看看它将如何帮助我们对抗模型中的过度拟合。
'''

#清除资源
tf.keras.backend.clear_session()
epochs = 80

#数据增强
'''
过度拟合通常发生在训练样本较少的情况下。解决这个问题的一种方法是扩充我们的数据集，使其具
有足够数量的训练示例。数据增强是从现有的训练样本中生成更多的训练数据的方法，通过一些随机
变换对样本进行增强，生成可信的图像。我们的目标是，在训练时，您的模型永远不会两次看到完全
相同的图像。这有助于将模型公开到数据的更多方面，并更好地一般化

在tf.keras中我们可以使用之前使用的ImageDataGenerator类来实现这一点。我们可以简单地将我们想
要的不同转换作为参数的形式传递给数据集，它将在我们的训练过程中负责将其应用到数据集。
'''
#增强和可视化数据
#我们可以从对数据集应用随机水平翻转增强开始，看看转换后的图像会是什么样子。

#应用水平翻转
#我们可以简单地将horizontal_flip作为参数传递给ImageDataGenerator类，并将其设置为True以应用这个扩展。
iamge_gen = ImageDataGenerator(rescale=1/255.0,horizontal_flip=True)

train_data_gen = iamge_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE,IMG_SHAPE)
)
#让我们从我们的训练示例中选取一个样本图像，并将其重复5次，这样就可以将增强随机地应用到相同的图像上5次，以查看增强效果。
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#随机旋转图片
#让我们来看看不同的叫做旋转的增强，并将45度的旋转随机应用到我们的训练例子中。
iamge_gen = ImageDataGenerator(rescale=1/255.0,rotation_range=45)
train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size, 
                                                directory=train_dir, 
                                                shuffle=True, 
                                                target_size=(IMG_SHAPE, IMG_SHAPE)
                                                )

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#应用放大
#让我们应用缩放增强到我们的数据集缩放图像高达50%的随机性
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size, 
                                                directory=train_dir, 
                                                shuffle=True, 
                                                target_size=(IMG_SHAPE, IMG_SHAPE)
                                                )

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)



































































