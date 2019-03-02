# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:17:39 2019

@author: Administrator
"""

import tensorflow as tf

tf.enable_eager_execution()

#Layer
layer = tf.keras.layers.Dense(100)
#None 自动推导
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
#要使用一个层，只需调用它
layer(tf.zeros([10, 5]))
#检测层变量
layer.variables
layer.kernel, layer.bias


#实现自定义层
'''
实现您自己的层的最佳方法是扩展tf.keras.layer:
    1.__init__:在这里可以进行所有与输入无关的初始化
    2.build:你知道输入张量的形状，并能完成其余的初始化
    3.call:这里是正向计算
'''
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,num_outputs):
        super(MyDenseLayer,self).__init__()
        self.num_outputs = num_outputs
    
    def build(self,input_shape):
        #注意，您不必等到调用build来创建变量时，也可以在__init__中创建它们。
        self.kernel = self.add_variable(
                "kernel",
                shape=[int(input_shape[-1]),self.num_outputs]
                )
    def call(self,input):
        return tf.matmul(input,self.kernel)
    
layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)


#模型:组合层
'''
机器学习模型中许多有趣的类似于层的东西都是通过组合现有的层来实现的。例如，
resnet中的每个残差快都是卷积、批处理规范化和快捷方式的组合。
创建包含其他层的类分层时使用的主类是tf.keras.Model。实现一个是通过继承tf.keras.Model来完成的。
'''
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self,kernel_size,filters):
        super(ResnetIdentityBlock,self).__init__(name='')
        filters1,filters2,filters3 = filters
        
        self.conv2a = tf.keras.layers.Conv2D(filters1,(1,1))
        self.bn2a = tf.keras.layers.BatchNormalization()
       
        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()
        
        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()
        
    def call(self, input_tensor, training=False):
        #BatchNormalization不进行反向传播
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
    
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
    
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        
        x += input_tensor
        return tf.nn.relu(x)

block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])

'''
然而，很多时候，由许多层组成的模型只是一个接一个地调用层。使用tf.keras.Sequential可以在非常少的代码中实现这一点
'''
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(2, 1, 
                                                      padding='same'),
                               tf.keras.layers.BatchNormalization(),
                               tf.keras.layers.Conv2D(3, (1, 1)),
                               tf.keras.layers.BatchNormalization()])
my_seq(tf.zeros([1, 2, 3, 3]))









