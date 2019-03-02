# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 09:48:13 2019

@author: Administrator
"""

import tensorflow as tf
tf.enable_eager_execution()


#变量
'''
TensorFlow中的张量是不可变的无状态对象。然而，机器学习模型需要改变状态:
随着模型的训练，计算预测的相同代码应该随着时间的推移表现不同(希望损失更小!)
要表示在计算过程中需要更改的状态，可以选择依赖于Python是一种有状态编程语言
这一事实
'''
#python状态
x = tf.zeros([10,10])
x += 2 #这就等于x = x + 2，它不改变原式
print(x)


v = tf.contrib.eager.Variable(1.0)
assert v.numpy() == 1.0
#重新指定v的值
v.assign(3.0)
assert v.numpy() == 3.0
#在像tf.square()这样的TensorFlow操作中使用“v”并重新分配
v.assign(tf.square(v))
assert v.numpy() == 9.0


#例子:拟合一个线性模型
'''
现在让我们把到目前为止的一些概念——张量，梯度带，变量——用来建立和训练一个简单的
模型。这通常涉及几个步骤:
    1:定义模型
    2:定义损失方程
    3:获取训练数据
    4:运行训练数据并使用“optimzer”调整变量以来fit数据
'''
class Model(object):
    def __init__(self):
        #初始化变量(5.0,0.0)
        #在实践中，这些应该初始化为随机值
        self.W = tf.contrib.eager.Variable(5.0)
        self.b = tf.contrib.eager.Variable(0.0)
        
    def __call__(self,x):
        return self.W * x + self.b

model = Model()
        
#定义损失函数
def loss(predicted_y,desired_y):
    #MAE损失函数
    return tf.reduce_mean(tf.square(predicted_y - desired_y))
    
#获取训练数据
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random_normal(shape=[NUM_EXAMPLES])
noise   = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

'''
在我们训练这个模型之前，让我们先看看这个模型的数据分布。
我们将用红色表示模型的预测，用蓝色表示训练数据。
'''
import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('当前损失: '),
print(loss(model(inputs), outputs).numpy())

#定义一个训练循环
def train(model,inputs,outputs,learning_rate):
    #使用自动微分
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs),outputs)
    dW,db = t.gradient(current_loss,[model.W,model.b])
    #更新参数
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


#最后，让我们重复运行训练数据，看看W和b是如何演变的
model = Model()
#收集w值和b值的历史记录，以便以后绘图
Ws,bs = [],[]
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)
    
    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

#绘制
plt.plot(epochs,Ws,'r',
         epochs,bs,'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true_b'])
plt.show()












