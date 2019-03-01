# -*- coding: utf-8 -*-
"""
自动微分 
tf.GradientTape 
"""
import tensorflow as tf
tf.enable_eager_execution()

#例子
x = tf.ones((2, 2))
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

#z对x求导
dz_dx = t.gradient(z, x)
for i in [0, 1]:
  for j in [0, 1]:
    assert dz_dx[i][j].numpy() == 8.0
    
  
x = tf.ones((2, 2)) 
with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

'''
默认情况下，GradientTape所持有的资源在调用gradienttap .gradient()方
法时立即释放
'''
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
  t.watch(x)
  y = x * x
  z = y * y
dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = t.gradient(y, x)  # 6.0
del t  # 释放资源

#函数式求导
def f(x, y):
  output = 1.0
  for i in range(y):
    if i > 1 and i < 5:
      output = tf.multiply(output, x)
  return output

def grad(x, y):
  with tf.GradientTape() as t:
    t.watch(x)
    out = f(x, y)
  return t.gradient(out, x) 

x = tf.convert_to_tensor(2.0)
assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0


#自动微分嵌套
x = tf.Variable(1.0)#创建初始化为1.0的Tensorflow变量
with tf.GradientTape() as t:
  with tf.GradientTape() as t2:
    y = x * x * x
  # 计算“t”上下文管理器中的梯度
  # 这意味着梯度计算也是可微的。
  dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)
assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0