import tensorflow as tf
tf.enable_eager_execution()

#Tensor
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))
print(tf.encode_base64("hello world"))
#支持操作符运算
print(tf.square(2) + tf.square(3))

#每个张量都有一个形状和一个数据类型
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

'''
NumPy数组和TensorFlow张量之间最明显的区别是:
    1:张量可以由硬件加速(如GPU、TPU)支持。
    2:张量是不可变的
'''



#NumPy兼容性

'''
张量流张量和NumPy ndarray之间的转换非常简单:
    1:TensorFlow操作自动将NumPy ndarra格式转换为Tensor。
    2:NumPy操作自动将Tensor转换为NumPy ndarray。
'''
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow操作自动将NumPy ndarra格式转换为Tensor")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("NumPy操作自动将Tensor转换为NumPy ndarray")
print(np.add(tensor, 1))

print(".numpy() 方法将Tensor转换为numpy数组")
print(tensor.numpy())


#GPU加速(如果你安装了gpu版tf)
x = tf.random_uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

#数据集
'''
使用tf.data.Dateset API:
    1:创建一个数据集
    2:迭代数据集在开启 eager execution模块  
'''
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)

#使用map batch shuffle等转换函数对数据集的记录应用转换。参见tf.data的API文档。数据集的详细信息。
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

#迭代打印
for x in ds_tensors:
  print(x)

for x in ds_file:
  print(x)