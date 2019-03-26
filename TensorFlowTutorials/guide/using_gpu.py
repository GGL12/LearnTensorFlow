# 使用gpu
'''
    支持的设备:
TensorFlow支持在各种类型的设备上运行计算，包括CPU和GPU。它们用字符串表示，例如:
1:“/cpu:0”:您机器的cpu。
2:“/device:GPU:0”:你机器上第一个对TensorFlow可见的GPU
3:“/device:GPU:1”:你机器上的第二个GPU，它对TensorFlow等可见。
如果一个TensorFlow操作同时具有CPU和GPU实现，默认情况下，当操作被分配给一个设备时，GPU设
备将获得优先级。例如，matmul同时具有CPU和GPU内核。在一个设备cpu:0和gpu:0的系统上，除非您
指定地请求在另一个设备上运行matmul，否则将选择gpu:0运行matmul。
'''
import tensorflow as tf
'''
    日志设备位置:
要找出操作和张量分配给了哪些设备，可以将tf.debug .set_log_device_placement(True)作为程序的第一个语句
'''
tf.debugging.set_log_device_placement(True)
# 创建一些张量
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
print(c)

# 您应该会看到以下输出:
'''
Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
'''

'''
    手动设备位置:
如果您希望在您所选择的设备上运行特定的操作，而不是自动为您选择的操作，您可以使用tf.device
来创建一个设备上下文，该上下文中的所有操作都将在相同的指定设备上运行。
'''
tf.debugging.set_log_device_placement(True)
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
print(c)

# 您将看到现在a和b被分配给cpu:0。由于没有为MatMul操作显式指定设备，TensorFlow运行时
# 将根据操作和可用设备(本例中的cpu:0)选择一个设备，如果需要，还会在设备之间自动复制张量。
'''
Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
tf.Tensor(
[[22. 28.]
 [49. 64.]], shape=(2, 2), dtype=float32)
'''
