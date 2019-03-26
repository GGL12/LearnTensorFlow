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

'''
允许GPU内存增长:
    默认情况下，TensorFlow将进程可见的所有GPU(取决于CUDA_VISIBLE_DEVICES)的几乎所有GPU
内存映射到该进程。这样做是为了通过减少内存碎片更有效地使用设备上相对珍贵的GPU内存资源。

    在某些情况下，希望进程只分配可用内存的子集，或者只根据进程的需要增加内存使用量。
TensorFlow提供了两种方法来控制这一点。

    第一个选项是通过调用tf.config.gpu.set_per_process_memory_growth打开内存增长()方法,
它试图分配仅GPU内存中所需的运行时配置:一开始分配内存很少,随着程序运行和需要更多的GPU内存,
我们扩展了GPU内存区域分配给TensorFlow过程。注意，我们不释放内存，因为这会导致更糟糕的
内存碎片。要打开进程内存增长，请将此作为程序的第一条语句:

        tf.config.gpu.set_per_process_memory_growth()

    第二种方法是tf.gpu.set_per_process_memory_fraction()，它确定每个可见GPU应该分配
的内存总量的百分比。例如，你可以告诉TensorFlow只分配每个GPU总内存的40%:

        tf.config.gpu.set_per_process_memory_fraction(0.4)

    如果您想真正地将GPU可用内存的数量绑定到TensorFlow进程，这是非常有用的。
'''

'''
在多GPU系统上使用一个GPU:
    如果您的系统中有多个GPU，默认情况下将选择ID最低的GPU。如果你想在不同的GPU上运行，你需要明确指定首选项:
'''
tf.debugging.set_log_device_placement(True)
with tf.device('/device:GPU:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
print(c)

# 如果您指定的设备不存在，您将得到RuntimeError:
'''
RuntimeError: Error copying tensor to device: /job:localhost/replica:0/task:0/device:GPU:2. /job:localhost/replica:0/task:0/device:GPU:2 unknown device.
'''

'''
    如果希望TensorFlow自动选择一个现有且受支持的设备来运行操作，以防指定的设备不存在，
可以调用tf.config.set_soft_device_placement(True)。
'''
tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

with tf.device('/device:GPU:2'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

print(c)


'''
使用多个处理器:
    使用多个gpu的最佳实践是使用tf. distribution . strategy。下面是一个简单的例子:
'''
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(1,))
    predictions = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mse',
                  optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.2))

'''
    这个程序将在每个GPU上运行你的模型的副本，在它们之间分割输入数据，也称为“数据并行性”。

    tf.distribute。策略通过跨设备复制计算而在幕后发挥作用。您可以通过在每个GPU上构建模型来手动实现复制。例如:
'''
tf.debugging.set_log_device_placement(True)

# Replicate your computation on multiple GPUs
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
    with tf.device(d):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
    sum = tf.add_n(c)

print(sum)
