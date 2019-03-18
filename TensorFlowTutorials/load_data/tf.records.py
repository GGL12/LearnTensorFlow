# 使用TFRecords和tf.Example
'''
    为了有效地读取数据，将数据序列化并将其存储在一组文件(每个文件100-200MB)中是很有帮助的，
这些文件可以线性读取。如果数据是通过网络传输的，这一点尤其正确。这对于缓存任何数据预处理也很有用。
    TFRecord格式是一种用于存储二进制记录序列的简单格式。
    协议缓冲区是一个跨平台、跨语言的库，用于高效地序列化结构化数据。协议消息由.proto文件定义，这通常是理解消息类型的最简单方法。
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf

import numpy as np
import IPython.display as display
# tf.Example
'''
用于tf.Example的数据类型
    从根本 tf.Example 是一个 {"string": tf.train.Feature} 映射.
1:tf.train.BytesList(以下类型可以强制)
    string
    byte
2:tf.train.FloatList(以下类型可以强制)
    float (float32)
    double (float64)
3:tf.train.Int64List(以下类型可以强制)
    bool
    enum
    int32
    uint32
    int64
    uint64

    以便将标准TensorFlow类型转换为tf。Example-compatible tf.train。功能，您可以使用以下快捷功能:
    每个函数接受一个标量输入值并返回一个tf.train。包含上述三种列表类型之一的特性。
'''

# 可以使用以下函数将值转换为兼容的类型与tf.Example


def _bytes_feature(value):
    # 从字符串/字节返回bytes_list
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList不会从eager张量中解压缩字符串
    return tf.train.Feature(float_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    # 从float / double返回一个float_list。
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    # 从bool / enum / int / uint返回一个int64_list。
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


'''
    注意:为了保持简单，本例仅使用标量输入。处理非标量特性的最简单方法是使用tf。serialize_张量
将张量转换成二进制字符串。字符串是tensorflow中的标量。使用tf。将二进制字符串转换回张量。
    下面是这些函数如何工作的一些例子。注意不同的输入类型和标准化的输出类型。如果函数的输入
类型与上面所述的可强制类型不匹配，函数将引发异常(例如_int64_feature(1.0)将出错，因为1.0是一个浮点数，
所以应该与_float_feature函数一起使用)。
'''
print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))

# 有原始消息都可以使用. serializetostring方法序列化为二进制字符串。
feature = _float_feature(np.exp(1))
feature.SerializeToString()

# 创建一个tf.Example 消息
'''
    假设您想创建在现有数据一个tf.Example消息。在实践中，数据集可能来自任何地方，
但是创建tf.Example的过程。来自单个观察的示例消息将是相同的。
    1:在每个观察中，需要将每个值转换为tf.train.Feature。使用上面的函数之一，包含3种兼容类型之一的特性。
    2:我们创建一个映射(字典)，从特性名称字符串到#1中生成的编码特性值。
    3:在#2中生成的映射被转换为一个特性消息。

在这个笔记本中，我们将使用NumPy创建一个数据集。
这个数据集将有4个特性。
    1:布尔特征，假或真具有相等的概率
    2:从[0,5]中均匀随机选取的整数特征
    3:使用整数特性作为索引，从字符串表生成的字符串特性
    4:标准正态分布的浮点特征

考虑一个包含10,000个来自上述每个分布的独立且相同分布的观测值的样本。
'''

# 数据集中观察到的数量
n_observations = int(1e4)

# 布尔特征，编码为假或真
feature0 = np.random.choice([False, True], n_observations)

# 整数特征，随机从0 ..4
feature1 = np.random.randint(0, 5, n_observations)

# 字符串功能
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# 浮动特征，来自标准正态分布
feature3 = np.random.randn(n_observations)

'''
    这些特性都可以强制转换为tf。使用_bytes_feature、_float_feature、_int64_feature中
的一个作为示例兼容类型。然后我们可以创建一个tf。来自这些编码特性的示例消息。
'''


def serialize_example(feature0, feature1, feature2, feature3):
    '''
    创建一个tf.Example消息准备写入文件
    创建一个字典，将特性名称映射到与tf. example兼容的数据类型
    '''
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


'''
    例如，假设我们从数据集中有一个观察值，[False, 4, bytes('goat')， 0.9876]。我们可以创建
并打印tf。使用create_message()为该观察提供示例消息。每个单独的观察结果都将按照上面所述作为
一个特性消息来编写。注意tf。示例消息只是围绕特性消息的包装器。
'''

# 这是来自数据集的一个示例观察。
example_observation = []
serialize_example = serialize_example(False, 4, b'goat', 0.9876)
serialize_example

# 使用tf.train.Example.FromString方法解码消息
example_proto = tf.train.Example.FromString(serialized_example)
example_proto

# TFRecords格式细节
'''
    TFRecord文件包含一系列记录。该文件只能按顺序读取。
每个记录包含一个字节字符串，用于数据有效负载，加上数据长度，以及用于完整性检查的CRC32C
(使用Castagnoli多项式的32位CRC)散列。每条记录都有格式

    uint64 length
    uint32 masked_crc32_of_length
    byte   data[length]
    uint32 masked_crc32_of_data

    这些记录被连接在一起产生文件。这里描述了CRCs, CRC的掩码为
    masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul
'''

# 使用tf.dataTFRecord文件
'''
tf.data模块还提供了在tensorflow中读写数据的工具。
    编写TFRecord文件
    将数据放入数据集中最简单的方法是使用from_tensor_sections方法。应用于数组，它返回一个标量数据集
'''
tf.data.Dataset.from_tensor_slices(feature1)

# 应用于数组的元组，它返回一个元组数据集:
features_dataset = tf.data.Dataset.from_tensor_slices(
    (feature0, feature1, feature2, feature3))
features_dataset

# 使用“take(1)”只从数据集中提取一个示例。
for f0, f1, f2, f3 in features_dataset.take(1):
    print(f0)
    print(f1)
    print(f2)
    print(f3)

# 使用tf.data.Dataset。方法将函数应用于数据集的每个元素。
'''
映射函数必须在张量流图模式下操作:它必须操作并返回tf.张量。一个非张量函数，比如create_example，可以用tf封装。py_func使其兼容。
'''


def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),  # 将这些参数传递给上面的函数。
        tf.string  # 返回类型是' tf.string '。
    )
    return tf.reshape(tf_string, ())  # 结果是一个标量


tf_serialize_example(f0, f1, f2, f3)

# 将此函数应用于数据集中的每个元素:
serialize_features_dataset = features_dataset.map(tf_serialize_example)
serialize_features_dataset


def generator():
    for features in features_dataset:
        yield serialize_example(*features)


serialized_features_dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=tf.string,
    output_shapes=()
)
serialized_features_dataset

# 并写入TFRecord文件:
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
