from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

'''
下载IMDB数据集
评论文本已经转换为整数，其中每个整数表示字典中的特定单词
'''
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

print("训练数据数: {}, 标签数: {}".format(len(train_data), len(train_labels)))
print(train_labels[0])


word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    '''
    创建一个助手函数来查询包含整数到字符串映射的字典对象，返回映射后的句子。
    '''
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(train_data[0])

# 查看两数据大小
len(train_data[0]), len(train_data[1])
'''
数据预处理：
    影评的长度可能会有所不同。由于神经网络的输入必须具有相同长度。

    两种处理方式：一 热编码 占用大量内存
                 二 填充最大语料句子长度 

'''

# 使用第二种方式处理训练集和测试集
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)
# 验证是否处理成功
len(train_data[0]), len(train_data[1])

'''
搭建模型 :
    输入的词汇大小为语料库单词数
    第一层：嵌入层
    第二层：全局平均池化层
    第三层：全连接层
    第四层：全连接层后接sigmoid 输出0、1概率
'''
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型 返回loss值
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# 评估模型 返回：[loss, acc]
results = model.evaluate(test_data, test_labels)
print(results)

# loss acc 可视化
history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# loss 可视化
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# acc 可视化
plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
