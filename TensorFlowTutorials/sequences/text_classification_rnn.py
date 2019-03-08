# 使用RNN进行文本分类
# 本文本分类教程在IMDB大型影评数据集上训练一个递归神经网络进行情绪分析。
# 导入相关包
from __future__ import absolute_import, division, print_function
import tensorflow_datasets as tfds
import tensorflow as tf

# 导入matplotlib并创建一个助手函数来绘制图形:
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel(string)
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# 设置输入管道
'''
IMDB大型影评数据集是一个二元分类数据集，所有影评都有正面或负面情绪。
使用TFDS下载数据集。dataset附带一个内置的子单词标记器。
'''
dataset, info = tfds.load(name="imdb_reviews/subwords8k", with_info=True,
                          as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
# 因为这是一个子单词记号赋予器，所以它可以传递任何字符串，记号赋予器将记号化它。
tokenizer = info.features['test'].encoder
# 词汇量
print('Vocabulary size: {}'.format(tokenizer.vocab_size))
# 样例
sample_string = 'TensorFlow is cool.'
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))
# 如果字符串不在其字典中，记号赋予器将其分解为子单词，从而对字符串进行编码
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

#BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(
    BATCH_SIZE, test_dataset.output_shapes)

# 模型创建
'''
建立一个tf.keras。顺序模型，并从一个嵌入层开始。嵌入层为每个单词存储一个向量。
调用时，它将字索引序列转换为向量序列。这些向量是可训练的。经过(对足够的数据)的训练，
具有相似含义的单词往往具有相似的向量
这种索引查找比通过tf.keras.layers传递一个热编码向量的等效操作要有效得多。
递归神经网络(RNN)通过遍历元素来处理序列输入。RNNs将输出从一个时间步传递到它们的输入—然后传递到下一个时间步。
tf.keras.layers。双向包装器也可以与RNN层一起使用。它通过RNN层向前和向后传播输入，然后连接输出。这有助于RNN学习长期依赖关系。
'''
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 训练模型
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)
# 打印测试集
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# 上面的模型没有屏蔽应用于序列的填充。如果我们对填充序列进行训练，并对未填充序列进行测试，就会导致偏斜度。理想情况下，模型应该学会忽略填充，但是正如您在下面看到的，它对输出的影响确实很小。
# 如果预测是>= 0.5，则为正，否则为负。


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    # 定义一个预测函数
    tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)

    if pad:
        tokenized_sample_pred_text = pad_to_size(
            tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return (predictions)


sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
# 对没有填充的示例文本进行预测
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)
# 预测带有填充的示例文本
sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

# 绘制正确率
plot_graphs(history, 'accuracy')
# 绘制损失值
# plot_graphs(history, 'loss')


# 改进模型添加多个LSTM层
'''
Keras循环层有两个可用的模式，由return_sequence构造函数参数控制:
    1：返回每个时间步的连续输出的完整序列(形状的三维张量(batch_size, timesteps, output_features))。
    2：返回每个输入序列的最后一个输出(形状的二维张量(batch_size, output_features))。
'''


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)

test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# 预测带有填充的示例文本
sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)
# 绘制正确率
plot_graphs(history, 'accuracy')
# 绘制损失值
plot_graphs(history, 'loss')
