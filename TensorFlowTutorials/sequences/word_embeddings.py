# 词嵌入
# 教程介绍word嵌入。它包含完整的代码，可以在一个小数据集中从零开始训练word嵌入
# 将文本表示为数字
import matplotlib.pyplot as plt
'''
机器学习模型以向量(数字数组)作为输入。在处理文本时，我们必须首先想出一个策略，将字符串转换为数字(或将文本“向量化”)，然后再将其提供给模型。在本节中，我们将研究三种策略。
1:One-hot
    首先，我们可以用“one-hot”对词汇表中的每个单词进行编码。想想“the cat sat on the mat”
这句话。这个句子中的词汇(或独特的单词)是(cat, mat, on, sat, The)。为了表示每个单词，我们将创建一个长度等于词汇表的零向量，然后在对应单词的索引中放置一个1
要创建包含句子编码的向量，我们可以将每个单词的一个热向量连接起来。
关键点:这种方法是低效的。一个热编码的向量是稀疏的(意思是，大多数指标是零)。假设我们有10000个单词。
要对每个单词进行一次热编码，我们将创建一个向量，其中99.99%的元素为零。

2:用唯一的数字编码每个单词
    我们可能尝试的第二种方法是使用唯一的数字编码每个单词。继续上面的例子，我们可以将1赋值给“cat”，将2赋值给“mat”，依此类推。然后我们可以把“猫坐在垫子上”这句话编码成
一个密集的向量，比如[5,1,4,3,5,2]。这种方法很有效。我们现在有一个稠密的向量(所有元素都是满的)，而不是稀疏的向量。
然而，这种方法有两个缺点:
2.1:整数编码是任意的(它不捕获单词之间的任何关系)。
2.2:对于模型来说，整数编码的解释是很有挑战性的。例如，线性分类器为每个特征学习单个权重。因为不同的单词可能具有相似的编码，所以这种特征权重组合没有意义。

3:词嵌入
    词嵌入为我们提供了一种使用高效、密集表示的方法，其中相似的单词具有相似的编码。重要的是，我们不必手工指定这种编码。嵌入是浮点值的密集
    向量(向量的长度是您指定的参数)。它们不是手工指定嵌入的值，而是可训练参数(模型在训练期间学习的权重
    ，与模型学习密集层的权重的方法相同)。常见的情况是，word嵌入为8维(对于小数据集)，工作时可达到1024维
'''

# 使用嵌入层
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

'''
嵌入层至少需要两个参数:
    词汇表中可能出现的单词数量，这里是1000(1 +最大单词索引)，
    嵌入的维数，这里是32。
'''
embedding_layer = layers.Embedding(1000, 32)
'''
    嵌入层可以理解为一个查询表，它从整数索引(表示特定的单词)映射到密集向量(它们的嵌入)。嵌入的维数(或宽度)是一个参数，
您可以用它进行实验，看看什么对您的问题有效，这与您在一个密集层中对神经元数量进行实验的方法非常相似。

    当您创建一个嵌入层时，嵌入的权重是随机初始化的(就像任何其他层一样)。在训练过程中，通过反向
传播逐步调整。一旦经过训练，所学习的单词嵌入将大致编码单词之间的相似性(因为它们是针对您的
模型所训练的特定问题而学习的)。

    作为输入，嵌入层采用一个形状(样本，sequence_length)的整数二维张量，其中每个条目都是整数序列。它可以嵌入可变长度的序列。
您可以将形状(32、10)(长度为10的32个序列的批次)或(64、15)(长度为15的64个序列的批次)导入上述批次的嵌入层。批处理中的所有序列
必须具有相同的长度，因此较短的序列应该用零填充，较长的序列应该被截断

    作为输出，嵌入层返回一个形状(sample, sequence_length, embedding_dimension)的三维浮点张量。
这样一个三维张量可以由一个RNN层来处理，也可以简单地由一个致密层来处理。我们将在本教程中展示第一种方法，您可以使用RNN参考文本分类来学习第二种方法。
'''

# 从头开始学习嵌入
vocab_size = 10000
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=vocab_size)

print(train_data[0])

# 将整数转换回单词
# 知道如何将整数转换回文本可能是有用的。这里，我们将创建一个helper函数来查询一个dictionary对象，该对象包含从整数到字符串的映射:
# 将单词映射到整数索引的字典
word_index = imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.geit(i, '?') for i in text])


# 影评论可以有不同的长度。我们将使用pad_sequences函数来标准化评审的长度。
maxlen = 500
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=maxlen
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=maxlen
)

print(train_data[0])

# 创建一个简单的模型
'''
    1:第一层是嵌入层。这一层使用整数编码的词汇表，并为每个单词索引查找嵌入向量。这些向量作为模型火车来学习。这些向量向输出数组添加一个维度。得到的维度是:'(批量、顺序、嵌入)' '。
    2:接下来，GlobalAveragePooling1D层通过对序列维数进行平均，为每个示例返回一个固定长度的输出向量。这允许模型以最简单的方式处理可变长度的输入。
    3:这个固定长度的输出向量通过一个有16个隐藏单元的全连接(密集)层来传输。
    4:最后一层与单个输出节点紧密连接。使用sigmoid激活函数，这个值是0到1之间的浮点数，表示评审结果为正的概率(或置信水平)。
'''

embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

# 编译和训练模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    train_labels,

    epochs=30,
    batch_size=512,
    validation_split=0.2
)

# 通过这种方法，我们的模型达到了88%左右的验证精度(注意，模型是过度拟合的，训练精度明显更高)。
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Train acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.figure(figsize=(16, 9))
plt.show()

# 检索已学习的嵌入
# 接下来，让我们检索在培训期间学到的单词embeddings。这将是一个形状矩阵(vocab_size, embedded -dimension)。
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

# 现在我们将把权重写到磁盘上。为了使用嵌入投影仪，我们将以tab分隔的格式上传两个文件:一个矢量文件(包含嵌入)和一个元数据文件(包含单词)。
out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for word_num in range(vocab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
