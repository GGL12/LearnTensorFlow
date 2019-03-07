# 本教程演示如何使用基于字符的RNN生成文本。
# 导入TensorFlow和其他库
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import time

# 载莎士比亚资料集
path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# 数据读取
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# 本的长度是其中的字符数
print('Length of text: {} characters'.format(len(text)))
print(text[:250])
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))

# 处理文本
# 在训练之前，我们需要将字符串映射到数字表示。创建两个查询表:一个将字符映射到数字，另一个将数字映射到字符。
# 从唯一字符到索引的映射
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
# 现在每个字符都有一个整数表示。注意，我们将字符映射为索引，从0映射到len(unique)。
print('{')
for char, _ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
# 显示文本的前13个字符如何映射到整数
print(
    '{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 预测的任务
'''
给定一个字符或一组字符，下一个最有可能的字符是什么?这就是我们正在训练模型执行的任务。该模型的输入
将是一个字符序列，我们训练该模型在每个时间步预测输出—如下所示的字符。
由于RNNs维护一个依赖于前面看到的元素的内部状态，给定到目前为止计算的所有字符，下一个字符是什么?
'''
# 创建训练示例和目标
'''
接下来将文本分成示例序列。每个输入序列将包含文本中的seq_length字符。
对于每个输入序列，对应的目标包含相同长度的文本，只是向右移动了一个字符。
因此，将文本分成seq_length+1的块。例如，假设seq_length是4，我们的文本是“Hello”。输入序列为“Hell”，目标序列为“ello”。
为此，首先使用tf.data. data. from_tensor_sections函数将文本向量转换为字符索引流。
'''
# 我们希望单个字符输入的最大长度句
seq_length = 100
examples_per_epoch = len(text)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for i in char_dataset.take(5):
    print(idx2char[i.numpy()])
# 批处理方法允许我们轻松地将这些单个字符转换为所需大小的序列。
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))
# 对于每个序列，使用map方法对每个批次应用一个简单的函数，复制并移动它，形成输入和目标文本:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)
# 打印第一个例子输入和目标值:
for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
# 这些向量的每个指标都作为一个时间步长处理。对于第0步的输入，模型接收“F”的索引，并尝试预测下一个字符是“i”的索引。在下一个timestep中，它做同样的事情，但是RNN除了考虑当前输入字符外，还考虑前面的步骤上下文。
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(
        target_idx, repr(idx2char[target_idx])))

# 创建训练批次
'''
我们使用tf.data将文本分割为可管理序列的数据。但是在将这些数据输入模型之前，我们需要重新
排列数据并将其打包成batch data。
'''
BATCH_SIZE = 64
'''
缓冲大小来洗牌数据集
TF数据被设计用来处理可能无限的序列，
所以它不会试图在内存中打乱整个序列。相反,
它维护一个缓冲区，在这个缓冲区中它打乱元素的顺序)。
'''
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset

# 搭建模型
'''
使用tf.keras.Sequential顺序地定义模型。对于这个简单的例子，我们使用了三层来定义我们的模型:
    tf.keras.layers.Embedding:输入层。一个可训练的查找表，它将每个字符的数字映射到一个具有embedding_dim维数的向量;
    tf.keras.layers.GRU: 一种尺寸为units=rnn_units的RNN类型(这里还可以使用LSTM层)。
    tf.keras.layers.Dense:输出层，带有vocab_size输出。
'''
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# 尝试一下模型
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape,
          "# (batch_size, sequence_length, vocab_size)")

model.summary()

'''
为了从模型中得到实际的预测，我们需要从输出分布中取样，得到实际的字符索引。这种分布是由字符
词汇表上的日志定义的。
'''
sampled_indices = tf.random.categorical(
    example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# 这给了我们，在每一个时间步，预测下一个字符索引:
sampled_indices

# 训练模型
'''
此时，该问题可以作为一个标准分类问题来处理。给定前面的RNN状态和这个时间步长的输入，预测下一个字符的类。
添加优化器和损失函数
    标准tf.keras.losses。在这种情况下，tf.keras.losses.sparse_softmax_crossentropy损失函数有效，因为它应用于预测的最后一个维度。
    因为我们的模型返回logits，所以需要设置from_logits标志。
'''


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape,
      " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())

# 编译模型
model.compile(optimizer='adam', loss=loss)

# 配置的检查点
# 用tf.keras.callbacks.ModelCheckpoint确保在培训过程中保存检查点:

# 将保存检查点的目录
checkpoint_dir = './training_checkpoints'
# 检查点文件的名称
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
# 执行训练
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# 生成文本
'''
恢复最新的检查点
    要保持此预测步骤的简单性，请使用batch大小为1的批处理。
    由于RNN状态从一个时间步传递到另一个时间步的方式，模型只接受构建后的固定batch大小。
    运行具有不同batch_size的模型，我们需要重新构建模型并从检查点恢复权重。
'''
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

# 预测
'''
1.它首先选择一个开始字符串，初始化RNN状态并设置要生成的字符数。
2.使用开始字符串和RNN状态获取下一个字符的预测分布。
3.然后，使用分类分布计算预测字符的索引。使用这个预测字符作为模型的下一个输入。
4.模型返回的RNN状态被反馈回模型，这样它就有了更多的上下文，而不是只有一个单词。在预测
下一个单词之后，修改后的RNN状态再次反馈到模型中，当它从之前预测的单词中获得更多上下文时，它就是这样学习的。
'''
# 查看生成的文本，您将看到该模型知道什么时候应该大写、生成段落并模仿莎士比亚式的写作词汇。由于训练的时代不多，它还没有学会形成连贯的句子。


def generate_text(model, start_string):
    # 评估步骤(使用学习模型生成文本)
    # 要生成的字符数
    num_generate = 1000
    # 将开始字符串转换为数字(向量化)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    # 空列表储存我们的结果
    text_generated = []
    '''
    lower temperature会产生更容易预测的文本。
    highertemperature会产生更多令人惊讶的文字。
    实验找出最佳设置。
    '''
    temperature = 1.0

    # batch size = 1
    model.reset_states()
    for i in range(num_generate):
        predictions = predictions / temperature
        prediction_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()

        # 我们将预测的单词作为下一个输入传递给模型以及之前的隐藏状态
        input_eval = tf.expand_dims([prediction_id], 0)
        text_generated.append(idx2char[prediction_id])

    return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"ROMEO: "))

# 高级:自定义训练
'''
上面的训练过程很简单，但是不能给你太多的控制。既然您已经了解了如何手动运行模型，那么让我们打开训练循环，
并自己实现它。例如，如果要实现课程学习以帮助稳定模型的开环输出，这就提供了一个起点。
我们将使用tf.GradientTape
    程序如下:
1.首先，初始化RNN状态。我们通过调用tf.keras.Model.reset_states来实现这一点。
2.下来，遍历数据集(逐批)并计算与每个数据集关联的预测。
3.打开一个tf.GradientTape并计算在那种情况下的预测和损失，。
4.使用tf.GradientTape计算损失相对于模型变量的梯度。
5.最后，使用优化器的tf.train.Optimizer.apply_gradients方法。
'''
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()


def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions)
        )
    gards = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gards, model.trainable_variables))

    return loss


EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(int, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch+1, batch_n, loss))

    # 每隔五个epoch存储模型
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print("Epoch {} Loss {:.4f}".format(epoch+1, loss))
    print("time taken for 1 epoch {} sec\n".format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))
