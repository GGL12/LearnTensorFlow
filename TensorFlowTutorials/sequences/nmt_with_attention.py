# 机器翻译注意力机制
'''
    笔记本训练一个序列到序列(seq2seq)模型为西班牙语到英语的翻译。这是一个高级示例，假定
您对序列到序列模型有一定的了解。
    在训练了这个笔记本中的模型之后，你将能够输入一个西班牙语句子，比如
“¿todavia estan en casa?”，并返回英文翻译:"are you still at home?"
    对于一个玩具例子来说，翻译质量是合理的，但是生成的注意情节可能更有趣。这说明在翻译过程中，模型注意到了输入句子的哪些部分:
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import unicodedata
import re
import numpy as np
import os
import io
import time

# 下载并准备数据集
'''
    我们将使用http://www.manythings.org/anki/提供的语言数据集。该数据集包含以下格式的语言翻译对:
May I borrow this book?    ¿Puedo tomar prestado este libro?
    有多种可用的语言，但是我们将使用英语-西班牙语数据集。为了方便起见，我们在谷歌云上托管
了此数据集的副本，但您也可以下载自己的副本。下载数据集后，我们会采取以下步骤准备数据:
    1:在每个句子中添加一个开始和结束标记。
    2:通过删除特殊字符来清除句子。
    3:创建一个单词索引并反转单词索引(字典从单词→id和id→word映射)。
    4:把每句话都拉长。
'''
# 下载数据
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)
path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# 将unicode文件转换为ascii


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    # 在单词和其后的标点符号之间创建一个空格
    # g: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"[?.!,¿]", r"\1", w)
    w = re.sub(r'[" "]+', " ", w)
    # 用空格替换所有东西，除了(a-z, a-z， ")。,“?”,“!”“,”)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    # 在句中添加开始和结束标记,这样模型就知道什么时候开始预测，什么时候停止预测。
    w = '<start> ' + w + ' <end>'
    return w


en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

'''
1:消除口音
2:清除句子
3:返回单词对的格式:[英语，西班牙语]
'''


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8'.read().strip().split('\n'))
    word_pairs = [[preprocess_sentence(w) for w in l.split(
        '\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)


en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])


def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize(lang):
    lang_tokenizer = tf.python.keras.preprocessing.text.tokenize(filter='')
    lang_tokenizer.fit_on_text(lang)

    tensor = lang_tokenizer.text_to_sequences(lang)
    tensor = tf.python.keras.preprocessing.pad_sequences(
        tensor,
        padding='post'
    )

    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # 创建干净的输入、输出对
    targ_lang, inp_lang = create_dataset(path, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


# 限制数据集的大小以更快地进行实验(可选)
'''
在完整的>100,000个句子的数据集上进行训练需要很长时间。为了更快的训练，我们可以将数据集的大小
限制在30000个句子(当然，翻译质量会随着数据的减少而下降):
'''
# 尝试使用数据集的大小进行试验
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(
    path_to_file, num_examples)
# 计算目标张量的max_length
max_length_targ, max_length_inp = max_length(
    target_tensor), max_length(input_tensor)

# 使用80-20分割创建培训和验证集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2)
len(input_tensor_train), len(target_tensor_train), len(
    input_tensor_val), len(target_tensor_val)


def convert(lang, tensor):
    for t in tensor:
        if t != 0:
            print("%d ----> %s" % (t, lang.index_word[t]))


print("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])

# 创建 tf.data.dataset
BUFFER_SIZE = len(input_tensor_train)
BTACH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BTACH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BTACH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

# 编写编码器和解码器模型
'''
    在这里，我们将实现一个编码器-解码器模型，您可以在TensorFlow神经机器翻译(seq2seq)教程中
了解到。这个例子使用了一组最新的api。本笔记本实现了来自seq2seq教程的注意方程式。如下图所示，注意机制为每个输入单词分配一个权重，
然后解码器使用该权重来预测句子中的下一个单词。
    输入通过编码器模型，该模型给出了形状的编码器输出(batch_size、max_length、hidden_size)和形状的编码器隐藏状态(batch_size、hidden_size)。
    我们在利用巴达瑙的注意力。在写出简化形式之前，我们先来决定符号:
    
    FC = Fully connected (dense) layer
    EO = Encoder output
    H = hidden state
    X = input to the decoder

    伪码:
        score = FC(tanh(FC(EO) + FC(H)))
        attention weights = softmax(score, axis = 1)Softmax默认应用于最后一个轴，但这里我们想应用于第一个轴，因为score的形状是
    (batch_size、max_length、hidden_size)。Max_length是输入的长度。因为我们试图为每个输入分配一个权重，所以应该在那个轴上应用softmax。
        context vector = sum(attention weights * EO, axis = 1)与上面选择轴为1的原因相同
        embedding output = 他输入到解码器X是通过一个嵌入层。
        merged vector = concat(embedding output, context vector)
        然后将合并后的向量赋给GRU
'''


class Encoder(tf.python.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_nits = enc_units
        self.embedding = tf.python.keras.Embedding(vocab_size, embedding_dim)
        self.gru = tf.python.keras.layers.GRU(
            self.enc_nits,
            return_sequences=Treu,
            return_state=True,
            recurrent_activation='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_nits))


encoder = Encoder(vocab_inp_size, embedding_dim, units, BTACH_SIZE)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print('Encoder output shape: (batch size, sequence length, units) {}'.format(
    sample_output.shape))
print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))


class BahdanauAttention(tf.python.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.python.keras.layers.Dense(units)
        self.W2 = tf.python.keras.layers.Dense(units)
        self.V = tf.python.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # 我们这样做是为了执行加法来计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(
            tf.nn.tanh(
                self.W1(values) + self.W2(hidden_with_time_axis)
            )
        )

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(
    sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.python.layers.Model):
    def __init__(self,vocab_size,embedding_dim,dec_units,batch_sz):
        super(Decoder,self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.python.layers.Embedding(vocab_size,embedding_dim)
        self.gru = tf.python.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )
        self.fc = tf.python.keras.layers.Dense(vocab_size)

        #使用注意力机制
        self.attention = BahdanauAttention(self.dec_units)

    def call(self,x,hidden,enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector,attention_weights = self.attention(hidden,enc_output)
        #通过嵌入后的x形状== (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        #连接后的x形状== (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector,1),x],axis=-1)

        #将连接后的向量传递给GRU
        output,state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output,(-1,output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x,state,attention_weights

decode = Decoder(vocab_tar_size,embedding_dim,units,BTACH_SIZE)
sample_decoder_output,_,_ = decode(tf.random.uniform((64,1)),sample_hidden,sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

#定义优化器和损失函数
optimizer = tf.python.keras.optimizers.Adam()
loss_object = tf.python.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#设置检查点
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

#训练
'''
    1:通过编码器传递输入，编码器返回编码器输出和编码器隐藏状态。
    2:将编码器输出、编码器隐藏状态和解码器输入(即开始令牌)传递给解码器。
    3:解码器返回预测和解码器隐藏状态
    4:然后将解码器的隐藏状态传递回模型，利用预测结果计算损耗。
    5:使用教师强制决定下一个输入到解码器。
    6:教师强迫是将目标单词作为下一个输入传递给解码器的技术。
    7:最后一步是计算梯度，并将其应用于优化器和反向传播。
'''
def trian_step(inp,targ,enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output,enc_hidden = encoder(inp,enc_hidden)

        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BTACH_SIZE,1)

        for t in range(1,targ.shape[1]):
            #将enc_output传递给解码器
            predictions,dec_hidden,_ = decode(dec_input,dec_hidden,enc_output)

            loss += loss_function(targ[:,t],predictions)

            dec_input = tf.expand_dims(targ[:,t],1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decode.trainable_variables

    gradients = tape.gradient(loss,variables)

    optimizer.apply_gradients(zip(gradients,variables))

    return batch_loss

EPOCHS = 10
for epoch in range(EPOCHS):
    start = time.time()
    
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch,(inp,targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = trian_step(inp,targ,enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                    batch,
                                    batch_loss.numpy()))
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))  

#翻译
'''
    评估函数类似于训练循环，只是我们这里没有使用教师强制
    解码器在每个时间步长的输入是其先前的预测，以及隐藏状态和编码器的输出。
    当模型预测结束令牌时停止预测。并为每一步都储存注意力。
    对于一个输入，编码器的输出只计算一次。
'''

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ,max_length_inp))

    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.python.keras.preprocessing.sequence.pad_sequences(
        [inputs],
        maxlen=max_length_inp,
        paddint='post'
    )

    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1,units))]
    enc_out,enc_hidden = encoder(inputs,hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']],0)

    for t in range(max_length_targ):
        predictions,dec_hidden,attention_weights = decode(
            dec_input,
            dec_hidden,
            enc_out
        )

        #把注意力集中在稍后的情节上
        attention_weights = tf.reshape(predictions[0]).numpy()
        attention_plot[t] = attention_weights.numpy()

        prediction_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[prediction_id] + ' '

        if targ_lang.index_word[prediction_id] == '<end>':
            return result,sentence,attention_plot
        #预测的ID被反馈回模型
        dec_input = tf.expand_dims([prediction_id],0)

    return result,sentence,attention_plot

#用于绘制注意力权重的函数
def plot_attention(attention,sentence,preprocess_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    ax.matshow(attention,cmap='biridis')

    fontdict = {'fontsize':14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict,retation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def translate(sentence):
    result,sentence,attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence).encode('utf-8'))
    print('Predicted translation: {}'.format(result))

    attention_plot = attention_plot[:len(result.split(' ')),:len(sentence.split(' '))]
    plot_attention(attention_plot,sentence.split(' '),result.split(' '))

#恢复最新的检查点和测试
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')
