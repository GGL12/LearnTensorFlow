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
