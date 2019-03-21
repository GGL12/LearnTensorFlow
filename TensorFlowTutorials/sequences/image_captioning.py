# 图像字幕
'''
    给定如下的图片，我们的目标是生成一个标题，例如“一个冲浪运动员骑在波浪上”。
    这里，我们将使用一个基于注意力的模型。这使我们能够看到模型在生成标题时所关注的图像的哪些部分。
'''
from __future__ import division, absolute_import, print_function
import tensorflow as tf
# 我们将生成注意力图，以查看图像的哪些部分
# 我们的模型专注于标题期间
import matplotlib.pyplot as plt
# Scikit-learn包括许多有用的实用程序
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import numpy as np
import os
import json
from glob import glob
from PIL import Image
import pickle

# 下载并准备MS-COCO数据集
'''
我们将使用MS-COCO数据集来训练我们的模型。这个数据集包含了>82,000个图像，每个图像都至少有5个不同的注释。下面的代码将自动下载并提取数据集。
注意:前面有大量的下载。我们将使用训练集，它是一个13GB的文件。
'''
annotation_zip = tf.keras.utils.get_file('captions.zip',
                                         cache_subdir=os.path.abspath('.'),
                                         origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                         extract=True)
annotation_file = os.path.dirname(
    annotation_zip)+'/annotations/captions_train2014.json'
name_of_zip = 'train2014.zip'
if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
    image_zip = tf.keras.utils.get_file(name_of_zip,
                                        cache_subdir=os.path.abspath('.'),
                                        origin='http://images.cocodataset.org/zips/train2014.zip',
                                        extract=True)
    PATH = os.path.dirname(image_zip)+'/train2014/'
else:
    PATH = os.path.abspath('.')+'/train2014/'


# 可以选择限制训练集的大小，以便更快地进行训练
'''
对于本例，我们将选择30,000个标题的子集，并使用这些标题和相应的图像来训练我们的模型。
与往常一样，如果您选择使用更多的数据，标题质量将会提高。
'''
# 读取json的文件
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# 在向量中存储标题和图像名称
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = "<start>" + annot['caption'] + '<end>'
    image_id = annot['image_id']
    full_coco_iamge_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_iamge_path)
    all_captions.append(caption)

# captions和image_names混在一起设置随机状态
train_captions, img_name_vector = shuffle(
    all_captions,
    all_img_name_vector,
    random_state=1
)
# 从洗牌集中选择前30000个标题
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

len(train_captions), len(all_captions)

# 使用InceptionV3对图像进行预处理
'''
接下来，我们将使用InceptionV3(在Imagenet上预先训练)对每个图像进行分类。我们将从最后一个卷积层中提取特征。
首先，我们需要将图像转换成inceptionV3期望的格式:
    1:将映像大小调整为(299,299)
    2:使用preprocess_input方法将像素放置在-1到1的范围内(以匹配用于训练InceptionV3的图像的格式)。
'''


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.Inception_v3.preprocess_input(img)
    return img, image_path


# 初始化InceptionV3并加载预训练的Imagenet权重
'''
    为此，我们将创建一个tf。keras模型，其中输出层是InceptionV3体系结构中的最后一个卷积层。
每个图像都通过网络转发，最后得到的向量存储在字典中(image_name—> feature_vector)。
我们使用最后一个卷积层因为我们在这个例子中使用了注意力。该层输出的形状为8x8x2048。
我们在训练中避免这样做，这样就不会成为瓶颈。
在所有图像通过网络之后，我们对字典进行pickle并将其保存到磁盘。
'''
image_model = tf.keras.applications.Inception_v3(
    include_top=False,
    weights='imagenet'
)
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

iamge_features_extract_model = tf.keras.Model(
    new_input,
    hidden_layer
)

# 缓存从InceptionV3中提取的特性
'''
    我们将使用InceptionV3对每个图像进行预处理，并将输出缓存到磁盘。将输出缓存到RAM中会
更快，但需要占用内存，每个映像需要8 * 8 * 2048个浮点数。在编写本文时，这将超过Colab的内存限制
(尽管这些限制可能会发生变化，但是一个实例目前似乎有大约12GB的内存)。
使用更复杂的缓存策略(例如，通过分片图像来减少随机访问磁盘I/O)可以提高性能，但代价是需要
更多的代码。
'''

# 得到唯一的图像
encode_train = sorted(set(img_name_vector))
# 根据您的系统配置随意更改batch_size
iamge_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
iamge_dataset = iamge_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE.batch(16)
)


for img, path in iamge_dataset:
    batch_features = iamge_features_extract_model(img)
    batch_features = tf.reshape(
        batch_features,
        (batch_features.shape[0], -1, batch_features.shape[3])
    )
    for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

# 对标题进行预处理和标记
'''
首先，我们将标记标题(例如，通过分隔空格)。这将为我们提供数据中所有独特单词的词汇表(如“surfing”、“football”等)。
接下来，我们将把词汇量限制在前5000个单词以内，以节省内存。我们将用标记“UNK”(表示未知)替换所有其他单词。
最后，我们创建一个单词——>索引映射，反之亦然。
然后我们将所有序列填充为与最长序列相同的长度。
'''
# 这将找到数据集中任何标题的最大长度


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# 上面的步骤是处理文本处理的一般过程
# 从词汇表中选择前5000个单词
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=top_k,
    oov_token='<unk>',
    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
)
tokenizer.fit_on_texts(train_captions)
train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# 创建标记向量
train_seqs = tokenizer.texts_to_sequences(train_captions)

# 将每个向量填充到标题的max_length如果没有提供max_length参数，那么pad_sequences会自动计算这个值
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
    train_seqs,
    padding='post'
)

# 计算max_length
# 用于存储注意力权重
max_lenght = calc_max_length(train_seqs)

# 将数据分解为训练和测试

# 使用80-20分割创建培训和验证集、
img_name_train, img_name_val, cap_train, cap_val = train_test_split(
    img_name_vector,
    cap_vector,
    test_size=0.2,
    random_state=0
)
len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

# 我们的图片和说明已经准备好了!接下来，让我们创建一个tf.data用于训练模型

'''
根据您的系统配置随意更改这些参数
'''
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
# 从InceptionV3中提取的向量的形状为(64,2048)这两个变量表示这个
features_shape = 2048
attention_features_shape = 64

# 加载文件


def map_func(img_name, cap):
    ima_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return ima_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# 使用map并行加载numpy文件
dataset = dataset.map(lambda iterm1, iterm2: tf.numpy_function(
    map_func, [iterm1, iterm2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.exprimental.AUTOTUNE
)

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# 模型
'''
有趣的是，下面的解码器与神经机器翻译示例中的解码器是相同的。
    1:在本例中，我们从InceptionV3的下卷积层提取特征，得到一个形状向量(8,8,2048)。
    2:我们把它压缩成(64,2048)的形状。
    3:然后这个向量通过CNN编码器(它由一个完全连接的层组成)传递。
    4:RNN(这里的GRU)负责预测下一个单词。
'''


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        '''
        feature (CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        hidden shape == (batch_size, hidden_size)
        hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        '''
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, 64, 1)
        # 最后一个轴是1因为我们要给self.V赋值
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    '''
    因为我们已经提取了这些特性，并使用pickle将其丢弃
    这个编码器通过一个完全连接的层来传递这些特性
    '''

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder).__init__()

        self.units = units
        self.embedding = tf.keras.layers.Embedding_dim(
            vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # 将注意力定义为一个单独的模型
        context_vector, attention_weights = self.attention(features, hidden)
        #通过嵌入后的x形状== (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        #连接后的x形状== (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=1)
        #shape == (batch_size, max_length, hidden_size)
        output, state = self.gru(x)

        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        x = self.fc2(x)
        # output shape == (batch_size * max_length, vocab)
        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros(batch_size, self.units)

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim,units,vocab_size)

optimizer = tf.keras.optimizer.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

#Checkpoint
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(
    encoder=encoder,
    decoder = decoder,
    optimizer = optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt,checkpoint_path,max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

#训练
'''
1:我们提取各自.npy文件中存储的特性，然后通过编码器传递这些特性。
2:将编码器输出、隐藏状态(初始化为0)和解码器输入(即开始令牌)传递给解码器。
3:解码器返回预测和解码器隐藏状态。
4:然后将解码器的隐藏状态传递回模型，利用预测结果计算损耗。
5:使用教师强制决定下一个输入到解码器
6:教师强迫是将目标单词作为下一个输入传递给解码器的技术。
7:最后一步是计算梯度，并将其应用于优化器和反向传播。
'''

#将其添加到单独的单元格中，因为如果运行训练单元格很多时候，会重置loss_plot数组
loss_plot = []
def train_step(img_tensor,target):
    loss = 0

    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        #初始化每个批处理的隐藏状态因为标题之间没有关联
        for i in range(1,target.shape[1]):
            #通过解码器传递特性
            predictions,hidden,_ = decoder(dec_input,features,hidden)

            loss += loss_function(target[:,i],predictions)
            #使用老师强迫v
            dec_input = tf.expand_dims(target[:,i],1)
    
    total_loss = (loss/ int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss,trainable_variables)

    optimizer.apply_gradients(zip(gradients,trainable_variables))

    return loss,total_loss

    EPOCHES = 20
    for epoch in range(start_epoch,EPOCHES):
        start = time.time()
        total_loss = 0
        for (batch,(img_tensor,target)) in enumerate(dataset):
            batch_loss,t_loss = train_step(img_tensor,target)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    #存储epoch结束损失值，以便稍后绘图
    loss_plot.append(total_loss / num_steps)
    if epoch %5 == 0:
        ckpt_manager.save()
    print ('Epoch {} Loss {:.6f}'.format(epoch + 1, 
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

plt.plot(loss_plot)
plt.xlabel('EPOCHS')
plt.ylabel("Loss")
plt.title("Loss Plot")
plt.show()

#加上说明
'''
1:评估函数类似于训练循环，只是我们这里没有使用教师强制。解码器在每个时间步长的输入是其先前的预测，以及隐藏状态和编码器的输出。
2:当模型预测结束令牌时停止预测。
3:并为每一步都储存注意力。
'''
def evalute(image):
    attention_plot = np.zeros((max_lenght,attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    
    temp_input = tf.expand_dims(load_image(image)[0],0)
    img_tensor_val = iamge_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val,(img_tensor_val.shape[0],-1,img_tensor_val[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']],0)
    result = []

    for i in range(max_lenght):
        predictions,hidden,attention_weights = decoder(dec_input,features,hidden)

        attention_plot[i] = tf.reshape(attention_weights,(-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.word_index[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result,attention_plot
        dec_input = tf.expand_dims([predicted_id],0)

    attention_plot = attention_plot[:len(result),:]
    return result,attention_plot

def plot_attention(image,result,attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10,10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i],(8.8))
        ax = fig.add_subplot(len_result//2,len_result//2,i+1)
        ax.set_title(result[1])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att,cmap='gray',alpha=0.6,extent=img.get_extent())
    plt.tight_layout()
    plt.show()

#验证集上的标题
rid = np.random.randint(0,len(img_name_val))
image = img_name_val[rid]
real_cation = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result,attention_plot = evalute(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
#打开图像
Image.open(img_name_val[rid])


#在你自己的图片上试试
'''
有趣的是，下面我们提供了一个方法，您可以使用我们刚刚训练的模型为您自己的图像添加标题。
请记住，它是基于相对较少的数据进行训练的，您的图像可能与训练数据不同(所以要准备好接受
奇怪的结果!)
'''
image_url = 'https://tensorflow.org/images/surf.jpg'
image_extension = image_url[-4:]
image_path = tf.keras.utils.get_file('image'+image_extension, 
                                     origin=image_url)

result, attention_plot = evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image_path, result, attention_plot)
Image.open(image_path)