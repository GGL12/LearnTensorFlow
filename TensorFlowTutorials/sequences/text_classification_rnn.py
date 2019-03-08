# ʹ��RNN�����ı�����
# ���ı�����̳���IMDB����Ӱ�����ݼ���ѵ��һ���ݹ��������������������
# ������ذ�
from __future__ import absolute_import, division, print_function
import tensorflow_datasets as tfds
import tensorflow as tf

# ����matplotlib������һ�����ֺ���������ͼ��:
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel(string)
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# ��������ܵ�
'''
IMDB����Ӱ�����ݼ���һ����Ԫ�������ݼ�������Ӱ�������������������
ʹ��TFDS�������ݼ���dataset����һ�����õ��ӵ��ʱ������
'''
dataset, info = tfds.load(name="imdb_reviews/subwords8k", with_info=True,
                          as_supervised=True)

train_dataset, test_dataset = dataset['train'], dataset['test']
# ��Ϊ����һ���ӵ��ʼǺŸ����������������Դ����κ��ַ������ǺŸ��������ǺŻ�����
tokenizer = info.features['test'].encoder
# �ʻ���
print('Vocabulary size: {}'.format(tokenizer.vocab_size))
# ����
sample_string = 'TensorFlow is cool.'
tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
original_string = tokenizer.decode(tokenized_string)
print('The original string: {}'.format(original_string))
# ����ַ����������ֵ��У��ǺŸ���������ֽ�Ϊ�ӵ��ʣ��Ӷ����ַ������б���
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

#BUFFER_SIZE = 10000
BATCH_SIZE = 64
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(
    BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(
    BATCH_SIZE, test_dataset.output_shapes)

# ģ�ʹ���
'''
����һ��tf.keras��˳��ģ�ͣ�����һ��Ƕ��㿪ʼ��Ƕ���Ϊÿ�����ʴ洢һ��������
����ʱ����������������ת��Ϊ�������С���Щ�����ǿ�ѵ���ġ�����(���㹻������)��ѵ����
�������ƺ���ĵ��������������Ƶ�����
�����������ұ�ͨ��tf.keras.layers����һ���ȱ��������ĵ�Ч����Ҫ��Ч�öࡣ
�ݹ�������(RNN)ͨ������Ԫ���������������롣RNNs�������һ��ʱ�䲽���ݵ����ǵ����롪Ȼ�󴫵ݵ���һ��ʱ�䲽��
tf.keras.layers��˫���װ��Ҳ������RNN��һ��ʹ�á���ͨ��RNN����ǰ����󴫲����룬Ȼ�������������������RNNѧϰ����������ϵ��
'''
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
# ����ģ��
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# ѵ��ģ��
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset)
# ��ӡ���Լ�
test_loss, test_acc = model.evaluate(test_dataset)
print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# �����ģ��û������Ӧ�������е���䡣������Ƕ�������н���ѵ��������δ������н��в��ԣ��ͻᵼ��ƫб�ȡ���������£�ģ��Ӧ��ѧ�������䣬���������������濴���ģ����������Ӱ��ȷʵ��С��
# ���Ԥ����>= 0.5����Ϊ��������Ϊ����


def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec


def sample_predict(sentence, pad):
    # ����һ��Ԥ�⺯��
    tokenized_sample_pred_text = tokenizer.encode(sample_pred_text)

    if pad:
        tokenized_sample_pred_text = pad_to_size(
            tokenized_sample_pred_text, 64)

    predictions = model.predict(tf.expand_dims(tokenized_sample_pred_text, 0))

    return (predictions)


sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
# ��û������ʾ���ı�����Ԥ��
predictions = sample_predict(sample_pred_text, pad=False)
print(predictions)
# Ԥ���������ʾ���ı�
sample_pred_text = ('The movie was cool. The animation and the graphics '
                    'were out of this world. I would recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)

# ������ȷ��
plot_graphs(history, 'accuracy')
# ������ʧֵ
# plot_graphs(history, 'loss')


# �Ľ�ģ����Ӷ��LSTM��
'''
Kerasѭ�������������õ�ģʽ����return_sequence���캯����������:
    1������ÿ��ʱ�䲽�������������������(��״����ά����(batch_size, timesteps, output_features))��
    2������ÿ���������е����һ�����(��״�Ķ�ά����(batch_size, output_features))��
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

# Ԥ���������ʾ���ı�
sample_pred_text = ('The movie was not good. The animation and the graphics '
                    'were terrible. I would not recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print(predictions)
# ������ȷ��
plot_graphs(history, 'accuracy')
# ������ʧֵ
plot_graphs(history, 'loss')
