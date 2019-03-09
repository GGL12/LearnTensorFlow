from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import feature_column
'''
���̳���ʾ��ζԽṹ������(����CSV�еı������)���з��ࡣ���ǽ�ʹ��Keras����ģ�ͣ���CSV�е���ӳ�䵽����ѵ��ģ�͵����ԡ����̳̰��������Ĵ��뵽:
    1��ʹ��pandas����csv�ļ�
    2������һ������ܵ���ʹ��tf.data���н����������ϴ�ơ�
    3����CSV�е���ӳ�䵽ʹ����������ѵ��ģ��
    4��ʹ��Keras��������ѵ������ģ�͡�
'''
# ���ݼ�
'''�ǽ�ʹ�ÿ����������ಡ�ٴ�������ṩ��С���ݼ���CSV���м����С�ÿһ������һ�����ˣ�
ÿһ������һ�����ԡ����ǽ�ʹ����Щ��Ϣ��Ԥ�⻼���Ƿ������ಡ����������ݼ��У�����һ��
��Ԫ��������
'''
# ������ذ�


# ʹ��pandas����dataframe��ʽ
# �������ݼ�
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

# �з����ݼ� ѵ���� ��֤�� ���Լ�
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'ѵ��������')
print(len(val), "��֤������")
print(len(test), '���Լ�����')

# ͨ��tf.data��������ͨ��
'''
�����������ǽ���tf.data��װ���������⽫ʹ�����ܹ�ʹ����������Ϊ��������panda dataframe
�е���ӳ�䵽����ѵ��ģ�͵����ԡ�������Ǵ������һ���ǳ����CSV�ļ�(�󵽲��ʺϴ洢)��
���ǽ�ʹ��tf.dataֱ�ӴӴ��̶�ȡ�����ݡ�
'''
# ����panda Dataframe�����ݼ�����tf��ʵ�ó��򷽷���


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size=batch_size)
    return ds


# ʹ��һ��С����������ʾĿ��
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# �������ܵ�
'''
���������Ѿ�����������ܵ��������ǵ��������鿴�����ص����ݵĸ�ʽ������ʹ����һ��С����
�����С����������Ŀɶ��ԡ�
'''
for feature_batch, label_batch in train_ds.take(1):
    print("ÿ��������", list(feature_batch.keys()))
    print("һ��batch��age������", feature_batch['age'])
    print("һ��batch��label", label_batch)

# ��ʾ�������͵�������
'''
TensorFlow�ṩ��������͵������С��ڱ����У����ǽ������������͵������У�����ʾ�������
��dataframeת���С�
'''
# ���ǽ�ʹ�ô���������ʾ�������͵������в�ת��Ϊbatch����
example_batch = next(iter(train_ds))[0]

# ���������е�ʵ�÷���


def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# ������
'''
�����е������Ϊģ�͵�����(ʹ�����涨�����ʾ���������ǽ��ܹ�׼ȷ�ؿ�������dataframe��
ÿһ�������ת����)������������򵥵������͡������ڱ�ʾʵֵ��������ʹ�ô���ʱ������ģ�ͽ�
����ش�dataframe������ֵ��
'''
age = feature_column.numeric_column('age')
demo(age)

# Bucketized columns
'''
ͨ��������ϣ��������ֱ������ģ�ͣ����Ǹ�����ֵ��Χ����ֵ����Ϊ��ͬ����𡣿��Ǵ���һ��
�������ԭʼ���ݡ����ǿ���ʹ��һ���˽����н�����ֳɼ���Ͱ���������������б�ʾ���䡣
ע�������һ����ֵ������ÿ��ƥ������䷶Χ��
'''
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# ������ one_hot
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

'''
�ڸ����ӵ����ݼ��У�����ж��Ƿ����(�����ַ���)���ڴ����������ʱ�������������м�ֵ�ġ�
��Ȼ��������ݼ���ֻ��һ�������У��������ǽ�ʹ��������ʾ�ڴ����������ݼ�ʱ����ʹ�õļ���
��Ҫ���͵������С�
'''
# Ƕ����
'''
���費��ֻ�м������ܵ��ַ���������ÿ���������ǧ��(�����)ֵ���������ԭ���������������
���ӣ�ʹ�õ�һ�ȱ���ѵ���������ò����С����ǿ���ʹ��Ƕ�������˷�������ơ�Ƕ���в���
�����ݱ�ʾΪ���ά�ȵĵ��������������ǽ������ݱ�ʾΪһ����ά�ȡ��ܼ�������������ÿ����Ԫ
����԰����������֣�����������0��1��Ƕ��Ĵ�С(���������������8)��һ��������ŵĲ�����
'''
# �ؼ���:���������������ܵ�ֵʱ��ʹ��Ƕ��������õġ�����������ʹ��һ��������ʾĿ�ģ���������һ��������ʾ�����������ڽ�����Բ�ͬ�����ݼ������޸ġ�
# ע�⣬Ƕ���е�������������ǰ�����ķ�����
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# ɢ�е�������
'''
��ʾ���д���ֵ�ķ����е���һ�ַ�����ʹ��categorical_column_with_hash_bucket��
��������м�������Ĺ�ϣֵ��Ȼ��ѡ��hash_bucket_sizeͰ�е�һ�����ַ������б��롣
��ʹ�ñ�ר��ʱ��������Ҫ�ṩ�ʻ��������ѡ��ʹhash_bucket����������С��ʵ�������������Խ�ʡ�ռ䡣
'''
# �ؼ���:���ּ�����һ����Ҫȱ���ǿ��ܻ��г�ͻ���ڳ�ͻ�в�ͬ���ַ�����ӳ�䵽ͬһ��bucket��ʵ���ϣ�������Σ������ĳЩ���ݼ������Ժܺõع�����
thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

# ���湦����
'''
��������ϳɵ�������(����Ϊ��֪�������Խ���)��ʹģ���ܹ�Ϊÿ���������ѧϰ������Ȩ�ء�
��������ǽ�����һ���µĹ��ܣ�������������Ľ��档ע�⣬crossed_column���ṹ�����п���
��ϵ�������(���ܷǳ���)���෴������hashed_column֧�֣����������ѡ���Ĵ�С��
'''
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))
# ����ѡ��
'''
�����Ѿ��˽������ʹ�ü������͵����ԡ��������ǽ���������ѵ��һ��ģ�͡����̵̳�Ŀ��������
չʾ�����������������������(�������)����������ѡ����һЩ����ѵ�������ģ�͡�
'''
# �ؼ���:�������Ŀ���ǹ���һ����ȷ��ģ�ͣ���ô�������Լ��ĸ�������ݼ�������ϸ������Щ��������������ģ��Լ�����Ӧ����α�ʾ��
feature_columns = []
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# bucketized cols
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# indicator cols
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# embedding cols
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# crossed cols
crossed_feature = feature_column.crossed_column(
    [age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

# ����������
# ��Ȼ�Ѿ������������У����ǽ�ʹ��DenseFeatures�㽫��������Kerasģ�͡�
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# ��ǰ�棬����ʹ����һ��С���������С����ʾ����������ι����ġ����Ǵ�����һ���µľ��и����������С������ܵ���

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# ��������ѵ��ģ��
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='relu'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)
loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
'''
�ؼ���:��ͨ�����ڸ�������ӵ����ݼ��п������ѧϰ����ѽ�����ڴ�����������С���ݼ�ʱ��
���ǽ���ʹ�þ����������ɭ����Ϊǿ���ߡ����̵̳�Ŀ�겻��ѵ��һ����ȷ��ģ�ͣ�������ʾ����
�ṹ�����ݵĻ��ƣ�����ڽ��������Լ������ݼ�ʱ������ʹ�ô�����Ϊ��㡣
'''
