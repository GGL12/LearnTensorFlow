from __future__ import absolute_import, division, print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import feature_column
'''
本教程演示如何对结构化数据(例如CSV中的表格数据)进行分类。我们将使用Keras定义模型，将CSV中的列映射到用于训练模型的特性。本教程包含完整的代码到:
    1：使用pandas加载csv文件
    2：构建一个输入管道，使用tf.data对行进行批处理和洗牌。
    3：将CSV中的列映射到使用特征列来训练模型
    4：使用Keras构建、培训和评估模型。
'''
# 数据集
'''们将使用克利夫兰心脏病临床基金会提供的小数据集。CSV中有几百行。每一行描述一个病人，
每一列描述一个属性。我们将使用这些信息来预测患者是否患有心脏病，在这个数据集中，这是一个
二元分类任务。
'''
# 导入相关包


# 使用pandas创建dataframe格式
# 导入数据集
URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)
dataframe.head()

# 切分数据集 训练集 验证集 测试集
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), '训练集样本')
print(len(val), "验证集样本")
print(len(test), '测试集样本')

# 通过tf.data创建输入通道
'''
接下来，我们将用tf.data包装数据流。这将使我们能够使用特性列作为桥梁，将panda dataframe
中的列映射到用于训练模型的特性。如果我们处理的是一个非常大的CSV文件(大到不适合存储)，
我们将使用tf.data直接从磁盘读取的数据。
'''
# 来自panda Dataframe的数据集创建tf的实用程序方法。


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('target')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size=batch_size)
    return ds


# 使用一个小批量用于演示目的
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 理解输入管道
'''
现在我们已经创建了输入管道，让我们调用它来查看它返回的数据的格式。我们使用了一个小的批
处理大小来保持输出的可读性。
'''
for feature_batch, label_batch in train_ds.take(1):
    print("每个特征列", list(feature_batch.keys()))
    print("一个batch的age特征列", feature_batch['age'])
    print("一个batch的label", label_batch)

# 演示几种类型的特性列
'''
TensorFlow提供了许多类型的特性列。在本节中，我们将创建几种类型的特性列，并演示它们如何
从dataframe转换列。
'''
# 我们将使用此批处理演示几种类型的特性列并转化为batch数据
example_batch = next(iter(train_ds))[0]

# 创建特征列的实用方法


def demo(feature_column):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


# 数字列
'''
特性列的输出成为模型的输入(使用上面定义的演示函数，我们将能够准确地看到来自dataframe的
每一列是如何转换的)。数字列是最简单的列类型。它用于表示实值特征。当使用此列时，您的模型将
不变地从dataframe接收列值。
'''
age = feature_column.numeric_column('age')
demo(age)

# Bucketized columns
'''
通常，您不希望将数字直接输入模型，而是根据数值范围将其值划分为不同的类别。考虑代表一个
人年龄的原始数据。我们可以使用一个八进制列将年龄分成几个桶，而不是用数字列表示年龄。
注意下面的一个热值描述了每行匹配的年龄范围。
'''
age_buckets = feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
demo(age_buckets)

# 分类列 one_hot
thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible']
)
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

'''
在更复杂的数据集中，许多列都是分类的(例如字符串)。在处理分类数据时，特性列是最有价值的。
虽然在这个数据集中只有一个分类列，但是我们将使用它来演示在处理其他数据集时可以使用的几种
重要类型的特性列。
'''
# 嵌入列
'''
假设不是只有几个可能的字符串，而是每个类别有数千个(或更多)值。由于许多原因，随着类别数量的
增加，使用单一热编码训练神经网络变得不可行。我们可以使用嵌入列来克服这个限制。嵌入列不是
将数据表示为多个维度的单个热向量，而是将该数据表示为一个低维度、密集的向量，其中每个单元
格可以包含任意数字，而不仅仅是0或1。嵌入的大小(在下面的例子中是8)是一个必须调优的参数。
'''
# 关键点:当分类列有许多可能的值时，使用嵌入列是最好的。我们在这里使用一个用于演示目的，所以您有一个完整的示例，您可以在将来针对不同的数据集进行修改。
# 注意，嵌入列的输入是我们以前创建的分类列
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# 散列的特征列
'''
表示具有大量值的分类列的另一种方法是使用categorical_column_with_hash_bucket。
这个特性列计算输入的哈希值，然后选择hash_bucket_size桶中的一个对字符串进行编码。
在使用本专栏时，您不需要提供词汇表，您可以选择使hash_bucket的数量显著小于实际类别的数量，以节省空间。
'''
# 关键点:这种技术的一个重要缺点是可能会有冲突，在冲突中不同的字符串被映射到同一个bucket。实际上，无论如何，这对于某些数据集都可以很好地工作。
thal_hashed = feature_column.categorical_column_with_hash_bucket(
      'thal', hash_bucket_size=1000)
demo(feature_column.indicator_column(thal_hashed))

# 交叉功能列
'''
将特性组合成单个特性(更广为人知的是特性交叉)，使模型能够为每个特性组合学习单独的权重。
在这里，我们将创建一个新的功能，是年龄和塔尔的交叉。注意，crossed_column不会构建所有可能
组合的完整表(可能非常大)。相反，它由hashed_column支持，因此您可以选择表的大小。
'''
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
demo(feature_column.indicator_column(crossed_feature))
# 特征选择
'''
我们已经了解了如何使用几种类型的特性。现在我们将用它们来训练一个模型。本教程的目标是向您
展示处理特性列所需的完整代码(例如机制)。我们随意选择了一些列来训练下面的模型。
'''
# 关键点:如果您的目标是构建一个精确的模型，那么尝试您自己的更大的数据集，并仔细考虑哪些特性是最有意义的，以及它们应该如何表示。
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

# 创建特征层
# 既然已经定义了特性列，我们将使用DenseFeatures层将它们输入Keras模型。
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
# 在前面，我们使用了一个小的批处理大小来演示特性列是如何工作的。我们创建了一个新的具有更大批处理大小的输入管道。

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 创建编译训练模型
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
关键点:你通常会在更大更复杂的数据集中看到深度学习的最佳结果。在处理像这样的小数据集时，
我们建议使用决策树或随机森林作为强基线。本教程的目标不是训练一个精确的模型，而是演示处理
结构化数据的机制，因此在将来处理自己的数据集时，可以使用代码作为起点。
'''
