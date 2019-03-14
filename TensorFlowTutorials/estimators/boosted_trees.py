# 如何在TensorFlow中训练提升树模型
from __future__ import absolute_import, division, print_function
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from IPython.display import clear_output
'''
    本教程是使用带有tf.estimator API的决策树训练梯度增强模型的端到端演练。增强树模型是最
流行和最有效的机器学习方法的回归和分类。它是一种集成技术，结合了来自几个树模型的预测(想想10s、100甚至1000s)。
    增强树模型受到许多机器学习实践者的欢迎，因为它们可以通过最小的超参数调优实现令人印象深刻的性能。
'''

# 加载泰坦尼克数据集
'''
您将使用泰坦尼克数据集，其中目标是预测乘客的生存，给定的特征，如性别、年龄、阶级等。
'''

# 加载数据
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

tf.random.set_seed(123)

# 探索数据
'''
让我们先预览一些数据，并创建关于训练集的汇总统计信息。
'''
dftrain.head()

dftrain.describe()

# 下面是627个示例和264个示例，分别位于培训集和评估集中。
dftrain.shape[0], dfeval.shape[0]

# 大多数乘客都是二三十岁。
dftrain.age.hist(bins=20)

# 船上的男性乘客大约是女性乘客的两倍。
dftrain.sex.value_counts().plot(kind='barh')

# 多数乘客都在“三等舱”。
dftrain['class'].value_counts().plot(kind='barh')

# 大多数乘客是从南安普敦上船的。
dftrain['embark_town'].value_counts().plot(kind='barh')

# 女性比男性有更高的生存机会。这显然是模型的一个预测特性。
pd.concat([dftrain, y_train], axis=1).grouby(
    'sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

# 创建特征列和输入函数
'''
梯度增强估计器可以利用数值和分类特征。Feature列与所有TensorFlow估计器一起工作，它们的目的
是定义用于建模的特性。此外，它们还提供了一些特性工程功能，如单热编码、规范化和巴克基化。
在本教程中，CATEGORICAL_COLUMNS中的字段从分类列转换为单热编码列(指示列):
'''
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def ont_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            feature_name,
            vocab
        )
    )


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # 需要热编码分类特性
    vocabulary = dftrain[feature_name]
    feature_columns.append(ont_hot_cat_column(feature_name, vocabulary))
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name,
        dtype=tf.float32
    ))

# 您可以查看特性列生成的转换。例如，下面是在单个例子上使用indicator_column时的输出:
example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))
print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ',
      tf.keras.layers.DenseFeatures([class_fc])(example).numpy())

# 此外，您可以一起查看所有的功能列转换:
tf.keras.layers.DenseFeatures(feature_columns)(example).numpy()

'''
接下来需要创建输入函数。这些将指定如何将数据读入我们的模型以进行训练和推理。您将在tf中使用from_tensor_sections方法。
直接从pandas中读取数据的数据API。这适用于较小的内存数据集。对于较大的数据集，tf.data API支持多种文件格式(包括csv)，因此可以处理内存中不适合的数据集。
'''
# 使用整个批处理
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        # 对于训练，可以根据需要多次循环使用dataset (n_epochs=None)。
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

# 训练和评估模型
'''
以下是你将采取的步骤:
    1:初始化模型，指定特征和超参数。
    2:使用train_input_fn将训练数据提供给模型，并使用train函数训练模型。
    3:您将使用评估集(在本例中为dfeval DataFrame)来评估模型性能。您将验证预测是否匹配y_eval数组中的标签。
在训练增强树模型之前，我们先训练一个线性分类器(logistic回归模型)。最好从更简单的模型开始建立基准。
'''
liner_est = tf.estimator.LinearClassifier(feature_columns)

# 训练模型
liner_est.train(train_input_fn, max_steps=100)

# 验证
result = liner_est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

'''
接下来，让我们训练一个增强树模型。对于增强树，支持回归(boostedtreesregression)和分类
(BoostedTreesClassifier)。因为目标是预测一个类是否存活，所以您将使用BoostedTreesClassifier。
'''
# 由于数据适合存储，所以每层使用整个数据集。它会更快。上面的一个批处理被定义为整个数据集。
n_batches = 1
est = tf.estimator.BoostedTreesClassifier(
    feature_columns,
    n_batches_per_layer=n_batches
)
# 一旦构建了指定数量的树，模型将停止训练，而不是基于步骤的数量。
est.train(train_input_fn, max_steps=100)
# 评估
result = est.evaluate(eval_input_fn)
clear_output()
print(pd.Series(result))

'''
现在，您可以使用火车模型从评估集对乘客进行预测。TensorFlow模型经过优化，可以同时对一批
或多个示例进行预测。前面，eval_input_fn是使用整个计算集定义的。
'''
pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='预测概率')

'''
最后，您还可以查看结果的接收者操作特性(ROC)，这将使我们更好地了解真实阳性率和假阳性率之间的权衡。
'''

fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.xlim(0,)
plt.ylim(0,)
