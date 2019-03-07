# Estimator - 一种可极大地简化机器学习编程的高阶 TensorFlow API
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output

#开启eager
tf.enable_eager_execution()

#导入tensorflow model https://github.com/tensorflow/models


#下载数据
from official.wide_deep import census_dataset
from official.wide_deep import census_main
census_dataset.download("/tmp/census_data/")

#加载数据
train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"
import pandas
train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)
train_df.head()
#这些列分为两类:类别列和数据特征列

#将数据转换为张量
'''
在构建tf.estimator模型时，输入数据是通过使用输入函数(或input_fn)指定的。这个生成器函数
返回一个tf.data。批量(特征-字典，标签)对的数据集。直到传递给tf.estimator.Estimator方法
(如train和evaluate)时才调用它。
输入生成器函数返回以下类型:
    feature:从特性名称到包含多个特性的Tensor或SparseTensors的字典。
    label:包含批次label的Tensor。
'''
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df),label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds

#由于我们启用了即时执行，因此很容易检查结果数据集:
ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys())[:5])
    print()
    print('A batch of Ages  :', feature_batch['age'])
    print()
    print('A batch of Labels:', label_batch )


'''但是这种方法的可伸缩性非常有限。更大的数据集应该从磁盘读取。
census_dataset。input_fn提供了一个使用tf.decode_csv和tf.data.TextLineDataset实现此目的的示例:
'''
import inspect
print(inspect.getsource(census_dataset.input_fn))
#这个input_fn函数返回相同的输出:
ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)
for feature_batch, label_batch in ds.take(1):
    print('Feature keys:', list(feature_batch.keys())[:5])
    print()
    print('Age batch   :', feature_batch['age'])
    print()
    print('Label batch :', label_batch )

'''因为Estimators期望input_fn不带参数，所以通常我们对train_inpf来对数据进行两次迭代
'''
import functools

train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

#基本的特征列

#数字列
age = fc.numeric_column('age')
#模型将使用feature_column定义来构建模型输入。您可以使用input_layer函数检查结果输出:
fc.input_layer(feature_batch, [age]).numpy()
#下面将训练和评估只使用年龄特征的模型:
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()
print(result)

#类似地，我们可以为我们想在模型中使用的每个连续特征列定义一个NumericColumn:
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
my_numeric_columns = [age,education_num, capital_gain, capital_loss, hours_per_week]
fc.input_layer(feature_batch, my_numeric_columns).numpy()

#您可以通过将feature_columns参数更改为构造函数来对这些特性重新培训模型:
lassifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()
for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))

#分类列
'''要为分类特征定义一个特征列，可以使用tf.feature_column之一创建一个
CategoricalColumn.categorical_column *功能
'''
relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
#这将从原始输入特性创建一个稀疏的one-hot热编码。

#运行输入层，配置年龄和关系列
fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)])
#如果我们事先不知道可能的值集，可以使用categorical_column_with_hash_bucket:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
#在这里，当我们在训练中遇到特征列occupation中的每个可能值时，它们都被哈希为一个整数ID。示例批有几个不同的职业:
for item in feature_batch['occupation'].numpy():
    print(item.decode())
#如果我们运行input_layer与散列，我们看到输出的形状是(batch_size, hash_bucket_size):
occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])
occupation_result.numpy().shape
#如果我们用tf，更容易看到实际结果。hash_bucket_size维度上的argmax。注意任何重复的职业是如何映射到相同的伪随机指数:
tf.argmax(occupation_result, axis=1).numpy()

#让我们用同样的技巧来定义其他类别特征
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])
my_categorical_columns = [relationship, occupation, education, marital_status, workclass]
#很容易使用这两组列来配置使用所有这些特性的模型:
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
clear_output()
for key,value in sorted(result.items()):
  print('%s: %s' % (key, value))


#通过嵌套使连续特征分类
'''
有时连续特征和标签之间的关系不是线性的。例如，年龄和收入——一个人的收入可能在职业生涯的早期
增长，然后增长可能在某一时刻放缓，最后，退休后收入下降。在这种情况下，使用原始年龄作为实值
一个好的选择，因为模型只能学习三种情况中的一种:
    随着年龄的增长，收入总是以一定的速度增长(正相关)，
    随着年龄的增长，收入总是以一定的速度下降(负相关)
    无论年龄多大，收入都保持不变(没有相关性)。
'''
#如果我们想分别了解收入和每个年龄组之间的细粒度相关性，我们可以利用板球化。嵌套是将一个连续特征的整个范围划分为一组连续的数据的过程
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
fc.input_layer(feature_batch, [age, age_buckets]).numpy()

education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

#定义逻辑回归模型
import tempfile

base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model = tf.estimator.LinearClassifier(
    model_dir=tempfile.mkdtemp(), 
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1)

#训练和评估模型
train_inpf = functools.partial(census_dataset.input_fn, train_file, 
                               num_epochs=40, shuffle=True, batch_size=64)

model.train(train_inpf)
clear_output()  # used for notebook display



#对模型进行训练后，通过预测测试的标签来评估模型的准确性:
results = model.evaluate(test_inpf)
clear_output()
for key,value in sorted(result.items()):
    print('%s: %0.2f' % (key, value))

#让我们更详细地看看这个模型是如何执行的:
import numpy as np
predict_df = test_df[:20].copy()

pred_iter = model.predict(
    lambda:easy_input_function(predict_df, label_key='income_bracket',
                               num_epochs=1, shuffle=False, batch_size=10))

classes = np.array(['<=50K', '>50K'])
pred_class_id = []

for pred_dict in pred_iter:
    pred_class_id.append(pred_dict['class_ids'])

predict_df['predicted_class'] = classes[np.array(pred_class_id)]
predict_df['correct'] = predict_df['predicted_class'] == predict_df['income_bracket']
clear_output()
predict_df[['income_bracket','predicted_class', 'correct']]


#添加正则化以防止过拟合
#您可以使用以下代码将L1和L2正则化添加到模型中:
model_l1 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=10.0,
        l2_regularization_strength=0.0))

model_l1.train(train_inpf)

results = model_l1.evaluate(test_inpf)
clear_output()
for key in sorted(results):
    print('%s: %0.2f' % (key, results[key]))


model_l2 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=10.0))
model_l2.train(train_inpf)
results = model_l2.evaluate(test_inpf)
clear_output()
for key in sorted(results):
    print('%s: %0.2f' % (key, results[key]))

#这些规范化模型的性能并不比基本模型好多少。让我们来看看模型的权值分布，以便更好地看到正则化的效果:
def get_flat_weights(model):
    weight_names = [
        name for name in model.get_variable_names()
        if "linear_model" in name and "Ftrl" not in name]

    weight_values = [model.get_variable_value(name) for name in weight_names]

    weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)

    return weights_flat

weights_flat = get_flat_weights(model)
weights_flat_l1 = get_flat_weights(model_l1)
weights_flat_l2 = get_flat_weights(model_l2)

eight_mask = weights_flat != 0
weights_base = weights_flat[weight_mask]
weights_l1 = weights_flat_l1[weight_mask]
weights_l2 = weights_flat_l2[weight_mask]

#现在画出分布:
plt.figure()
_ = plt.hist(weights_base, bins=np.linspace(-3,3,30))
plt.title('Base Model')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l1, bins=np.linspace(-3,3,30))
plt.title('L1 - Regularization')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l2, bins=np.linspace(-3,3,30))
plt.title('L2 - Regularization')
_=plt.ylim([0,500])