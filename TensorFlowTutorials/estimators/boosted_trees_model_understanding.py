# 梯度增强树:模型理解
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output
import pandas as pd
import numpy as np
from __future__ import absolute_import, division, print_function
'''
如果你想对梯度增强模型进行端到端演练，请查看增强树教程。在本教程中，您将:
学习如何解释局部和全局增强树模型
如何适合数据集来获取增强树模型
'''

# 如何解释本地和全局的增强树模型
'''
    局部可解释性是指在单个示例级别上理解模型的预测，而全局可解释性是指从整体上理解模型。
这些技术可以帮助机器学习(ML)实践者在模型开发阶段检测偏差和错误
    于局部可解释性，您将学习如何创建和可视化每个实例的贡献。为了将其与特性重要性区分开来，
我们将这些值称为方向特性贡献(dfc)。
    对于全局可解释性，您将检索并可视化基于增益的特性重要性、排列特性重要性，还将显示聚合的dfc。
'''
# 加载泰坦尼克数据集


tf.random.set_seed(123)

# 加载数据集
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# 创建特性列、输入函数和训练估计器

# 预处理的数据
'''
使用原始数字列和一个热编码分类变量创建特性列。
'''
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']


def one_hot_cat_column(feature_name, vocab):
    return fc.indicator_column(
        fc.categorical_column_with_vocabulary_list(feature_name,
                                                   vocab))


feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    # 需要热编码分类特性。
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name,
                                             dtype=tf.float32))

# 构建输入管道
'''
使用tf中的from_tensor_sections方法创建输入函数。
'''

# 使用整个批处理，因为这是一个如此小的数据集。
NUM_EXAMPLES = len(y_train)


def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (X.to_dict(orient='list'), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
    # 对于训练，可以根据需要多次循环使用dataset (n_epochs=None)。
        dataset = (dataset.repeat(n_epochs).batch(NUM_EXAMPLES))
        return dataset
    return input_fn


# 训练和评估输入函数。
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

# 训练模型
params = {
    'n_trees': 50,
    'max_depth': 3,
    'n_batches_per_layer': 1,
    # 必须启用center_bias = True才能获得dfc。这将迫使模型在使用任何特性(例如，使用…的平均值)之前做一个初步的预测
    # 用于回归的训练标签或用于分类的日志几率使用交叉熵损失
    'center_bias': True
}
est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# 训练
est.train(train_input_fn, max_steps=100)
# 评估
results = est.evaluate(eval_input_fn)
clear_output()
pd.Series(results).to_frame()

# 模型解释及绘图
sns_colors = sns.color_palette('colorblind')

# 局部的可解释性
'''
    接下来，您将输出方向特性贡献(DFCs)，使用Palczewska等人介绍的方法和Saabas解释随机森林的
方法来解释单个预测(该方法也可以在scikit-learn for Random forest的treeinterpreter包中使用)。所产生的资料来源包括:
    pred_dicts = list(est.experimental_predict_with_explanations(pred_input_fn))
'''
pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

# 创建DFC pandas dataframe 数据格式。
labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc.describe().T

# DFCs的一个很好的性质是贡献的和加上偏差等于给定例子的预测。
# Sum of DFCs + bias == probabality.
bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
np.testing.assert_almost_equal(dfc_prob.values,
                               probs.values)

# 为一名单独的乘客绘制DFCs图。让我们根据贡献的方向性对图进行颜色编码，并在图上添加特征值，从而使图更漂亮。


def _get_color(value):
    """使积极的DFCs地块绿色，消极的DFCs地块红色."""
    green, red = sns.color_palette()[2:4]
    if value >= 0:
        return green
    return red


def _add_feature_values(feature_values, ax):
    """在图的左侧显示feature的值"""
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
                 fontproperties=font, size=12)


def plot_example(example):
    TOP_N = 8  # 查看最为相关的8个特征
    # Sort by magnitude.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    ax = example.to_frame().plot(kind='barh',
                                 color=[colors],
                                 legend=None,
                                 alpha=0.75,
                                 figsize=(10, 6))
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # 增加特征的值
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
    return ax


# 绘制结果
ID = 182
example = df_dfc.iloc[ID]  # 从评估集中选择第i个例子。
TOP_N = 8  # 查看最为相关的8个特征
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(
    ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)

'''
    震级贡献越大，对模型预测的影响越大。负的贡献表明，该实例的特征值降低了模型的预测，而正的贡献增加了预测。
    您还可以使用voilin将示例的dfc与整个分布进行比较。
'''


def dist_violin_plot(df_dfc, ID):
    # 初始化绘画
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 创建dataframe示例。
    TOP_N = 8  # 查看最为相关的8个特征
    example = df_dfc.iloc[ID]
    ix = example.abs().sort_values()[-TOP_N:].index
    example = example[ix]
    example_df = example.to_frame(name='dfc')
   # 添加整个发行版的贡献
    parts = ax.violinplot([df_dfc[w] for w in ix],
                          vert=False,
                          showextrema=False,
                          widths=0.7,
                          positions=np.arange(len(ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # 增加特征值
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)

    # 添加本地贡献。
    ax.scatter(example,
               np.arange(example.shape[0]),
               color=sns.color_palette()[2],
               s=100,
               marker="s",
               label='contributions for example')

    ax.plot([0, 0], [1, 1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                       frameon=True)
    legend.get_frame().set_facecolor('white')

    # 格式化绘画
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)


# 绘画
dist_violin_plot(df_dfc, ID)
plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(
    ID, probs[ID], labels[ID]))
