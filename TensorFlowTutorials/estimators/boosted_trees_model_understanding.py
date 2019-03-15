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

#全局特征重要性
'''
此外，您可能希望从整体上理解模型，而不是研究单个预测。下面，您将计算并使用:
    1:使用est.experimental ental_feature_importances实现基于增益的特性导入
    2:排列重要性
    3:使用est.experimental ental_predict_with_explanation聚合dfc

    基于收益的特征重要性度量的是对特定特征进行分割时的损失变化，而排列特征重要性的计算则是通过
对评价集上的模型性能进行评估来计算的，方法是对每个特征逐一进行洗牌，并将模型性能的变化归因于洗牌后的特征。  
    般来说，排列特征重要性优于基于收益的特征重要性，尽管在潜在预测变量的测量规模或类别数量不同以及特征相关(源)的情况下，这两种方法都可能不可靠
'''
#1. Gain-based特性重要性
importances = est.experimental_feature_importances(normalize=True)
df_imp = pd.Series(importances)

#可视化重要性。
N = 8
ax = (df_imp.iloc[0:N][::-1].plot(kind='barh',
          color=sns_colors[0],
          title='Gain feature importances',
          figsize=(10, 6)))
ax.grid(False, axis='y')

#DFCs平均绝对值
dfc_mean = df_dfc.abs().mean()
N = 8
sorted_ix = dfc_mean.abs().sort_values()[-N:].index  # Average and sort by absolute.
ax = dfc_mean[sorted_ix].plot(kind='barh',
                       color=sns_colors[1],
                       title='Mean |directional feature contributions|',
                       figsize=(10, 6))
ax.grid(False, axis='y')

FEATURE = 'fare'
feature = pd.Series(df_dfc[FEATURE].values, index=dfeval[FEATURE].values).sort_index()
ax = sns.regplot(feature.index.values, feature.values, lowess=True);
ax.set_ylabel('contribution')
ax.set_xlabel(FEATURE);
ax.set_xlim(0, 100);

#3.排列特性重要性
def permutation_importances(est, X_eval, y_eval, metric, features):
    """逐列洗牌值，观察对eval集的影响。.
    source: http://explained.ai/rf-importance/index.html
    """
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in features:
        save = X_eval[col].copy()
        X_eval[col] = np.random.permutation(X_eval[col])
        m = metric(est, X_eval, y_eval)
        X_eval[col] = save
        imp.append(baseline - m)
    return np.array(imp)

def accuracy_metric(est, X, y):
    """TensorFlow estimator accuracy."""
    eval_input_fn = make_input_fn(X,
                                  y=y,
                                  shuffle=False,
                                  n_epochs=1)
    return est.evaluate(input_fn=eval_input_fn)['accuracy']
features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
importances = permutation_importances(est, dfeval, y_eval, accuracy_metric,
                                      features)
df_imp = pd.Series(importances, index=features)

sorted_ix = df_imp.abs().sort_values().index
ax = df_imp[sorted_ix][-5:].plot(kind='barh', color=sns_colors[2], figsize=(10, 6))
ax.grid(False, axis='y')
ax.set_title('Permutation feature importance');

#可视化模型拟合
from numpy.random import uniform, seed
from matplotlib.mlab import griddata

#创建假的数据
seed(0)
npts = 5000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)

#训练准备数据
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

xi = np.linspace(-2.0, 2.0, 200),
yi = np.linspace(-2.1, 2.1, 210),
xi,yi = np.meshgrid(xi, yi);

df_predict = pd.DataFrame({
    'x' : xi.flatten(),
    'y' : yi.flatten(),
})
predict_shape = xi.shape

def plot_contour(x, y, z, **kwargs):
    #网格数据。
    plt.figure(figsize=(10, 8))
    #对网格数据进行等高线，在非均匀数据点上绘制点。
    CS = plt.contour(x, y, z, 15, linewidths=0.5, colors='k')
    CS = plt.contourf(x, y, z, 15,
                    vmax=abs(zi).max(), vmin=-abs(zi).max(), cmap='RdBu_r')
    plt.colorbar()
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

#你可以把这个函数形象化。较红的颜色对应较大的函数值。
zi = griddata(x, y, z, xi, yi, interp='linear')
plot_contour(xi, yi, zi)
plt.scatter(df.x, df.y, marker='.')
plt.title('Contour on training data');

fc = [tf.feature_column.numeric_column('x'),
      tf.feature_column.numeric_column('y')]

def predict(est):
    """来自给定estimator的预测"""
    predict_input_fn = lambda: tf.data.Dataset.from_tensors(dict(df_predict))
    preds = np.array([p['predictions'][0] for p in est.predict(predict_input_fn)])
    return preds.reshape(predict_shape)

#首先，让我们尝试将线性模型与数据相匹配。
train_input_fn = make_input_fn(df, df.z)
est = tf.estimator.LinearRegressor(fc)
est.train(train_input_fn, max_steps=500);

plot_contour(xi, yi, predict(est))

#不太合适。接下来，让我们试着将一个GBDT模型拟合到它上面，并试着理解这个模型是如何拟合函数的。
n_trees = 1 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
clear_output()
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20);