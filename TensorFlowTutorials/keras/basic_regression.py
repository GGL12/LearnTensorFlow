from __future__ import absolute_import, division, print_function
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

#下载数据集
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

#导入数据
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
#查看数据
dataset.tail()
#查看缺失数据信息
dataset.isna().sum()
#删除含有缺失值的项
dataset = dataset.dropna()

#字段origin是类别数据，转化为one-hot 热编码 哑变量。
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

#切分数据集 0.8 train\0.2 test
'''
pandas 还可以这样切分，以前都用的sklearn来切分
'''
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

#查看数据信息 统计学一下数据信息 mean std max min
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#提取标签数据
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#归一化数据
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

'''
搭建模型:
    三个全连接层。
'''
def build_model():
    model = keras.Sequential([
        layers.Dense(64,activation=tf.nn.relu,input_shape=[len(train_dataset.keys())]),
        layers.Dense(64,activation=tf.nn.relu),
        layers.Dense(1)
    ])

    #优化器
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    #编译模型
    model.compile(loss='mean_squared_error',
    optimizer=optimizer,
    metrics=['mean_absolute_error','mean_squared_error'])

    return model

model = build_model()

#查看模型
model.summary()
#试试模型输出效果
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

#训练模型
'''
通过为每个完成的epoch打印一个点来显示训练进度
'''
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0:print('')
        print('.',end='')
EPOCHS = 1000
history = model.fit(
    normed_train_data,train_labels,
    epochs=EPOCHS,validation_split=0.2,verbose=0,
    callbacks=[PrintDot()]
)

#将模型训练的相关数据存入到pandas中
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#训练可视化
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()
  
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)

'''
图表显示，在大约100次迭代之后，验证错误几乎没有改善，甚至有所下降。
让我们更新模型。当验证分数没有提高时，fit会自动停止训练。我们将使用一个
early stop回调函数来测试每个epoch的训练条件。如果一组时间没有显示改进，
那么自动停止训练
'''

#避免过多的浪费时间和算力
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

#验证模型
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("验证集上的平均绝对误差: {:5.2f} MPG".format(mae))