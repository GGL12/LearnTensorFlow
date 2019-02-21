import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1
# 加载数据
(train_features, train_labels), (test_features,
                                 test_labels) = keras.datasets.boston_housing.load_data()

# 数据归一化
train_mean = np.mean(train_features, axis=0)
train_std = np.std(train_features, axis=0)
train_features = (train_features - train_mean) / train_std

# 构建模型 keras构建模型方法一


def build_model():

    # 模型搭建
    model = keras.Sequential(
        [
            # Dense 全连接层
            Dense(20, activation=tf.nn.relu,
                  input_shape=[len(train_features[0])]),
            Dense(1)
        ]
    )
    # 模型编译（传入优化器、损失函数、评价函数）
    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='mse',
        metrics=['mae', 'mse']
    )
    return model


model = build_model()


'''
这部分不太懂，查看了api说明。
tf.keras.callbacks：Callbacks: utilities called at certain points during model training
EarlyStopping：Stop training when a monitored quantity has stopped improving
            大概就是模型loss不在下降或者正确率不在提升后停止训练，避免时间的浪费。
model.fit()里面的参数callbacks=[early_stop, PrintDot()]：List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks
'''


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(train_features, train_labels, epochs=1000, verbose=0, validation_split=0.1,
                    callbacks=[early_stop, PrintDot()]
                    )

# loss,epoch 存储为datafram格式
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

# rmse
rmse_final = np.sqrt(float(hist['val_mean_squared_error'].tail(1)))

# 打印验证集rmse
print()
print("验证集最后的RMSE：{}".format(round(rmse_final, 3)))

# loss 可视化


def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Val Error')
    plt.legend()
    plt.ylim([0, 50])
    plt.show()


plot_history()

# 测试集归一化
test_features_norm = (test_features - train_mean) / train_std
mse, _, _ = model.evaluate(test_features_norm, test_labels)
rmse = np.sqrt(mse)

# 打印测试集rmse
print("测试集最后的RMSE：{}".format(round(rmse_final, 3)))

