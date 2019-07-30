# 使用keras回调函数和Tensorboard来检查并监督深度学习模型
import numpy as np
import tensorflow.python import keras
'''
模型检查点
提前终止
在训练过程中动态调节某些参数值
在训练过程中记录训练指标和验证指标。（可视化操作）
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
'''

# ModelCheckpoint 与 EarlyStopping 回调函数
from tensorflow.python import keras


'''
EarlyStopping: monitor指监控模型的验证精度 patience=1如果精度多于一轮的时间内不在改善，中断训练
ModelCheckpoint: filepath:保存模型的文件路径 monitor=val_acc save_best_only监控验证集的loss值。没有改善就不需要覆盖模型文件
'''
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=1),
                  keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                                  monitor='val_loss', save_best_only=True)]
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['acc']
)

model.fit(
    x,
    y,
    epochs=10,
    callbacks=callbacks_list,
    validation_data=(x_val, y_val)
)

# ReduceLROnPlateau 回调函数
'''
如果验证损失函数不在改善,那么可以使用这个回调函数来减低学习率。
在训练过程中如果出现了损失平台(loss plateau) ,那么增大或减少学习率都是跳出局部最小值的有效策略
'''
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  # 监控模型的验证损失
        factor=0.1,  # 触发时将学习率除以10
        patience=10，  # 如果验证损失在10轮内都没有改善，那么久触发这个回调函数
    )
]
model.fit(
    x, y,
    batch_size=32,
    callbacks=callbacks_list,
    validation_data=(x_val, y_val)
)

# 编写自己的回调函数
'''
on_epoch_begin
on_epoch_end

on_batch_begin
on_batch_end

on_train_begin
on_train_end
这些方法被调用时都有一个 logs 参数，这个参数是一个字典，里面包含前一个批量、前
一个轮次或前一次训练的信息，即训练指标和验证指标等。此外，回调函数还可以访问下列属性。
 self.model：调用回调函数的模型实例。
 self.validation_data：传入 fit 作为验证数据的值。
'''
# 实例


class ActivationLogger(keras.callbacks.Callback):

    def set_model(self, model):
        self.model = model
        layer_outputs = [layer.outputs for layer in model.layers]
        self.activations_model = keras.models.Model(model.input, layer_outputs)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            raise RuntimeError('Requires validation_data.')
        validation_sample = self.validation_data[0][0:1]
        activations = self.activations_model.predict(validation_sample)
        f = open('activations_at_epoch_' + str(epoch) + '.npz', 'w')
        np.savez(f, activations)
        f.close()
