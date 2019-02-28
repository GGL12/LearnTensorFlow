from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras

# 加载数据 MNISt
(train_images, train_labels), (test_images,
                               test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# 搭建模型


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(
            512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


model = create_model()
model.summary()

# 储存模型
checkpoint_path = "train_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# 创建检查点回调
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)
model = create_model()
model.fit(
    train_images,
    train_labels,
    epochs=5,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]
)

# 加载模型
model = create_model()
# 未添加权重
loss, acc = model.evaluate(test_images, test_labels)
print("未训练模型的正确率: {:5.2f}%".format(100*acc))
# 添加权重
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("重加载模型的正确率: {:5.2f}%".format(100*acc))

# 回调函数设置多个模型文件的保存
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # period=n每隔多少保存一次
    period=5)
model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# 手动保存权重Model.save_weights
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("重加载模型的正确率: {:5.2f}%".format(100*acc))

# 保存整个模型 Keras使用HDF5标准提供基本的保存格式.
model = create_model()
model.fit(train_images, train_labels, epochs=5)
# 保存模型为.5后缀名的HDF5格式
model.save('my_model.h5')
# 加载HDF5格式的模型
new_model = keras.models.load_model('my_model.h5')
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels)
print("重加载模型的正确率: {:5.2f}%".format(100*acc))

'''
keras无法保存TensorFlow的优化器（tf.train）.使用tf的编译器需加载模型后重新进行编译。
'''
