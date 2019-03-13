'''tf.distribute.Strategy提供一个抽象，用于跨多个处理单元分发您的训练。其目标是允许用户
使用现有的模型和培训代码，以最小的更改来启用分布式培训。
    通过在一台机器上的多个gpu上同步训练来进行图形内复制。本质上，它将模型的所有变量复制
到每个处理器。然后，它使用all-reduce组合来自所有处理器的梯度，并将组合值应用于模型的所有副本。
    本例使用tf。keras API来构建模型和训练循环。
'''
from __future__ import absolute_import, division, print_function

import tensorflow_datasets as tfds
import tensorflow as tf
import os

# 下载数据集
datasets, ds_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']

# 定义分配策略
'''
创建一个MirroredStrategy对象。这将处理分发，并提供一个上下文管理器(tf.distribute.MirroredStrategy.scope)来在内部构建模型。
'''

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 设置输入管道
'''
如果一个模型是在多个gpu上训练的，那么就应该相应地增加批处理大小，以便有效地利用额外的计算能力。此外，学习速度应该相应地调整。
'''
# 你也可以做ds_info.splits.total_num_examples来获得总数数据集中实例的数量。
num_train_examples = ds_info.splits['train'].num_examples
num_test_examples = ds_info.splits['test'].num_examples

BUFFER_SIZE = 10000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# 像素值为0-255，必须规范化为0-1范围。在函数中定义这个比例。


def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

# 将此函数应用于训练和测试数据，对训练数据进行洗牌，并对其进行批处理以进行训练。


train_dataset = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# 创建模型
'''
在strategy.scope上下文中创建和编译Keras模型。
'''
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

# 定义回调函数。
'''
这里使用的回调是:
    1:Tensorboard:这个回调函数为Tensorboard写了一个日志，允许您可视化图形。
    2:模型检查点:这个回调函数在每个epoch之后保存模型。
    3:学习率调度程序:使用这个回调函数，您可以在每个epoch/批处理之后调度学习率。
'''

# 定义检查点目录来存储检查点
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# 函数用于衰减学习率。你可以定义任何你需要的衰减函数。


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# 在每个历元结束时打印LR的回调。


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n学习率在 {} 是 {}'.format(epoch + 1,
                                      model.optimizer.lr.numpy()))


callbacks = [
    tf.keras.callbacks.Tensorboard(lod_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay)
]

# 训练和评估
'''
现在，以通常的方式训练模型，调用模型上的fit并传递在本教程开始时创建的数据集。无论您是否分发训练，这一步都是相同的。
'''

model.fit(train_dataset, epochs=10, callbacks=callbacks)

# 要查看模型如何执行，请加载最新的检查点并对测试数据调用evaluate。调用评估之前使用适当的数据集。
model.load_weighrs(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)
print('测试集损失: {}, 测试集正确率: {}'.format(eval_loss, eval_acc))

# 导出模型
'''
如果想导出图形和变量，SavedModel是最好的方法。模型可以在有作用域的情况下加载，也可以在没有作用域的情况下加载。此外，SavedModel与平台无关。
'''
path = 'saved_model/'
tf.keras.experimental.export_save_model(model, path)

# 在没有strategy.scope的情况下加载模型。
unreplicated_model = tf.keras.experimental.load_from_saved_model(path)

unreplicated_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)
print('测试集损失: {}, 测试集正确率: {}'.format(eval_loss, eval_acc))

# 用strategy.scope加载模型。
with strategy.scope():
    replicated_model = tf.keras.experimental.load_from_saved_model(path)
    replicated_model.compile(loss='sparse_categorical_crossentropy',
                             optimizer=tf.keras.optimizers.Adam(),
                             metrics=['accuracy'])

    eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))
