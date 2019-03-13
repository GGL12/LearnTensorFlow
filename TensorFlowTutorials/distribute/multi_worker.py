'''
本教程演示tf. distribution。该方法可用于分布式多工种的训练。如果您使用tf.estimator编写代码，并且希望扩展到具有高性能的单台机器之外，那么本教程适合您。
在开始之前，请先阅读tf. distribution。策略指导。多gpu培训教程也是相关的，因为本教程使用相同的模型。
'''
from __future__ import absolute_import, division, print_function
import tensorflow_datasets as tfds
import tensorflow as tf

import os, json

#输入函数
'''
本教程使用来自TensorFlow数据集的MNIST数据集。这里的代码类似于多gpu训练教程，
但有一个关键的区别:当使用Estimator进行多工人培训时，需要根据工人的数量对数据集进行切分，
以确保模型收敛。输入数据由worker索引分片，因此每个worker处理数据集的不同部分1/num_workers。
'''
BUFFER_SIZE = 10000
BATCH_SIZE = 64
def input_fn(mode,input_context=None):
    datasets,ds_info = tfds.load(
        name='mnist',
        with_info=True,
        as_supervised=True
    )
    mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKey.TRAIN else datasets['test'])

    def scale(image,label):
        image = tf.cast(image,tf.float32)
        image /= 255
        return image,label

    if input_context:
        mnist_dataset = mnist_dataset.shard(
            input_context.num_input_pipelines,
            input_context.input_pipeline_id
        )
    return mnist_dataset.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

#Multi-worker配置
'''
    与多gpu训练相比，本教程的一个关键区别是multi-worker设置。TF_CONFIG环境变量是向集
群中的每个工作人员指定集群配置的标准方法。
    TF_CONFIG有两个组件:cluster 和task.cluster提供关于整个集群的信息，即集群中的worker
和参数服务器。任务提供当前任务的信息。在本例中，任务类型为worker，任务索引为0。
'''

#定义模型
LEARNING_RATE = 1e-4
def model_fn(feature,labels,mode):
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    logits = model(feature,training=False)

    if mode == tf.estimator.ModeKey.PREDICT:
        predictions = {'logits':logits}
        return tf.estimator.EstimatorSpec(labels=labels,predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescalGrossentropy(
        learning_rate=LEARNING_RATE
    )
    loss = tf.keras.losses.SparseCategoricalGrossentropy(
        from_logits=True
    )(labels,logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))

#MultiWorkerMirroredStrategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

#训练并评估模型
config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)

