# Pix2Pix
'''
    本笔记本演示了使用条件GAN的图像到图像的翻译，如条件对抗性网络的图像到图像的翻译中所述。
利用这项技术，我们可以为黑白照片上色，将谷歌地图转换为谷歌地球，等等。在这里，我们将建筑
立面转换为真实的建筑。
    例如，我们将使用位于布拉格的捷克技术大学机器感知中心提供的CMP外观数据库。为了保持示例
简短，我们将使用这个数据集的预处理副本，该副本由上述文章的作者创建。
在一个P100 GPU上，每个epoch大约需要58秒。
'''
# 导入TensorFlow和其他库
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# 加载数据
'''
    您可以从这里下载这个数据集和类似的数据集。如本文所述，我们将随机抖动和镜像应用于训练数据集。
1:在随机抖动中，图像大小调整为286 x 286，然后随机裁剪为256 x 256
2:在随机镜像中，图像水平随机翻转i。e从左到右。
'''
_URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'

path_to_zip = tf.keras.utils.get_file('facades.tar.gz',
                                      origin=_URL,
                                      extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

BUFFER_SIZE = 400
BATCH_SIZE = 256
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]

    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


inp, re = load(PATH + 'train/100.jpg')
# 为matplotlib转换为int以显示图像
plt.figure()
plt.imshow(inp/255.0)
plt.figure()
plt.imshow(re/255.0)


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    strcked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        strcked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]

##将图像归一化为[- 1,1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


def random_jitter(input_image, real_image):
    # resize 286 * 286 * 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # resize 256*256*3
    input_image, real_image = resize(input_image, real_image, 256, 256)

    if tf.random.uniform(()) > 0.5:
        # 随机的镜像
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image


'''
如下面的图片所示
他们正在经历随机的紧张
#随机抖动在论文中描述的是
# 1。将图像的大小调整为更大的高度和宽度
# 2。随机裁剪到原始大小
# 3。随机水平翻转图像
'''
plt.figure(figsize=(6, 6))
for i in range(4):
    rj_inp, rj_re = random_jitter(inp, re)
    plt.subplot(2, 2, i+1)
    plt.imshow(rj_inp/255.0)
    plt.axis('off')
plt.show()


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(
        input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


# 输入管道
train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(
    load_image_train,
    num_parallel_calls=tf.data.experimental.AUTOTRUE
)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/.jpg')
# 改组，以便为每个epoch生成一个不同的图像预测和显示模型的进度。
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = train_dataset.batch(1)

# 构建生成器
'''
1:generator的结构是一个改进的U-Net
2:编码器中的每个块为(Conv -> Batchnorm -> Leaky ReLU)
3:解码器中的每个块是(转置Conv -> Batchnorm -> Dropout(应用于前3块)-> ReLU)
4:编码器和解码器之间有跳过连接(如在U-Net中)。
'''
OUTPUT_CHANNELS = 3


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.python.keras.Sequential()
    result.add(
        tf.python.keras.layers.Conv2D(
            filters, size, strides=2, padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    if apply_batchnorm:
        result.add(tf.python.keras.layers.BatchNormalization())
    result.add(tf.python.keras.layers.LeakyReLU())

    return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print(down_result.shape)


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.python.keras.Sequential()
    result.add(
        tf.python.keras.layers.Conv2DTranspose(
            filters,
            size, strides=2,
            padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
    )
    if apply_dropout:
        result.add(tf.python.keras.layers.Dropout(0.5))
    result.add(tf.python.keras.layers.ReLU())

    return result


up_model = upsample(3, 4)
up_result = up_model(down_result)
print(up_result.shape)


def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 8, 512)
        downsample(512, 4),  # (bs, 2, 8, 512)
        downsample(512, 4),  # (bs, 1, 8, 512)
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.python.keras.layers.Conv2DTranspose(
        OUTPUT_CHANNELS,
        4,
        strides=2,
        padding='same',
        kernel_initializer=initializer,
        activation='tanh'
    )  # (bs, 256, 256, 3)
    concat = tf.python.keras.layers.Concatenate()

    inputs = tf.python.keras.layers.Input(shape=[None, None, 3])

    x = inputs

    # 通过模型向下采样
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    # 向上采样并建立跳过连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.python.keras.Model(inputs=inputs, output=x)


generator = Generator()
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])

# 构建鉴别器
'''
1:鉴别器是一个PatchGAN。
2:鉴别器中的每个块为(Conv -> BatchNorm -> Leaky ReLU)
3:最后一层之后输出的形状为(batch_size, 30, 30, 1)
4:输出的每个30x30补丁对输入图像的70x70部分进行分类(这样的架构称为PatchGAN)。
5:鉴别器接收2个输入。
    5.1:输入图像和目标图像，将其分类为实图像。
    5.2:输入图像和生成的图像(生成器的输出)，应分类为伪图像。
    5.3:我们在代码中将这两个输入连接在一起(tf.concat([inp, tar], axis=-1))
'''


def Discriminator():
    initializer = tf.random_normal_initializer(0, 0.02)

    inp = tf.python.keras.layers.Input(
        shape=[None, None, 3], name='input_image')
    tar = tf.python.keras.layers.Input(
        shape=[None, None, 3], name='target_image')

    x = tf.python.keras.layers.concatenate(
        [inp, tar])(bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_padl = tf.python.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.python.keras.layers.Conv2D(
        512,
        4,
        strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_padl)  # (bs, 31, 31, 512)

    batchnorml = tf.python.keras.layers.BatchNormalization()(conv)

    leake_rule = tf.python.keras.layers.LeakyReLU()(batchnorml)

    zero_pad2 = tf.python.keras.layers.ZeroPadding2D()(batchnorml)

    last = tf.python.keras.layers.Conv2D(
        1,
        4, strides=1,
        kernel_initializer=initializer,
    )(zero_pad2)  # (bs, 30, 30, 1)
    return tf.python.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=20, vmax=20, cmap='RdBu_r')
plt.colorbar()

# 定义损失函数和优化器
'''
鉴别器的损失:
    1:鉴别器损失函数有2个输入;真实的图像，生成的图像
    2:real_loss是真实图像和一组图像的sigmoid交叉熵损失(因为这些是真实图像)
    3:generated_loss是生成的图像和一组0的sigmoid交叉熵损失(因为这些是假图像)
    4:那么total_loss是real_loss和generated_loss的和
生成器损失:
    1:它是生成的图像和一组图像的s形交叉熵损失。
    2:文中还包括L1损失，即生成的图像与目标图像之间的平均绝对误差。
    3:这允许生成的图像在结构上与目标图像相似。
    4:计算生成器总损耗= gan_loss + * l1_loss的公式，其中= 100。这个值是由本文作者决定的。
'''
LAMBDA = 100
loss_object = tf.python.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generator_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generator_loss = loss_object(tf.zeros_like(
        disc_generator_output), disc_generator_output)

    total_disc_loss = real_loss + generator_loss

    return total_disc_loss


def generator_loss(disc_generator_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(
        disc_generator_output), disc_generator_output)

    # 平均绝对误差
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    tatal_gan_loss = gan_loss + (LAMBDA * l1_loss)

    return tatal_gan_loss


generator_optimizer = tf.python.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.python.keras.optimizers.Adam(2e-4, beta_1=0.5)

# 检查点(基于对象的储存)
checkppint_dir = './training_checkpoints'
checkppint_prefix = os.path.join(checkppint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator
)

# 训练
'''
1:我们首先遍历数据集
2:生成器获取输入图像，然后我们得到生成的输出。
3:鉴别器接收input_image和生成的图像作为第一个输入。第二个输入是input_image和target_image。
4:接下来，我们计算了生成器和鉴别器的损耗。
5:然后，我们计算与生成器和鉴别器变量(输入)相关的损失梯度，并将这些梯度应用于优化器。

生成图像:
    1:经过训练，是时候生成一些图像了!
    2:我们将图像从测试数据集传递到生成器。
    3:然后生成器将把输入图像转换成我们期望的输出。
    4:最后一步是绘制预测图，瞧!
'''
EPOCHS = 200


def generate_images(model, test_input, tar):
    '''
    这里的训练=True是故意的
    我们希望在运行模型时获得批处理统计数据测试数据集上的#。如果我们使用training=False，
    我们将得到从训练数据集中获得的累计统计信息(我们不想要)
    '''
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]

    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # 获取[0,1]之间的像素值来绘制它。
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )
    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(
        discriminator_gradients,
        discriminator.trainable_variables
    ))


def train(dataset, epochs):
    for epoch in epochs:
        start = time.time()

        for input_image, target in dataset:
            train_step(input_image, target)
        clear_output(wait=True)

        for inp, tar in test_dataset.take(1):
            generate_images(generator, target)

        # 每隔二十个epoch保存训练模型
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkppint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time()-start))


train(train_dataset, EPOCHS)

# 恢复最新的检查点和测试
checkpoint.restore(tf.train.latest_checkpoint(checkppint_dir))

# 对整个测试数据集进行测试
for inp, tar in test_dataset:
    generate_images(generator, inp, tar)
