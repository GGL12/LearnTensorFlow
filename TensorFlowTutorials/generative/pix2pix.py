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
