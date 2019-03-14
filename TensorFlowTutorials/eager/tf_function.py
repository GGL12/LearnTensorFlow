import timeit
from __future__ import absolute_import, division, print_function

import tensorflow as tf


def add(a, b):
    return a + b


add(tf.ones([2, 2]), tf.ones([2, 2]))
# functons 梯度
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
    result = add(v, 1.0)
tape.gradient(result, v)

# 可以在函数中使用函数


def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)


dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))

# 多态性
# 您可以调用具有不同类型参数的函数来查看发生了什么。


def add(a):
    return a + a


print("add 1", add(1))
print("add 1.1", add(1.1))
print("add string tensor", add(tf.constant('a')))
c = add.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
c(a=tf.constant("a"))

# Functions can be faster than eager code, for graphs with many small ops
conv_layer = tf.keras.layers.Conv2D(100, 3)


def conv_fn(image):
    return conv_layer(image)


image = tf.zeros([1, 200, 200, 100])
conv_layer(image)
conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("请注意卷积的性能没有多大差别")

lstm_cell = tf.keras.layers.LSTMCell(10)


def lstm_fn(input, state):
    return lstm_cell(input, state)


input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
lstm_cell(input, state)
lstm_fn(input, state)
print("eager lstm:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
print("function lstm:", timeit.timeit(lambda: lstm_fn(input, state), number=10))

'''tf.function 状态:
函数作为编程模型的一个非常吸引人的特性是，在一般的数据流图上，函数可以向运行时提供关于
代码预期行为的更多信息。例如，当编写对相同变量具有多次读写的代码时，数据流图可能不会自然
地编码最初预期的操作顺序。在tf.function中，但是，因为我们正在转换从Python跟踪的代码，所以我
们知道预期的执行顺序。这意味着不需要添加手动控制依赖项;tf.function足够聪明，可以添加最小
的必要和足够的控制依赖项集，以便代码正确运行。
'''
# 自动控制依赖关系
a = tf.Variable(1.0)
b = tf.Variable(2.0)


def f(x, y):
    a.assign(y*b)
    b.assign_add(x*a)
    return a + b


f(1.0, 2.0)

# 变量
'''
我们可以使用相同的思想，利用代码的预期执行顺序，使变量在tf.function中创建和使用非常容易
不过，有一个非常重要的警告，那就是使用变量可以编写代码，当多次急切地调用它时，以及当多次计算它的输出张量时，代码的行为会有所不同。
'''
# 下面是一个简单的例子


def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v


f(1.)
# 如果你快速地运行它，你总会得到“2”作为答案;但是如果你反复计算从f(1)得到的张量在图的上下文中你会得到越来越多的数字。

# 所以tf.function不允许您这样编写代码。
# 不过，无歧义代码是可以的
v = tf.Variable(1.0)


def f(x):
    return v.assign_add(x)


f(1.0)  # 2.0
f(2.0)  # 4.0

# 您还可以在tf中创建变量。函数只要我们能证明这些变量只在函数第一次执行时创建。


class C:
    pass


obj = C()
obj.v = None


def g(x):
    if obj.v is None:
        obj.v = tf.Variable(1.0)
    return obj.v.assign_add(x)


g(1.0)  # 2.0
g(2.0)  # 4.0
# 可变初始化器可以依赖于函数参数和其他变量。我们可以用同样的方法计算出正确的初始化顺序方法来生成控件依赖项。
state = []


def fn(x):
    if not state:
        state.append(tf.Variable(2.0 * x))
        state.append(tf.Variable(state[0] * 3.0))
    return state[0] * x * state[1]


fn(tf.constant(1.0))
fn(tf.constant(3.0))

# 控制流程和签名
'''
同时tf.cond和tf.while_loop继续使用tf.function，我们提供了基于Python代码的轻量级编译的更好的替代方法。
签名库与完全集成tf.function，它将重写依赖于张量在图中动态运行的条件和循环。
'''


def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


f(tf.random.uniform([10]))

# 如果您感兴趣，可以查看代码签名生成。不过，这感觉就像在阅读汇编语言。


def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x


print(tf.autograph.to_code(f))
# 要控制签名，请记住，它只影响Python中的基本控制流构造(if、for、while、break等)，并且只在谓词是张量时才更改它们。

# 所以在下面的例子中，第一个循环是静态展开的，而第二个循环是动态转换的:


def f(x):
    for i in range(10):  # 静态python循环，我们不转换它
        do_stuff()
    for i in tf.range(10):  # 取决于一个张量，我们将它转换
        # 类似地，要保证打印和断言是动态发生的，可以使用tf.print和tf.assert:


def f(x):
    for i in tf.range(10):
        tf.print(i)
        tf.Assert(i < 10, ["a"])
        x += x
    return x


f(10)
# 最后，autograph不能将任意的Python代码编译成张量流图。具体来说，动态使用的数据结构仍然需要是TensorFlow数据结构。
# 因此，例如，在循环中积累数据的最佳方法仍然是使用tf.TensorArray:


def f(x):
    ta = tf.TensorArray(tf.float32, size=10)
    for i in tf.range(10):
        x += x
        ta = ta.write(i, x)
    return ta.stack()


f(10.0)
