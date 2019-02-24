import numpy as np
import tensorflow as tf
import reader
#from tensorflow.models.tutorials.rnn.ptb import util
from tensorflow.models.tutorials.rnn.ptb import reader
DATA_PATH = r"C:\Users\Administrator\Desktop\project\projects\RNN\PTB_data"

#隐藏层大小
HIDDEN_SIZE = 200

#LSTM结构层数
NUM_LAYERS = 2

#词典规模 ，生成one-hot
VOCAB_SIZE = 10000

#学习率
LEARNING_RATE = 1.0

#训练数据batch大小
TRAIN_BATCH_SIZE = 20

#训练数据截断数
TRAIN_NUM_STEP = 35

#测试集batch大小
EVAL_BATCH_SIZE = 1

#测试步数据截断数
EVAL_NUM_STEP = 1

#训练轮数
NUM_EPOCH = 2

#dropout
KEEP_PROB = 0.5

#控制梯度膨胀的参数
MAX_GRAD_NORM = 5

class Model(object):
    '''
    定义一个关于模型的类
    '''
    
    def __init__(self, is_training, batch_size, num_steps):
        
        #记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps
        
        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        
        # 定义使用LSTM结构及训练时使用dropout。
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
            
        #双层LTSM
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
        
        # 初始化最初的状态。
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        
        #总共有VOCAB_SIZE个单词，每个单词的维度为HIDDER)SIZE，所有embedding参数的维度为VOCAB_SIZE * HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        
        # 将原本单词ID转为单词向量。
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        #dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        #定义输出列表，这里先将不同时刻的LSTM结构的输入收集起来，再通过一个全连接层得到最终的输出
        outputs = []
        
        #state存储不同batch中LTSM的状态，将其初始化为0向量
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                #从输入数据中获取当前时刻的输入并传入LSTM结构当中
                cell_output, state = cell(inputs[:, time_step, :], state)
                #将当前输入添加到输出队列
                outputs.append(cell_output) 
                
        #输出队列展开成[batch,hidden_size*num_steps]然后 reshpae成[batch*numsteps,hidden_size]的形状，
        #便于接下来的全连接操作
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        
        #最终的预测结构在每一个时刻都是VOCAB_SIZE的数组，通过softmax层之后表示下一次位置的不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        
        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], #预测的结果
            [tf.reshape(self.targets, [-1])], #reshape成一维数组
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])#损失的权重都是为1，表示不同batch和不同时刻的重要程度都是1
        #计算每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state
        
        # 只在训练模型时定义反向传播操作。
        if not is_training: return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        #通过clip_by_global_norm函数控制的大小，避免梯度膨胀的问题
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        #使用GradientDescentOptimizer优化模型
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #定义训练方法
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        
#使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值       
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    '''计算perplexity的辅助变量'''
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 使用当前数据训练或者测试模型
    for step in range(epoch_size):
        x, y = session.run(data)
        #在当前batch上运行train_op并计算损失值。交叉嫡损失函数计算的就是下一个单词为给定单词的概率
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        #将不同时刻，不同batch的概率加起来就可以得到第二个perplexity公式等号右边的部分，再将这个做指数运算就可以得到perplexity的值。
        total_costs += cost
        iters += model.num_steps
        
        #只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("经过 %d 步后, perplexity 值是 %.3f" % (step, np.exp(total_costs / iters)))
            
    #返回给定模型在给定数据上的perplexity的值
    return np.exp(total_costs / iters)

def main():
    #获得原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP
    
    #定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    
    #定义训练用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = Model(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    #定义评测用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = Model(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)


    with tf.Session() as session:
        #初始化所以的 变量
        tf.global_variables_initializer().run()
        
        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        
        #使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            
            #在所有训练数据上序列循环神经网络模型
            run_epoch(session, train_model, train_queue, train_model.train_op, True, train_epoch_size)

            #使用验证数据评测模型效果
            valid_perplexity = run_epoch(session, eval_model, eval_queue, tf.no_op(), False, valid_epoch_size)
            print("Epoch: %d 验证集 Perplexity: %.3f" % (i + 1, valid_perplexity))
        
        #最后使用测试数据评估模型
        test_perplexity = run_epoch(session, eval_model, test_queue, tf.no_op(), False, test_epoch_size)
        print("测试集 Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()