# 多输入和多输出模型
from tensorflow.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.python.keras.models import Model

main_input = Input(shape=(100,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')

auxiliary_input = Input(shape=(5,), name='aux_input')
x = concatenate([lstm_out, auxiliary_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x
x=Dense(64, activation='relu')(x

main_output=Dense(1, activation='sigmoid', name='main_output')

model=Model(inputs=[main_input, auxiliary_input],
            outputs=[main_output, auxiliary_output])

# 对两个输出设置loss 权重
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1, 0.2])

model.fit([headline_data, additional_data], [
          labels, labels], epochs=50, batch_size=32)
# 在编译时可以不使用列表的形式传递。根据上述定义的name熟悉以字典的形式上传
