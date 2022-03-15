#! -*- coding: utf-8 -
import numpy as np
import tensorflow as tf

def test_rnn():
    hidden_num = 5
    x = [[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3] , [0, 0, 0], [0, 0, 0]],[[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3] , [1, 2, 3], [1, 2, 3]]]
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    # print(x)
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_num)
    outputs, final_states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x_ = sess.run(x)
        o = sess.run(outputs)
        f = sess.run(final_states)
        print(x_)
        print(50 * "-")
        print(o)
        print(50 * "-")
        print(f)

def test_bidirect_rnn():
    x = [[[1,2,3],[2,3,4],[0,0,0]],
    [[7,8,9],[2,5,7],[8,5,3]],
    [[4,6,7],[0,0,0],[0,0,0]],
    [[2,6,9],[7,4,7],[0,0,0]]]

    x = tf.convert_to_tensor(x,dtype=tf.float32)
    print(x.shape)

    lstm1 = tf.contrib.rnn.LSTMCell(5, state_is_tuple=True)
    lstm2 = tf.contrib.rnn.LSTMCell(5, state_is_tuple=True)

    outputs, final_state = tf.nn.bidirectional_dynamic_rnn(lstm1, lstm2, x,sequence_length=[2,3,1,2], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        o = sess.run(outputs)
        f = sess.run(final_state)
        # print (o)
        # print ('======================')
        # print (f)

if __name__ == '__main__':
    test_rnn()