# Author: Chao Ban
# Student Number: 16042900

import tensorflow as tf
import os
import numpy as np
import math

train_file_path = 'D:/DM_sub_data/Folder1/train_feature.txt'
vali_or_test_path = 'D:/DM_sub_data/Folder1/vali_feature.txt'
store_score_path = 'D:/DM_sub_data/LSTM/score_lstm_136_feature_128_units_0001_lr_40000_epoch_fold_2_test.txt'

file_ori_train = np.loadtxt(train_file_path)

whole_length = np.size(file_ori_train, axis=0)

file_train = file_ori_train[:, 2:]

file_label = file_ori_train[:, 0]

file_label_one_hot = np.zeros((whole_length, 5))

for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


file_ori_vali = np.loadtxt(vali_or_test_path)
vali_length = np.size(file_ori_vali, axis=0)
file_vali = file_ori_vali[:, 2:]

file_vali_label = file_ori_train[:, 0]

filr_vali_qid = file_ori_vali[:, 1].tolist()

file_vali_label_one_hot = np.zeros((vali_length, 5))

for i_batch_index in range(vali_length):
    position = file_vali_label[i_batch_index].astype(int)

    file_vali_label_one_hot[i_batch_index, position] = 1

batch_size = 1024

chunk_size = 1
n_chunks = 136

n_classes = 5
lstm_unit = 128
hidden_unit = 100

w_in = tf.Variable(tf.random_normal([chunk_size, lstm_unit]))
b_in = tf.Variable(tf.random_normal([lstm_unit]))

w_lin = tf.Variable(tf.random_normal([lstm_unit, hidden_unit]))
b_lin = tf.Variable(tf.random_normal([hidden_unit]))

w_out = tf.Variable(tf.random_normal([hidden_unit, n_classes]))
b_out = tf.Variable(tf.random_normal([n_classes]))


def train_lstm(x):

    x = tf.transpose(x, [1, 0, 2])

    x = tf.reshape(x, [-1, chunk_size])

    input_lstm = tf.matmul(x, w_in) + b_in

    input_lstm = tf.split(input_lstm, n_chunks, 0)

    lstm_cell = tf.contrib.rnn.LSTMCell(lstm_unit, state_is_tuple=True)
    outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, input_lstm, dtype=tf.float32)

    lin = tf.nn.relu(tf.matmul(outputs[-1], w_lin) + b_lin)

    output = tf.matmul(lin, w_out) + b_out

    return output


def main():

    x = tf.placeholder('float', [None, n_chunks, chunk_size])
    y = tf.placeholder('float')

    prediction = train_lstm(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    prediction_t = tf.nn.softmax(prediction)
    correct = tf.equal(tf.argmax(prediction_t, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()

    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for indexIter in range(40000):

                choice_batch_index = np.random.choice(whole_length, batch_size)

                batch_x = np.zeros([batch_size, n_chunks])
                batch_y = np.zeros([batch_size, 5])
                for this_batch_index in range(batch_size):
                    batch_x[this_batch_index, :] = file_train[choice_batch_index[this_batch_index]]
                    batch_y[this_batch_index, :] = file_label_one_hot[choice_batch_index[this_batch_index], :]

                batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))


                _, loss_l = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

                if indexIter % 100 == 0:

                    accuracy_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

                    print('Iteration %d: loss %.5f Accuracy %.5f' % (indexIter, loss_l, accuracy_train))

            x_test = file_vali.reshape((-1, n_chunks, chunk_size))
            y_test = file_vali_label_one_hot

            test_len = len(x_test)
            rank = np.zeros(test_len)
            accuracy_test = 0
            # 1000..... + left(less than 1000)
            for this_test in range(int(math.floor(test_len/1000))):
                prediction = sess.run(prediction_t, feed_dict={x: x_test[this_test*1000:(this_test+1)*1000], y: y_test[this_test*1000:(this_test+1)*1000]})
                rank[this_test*1000:(this_test+1)*1000] = np.argmax(prediction, 1).astype(int)

                accuracy_test += np.sum(sess.run(accuracy, feed_dict={x: x_test[this_test*1000:(this_test+1)*1000], y: y_test[this_test*1000:(this_test+1)*1000]}))

            # left(less than 1000)
            prediction = sess.run(prediction_t, feed_dict={x: x_test[int(math.floor(test_len/1000))*1000:], y: y_test[int(math.floor(test_len/1000))*1000:]})
            rank[int(math.floor(test_len/1000))*1000:] = (np.argmax(prediction, 1).astype(int))
            accuracy_test += np.sum(sess.run(accuracy, feed_dict={x: x_test[int(math.floor(test_len/1000))*1000:], y: y_test[int(math.floor(test_len/1000))*1000:]}))

            accuracy_test /= (int(math.floor(test_len/1000))+1)

            print('accuracy_test', accuracy_test)

            rank_results = []


            whole_count = 0

            while True:

                this_qid = file_ori_vali[whole_count, 1]

                count_same_qid = filr_vali_qid.count(this_qid)

                for i_this_qid in range(whole_count, whole_count + count_same_qid):
                    rank_results.append([this_qid.astype(int), i_this_qid, rank[i_this_qid]])

                # previous_same_qid = count_same_qid

                whole_count += count_same_qid

                if whole_count >= vali_length:
                    break

            np.savetxt(store_score_path, rank_results, delimiter=' ', newline='\r\n', fmt='%s')

            # save the model
            # if not os.path.exists('./lstm_model_fold_2/'):
            #     os.mkdir('./lstm_model_fold_2/')
            # saver.save(sess, "./lstm_model_fold_2/")

if __name__ == '__main__':
    main()

