import tensorflow as tf
import os
import numpy as np
import math

file_ori_train = np.loadtxt('D:/UCL/IRDM/MSLR-WEB10K/Fold1/train_feature.txt')

whole_length = np.size(file_ori_train, axis=0)

file_train = file_ori_train[:, 2:]
# file_train = np.column_stack((file_train, file_ori_train[:, 1]))

# file_label = file_ori_train[:, 0].reshape(-1, 1)
file_label = file_ori_train[:, 0]

file_label_one_hot = np.zeros((whole_length, 5))

for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


file_ori_vali = np.loadtxt('D:/UCL/IRDM/MSLR-WEB10K/Fold1/test_feature.txt')
vali_length = np.size(file_ori_vali, axis=0)
file_vali = file_ori_vali[:, 2:]
# file_vali = np.column_stack((file_vali, file_ori_vali[:, 1]))

file_vali_label = file_ori_vali[:, 0]

filr_vali_qid = file_ori_vali[:, 1].tolist()

file_vali_label_one_hot = np.zeros((vali_length, 5))

for i_batch_index in range(vali_length):
    position = file_vali_label[i_batch_index].astype(int)

    file_vali_label_one_hot[i_batch_index, position] = 1


batch_size = 1024

lstm_chunk_size = 5
lstm_n_chunks = 25
size_stack = 136-125

n_classes = 5
lstm_unit = 128
hidden_unit = 100

weights = {
    'conv1': tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 4])),
    'flat': tf.Variable(tf.truncated_normal(shape=[25*5*4, 128])),
    'out': tf.Variable(tf.truncated_normal(shape=[128, 5])),
}
biases = {
    'conv1': tf.Variable(tf.constant(0.01, shape=[4])),
    'flat': tf.Variable(tf.truncated_normal(shape=[128])),
    'out': tf.Variable(tf.truncated_normal(shape=[5])),
}

w_stack_hi = tf.Variable(tf.random_normal([16, hidden_unit]))
b_stack_hi = tf.Variable(tf.random_normal([hidden_unit]))

w_stack_out = tf.Variable(tf.random_normal([hidden_unit, n_classes]))
b_stack_out = tf.Variable(tf.random_normal([n_classes]))


def split_train(x_in):
    x_lstm = x_in[:, 0:125]
    x_stack = x_in[:, 125:]
    return x_lstm, x_stack


def train_stack(x):
    hi_1 = tf.nn.relu(tf.matmul(x, w_stack_hi) + b_stack_hi)
    out = tf.matmul(hi_1, w_stack_out) + b_stack_out
    return out


def train_lstm(x):
    input_layer = tf.reshape(x, [-1, lstm_n_chunks, lstm_chunk_size, 1])
    conv = tf.nn.relu(tf.nn.conv2d(input_layer, weights['conv1'], [1, 1, 1, 1], padding='SAME') + biases['conv1'])
    flat = tf.matmul(tf.reshape(conv, [-1, lstm_n_chunks*lstm_chunk_size*4]), weights['flat']) + biases['flat']
    out = tf.matmul(flat, weights['out']) + biases['out']
    return out


def main():

    x_lstm = tf.placeholder('float', [None, lstm_n_chunks, lstm_chunk_size])
    x_stack = tf.placeholder('float', [None, size_stack])
    y = tf.placeholder('float')

    # to predict and optimize
    lstm_pred = train_lstm(x_lstm)
    re_lstm_pred = tf.reshape(lstm_pred, [-1, 5])
    re_x_stack = tf.reshape(x_stack, [-1, size_stack])
    x_next_in = tf.concat([re_lstm_pred, re_x_stack], 1)
    stacked_pred = train_stack(x_next_in)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=stacked_pred, labels=y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    # when training the model, the softmax layer and cross entropy are combined together
    # when testing or predicting, the spftmax layer are calculated as it is not in the neural network model
    prediction_t = tf.nn.softmax(stacked_pred)
    correct = tf.equal(tf.argmax(prediction_t, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    saver = tf.train.Saver()

    with tf.device('/gpu:0'):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            for indexIter in range(40000):

                choice_batch_index = np.random.choice(whole_length, batch_size)

                batch_x = np.zeros([batch_size, 136])
                batch_y = np.zeros([batch_size, 5])
                for this_batch_index in range(batch_size):
                    batch_x[this_batch_index, :] = file_train[choice_batch_index[this_batch_index]]
                    batch_y[this_batch_index, :] = file_label_one_hot[choice_batch_index[this_batch_index], :]
                # to reshape to [batch size X 25 X 5]
                batch_x_lstm, batch_x_stack = split_train(batch_x)
                batch_x_lstm = batch_x_lstm.reshape((batch_size, lstm_n_chunks, lstm_chunk_size))
                batch_x_stack = batch_x_stack.reshape((batch_size, -1))

                # batch_y = file_label

                _, loss_l = sess.run([optimizer, loss], feed_dict={x_lstm: batch_x_lstm, x_stack: batch_x_stack, y: batch_y})

                if indexIter % 100 == 0:

                    accuracy_train = sess.run(accuracy, feed_dict={x_lstm: batch_x_lstm, x_stack: batch_x_stack, y: batch_y})

                    # print('Iteration %d: loss %.5f Accuracy %.5f' % (indexIter, loss_l, accuracy_train))

            # use validation set
            x_test = file_vali
            y_test = file_vali_label_one_hot
            x_test_lstm, x_test_stack = split_train(x_test)
            x_test_lstm = x_test_lstm.reshape((-1, lstm_n_chunks, lstm_chunk_size))

            test_len = len(x_test)
            rank = np.zeros([test_len, 1])
            accuracy_test = 0
            # batch_size..... + left(less than batch_size)
            for this_test in range(int(math.floor(test_len / batch_size))):
                prediction = sess.run(prediction_t,
                                      feed_dict={x_lstm: x_test_lstm[this_test * batch_size:(this_test + 1) * batch_size],
                                                 x_stack: x_test_stack[this_test * batch_size:(this_test + 1) * batch_size],
                                                 y: y_test[this_test * batch_size:(this_test + 1) * batch_size]})
                rank[this_test * batch_size:(this_test + 1) * batch_size] = (np.argmax(prediction, 1).astype(int)).reshape(
                    [batch_size, 1])

                accuracy_test += np.sum(sess.run(accuracy,
                                                 feed_dict={
                                                     x_lstm: x_test_lstm[this_test * batch_size:(this_test + 1) * batch_size],
                                                     x_stack: x_test_stack[this_test * batch_size:(this_test + 1) * batch_size],
                                                     y: y_test[this_test * batch_size:(this_test + 1) * batch_size]}))

            # left(less than batch_size)
            prediction = sess.run(prediction_t, feed_dict={
                                                     x_lstm: x_test_lstm[int(math.floor(test_len/batch_size))*batch_size:],
                                                     x_stack: x_test_stack[int(math.floor(test_len/batch_size))*batch_size:],
                                                     y: y_test[int(math.floor(test_len/batch_size))*batch_size:]})

            rank[int(math.floor(test_len / batch_size)) * batch_size:] = (np.argmax(prediction, 1).astype(int)).reshape(
                [test_len - int(math.floor(test_len / batch_size)) * batch_size, 1])

            accuracy_test += np.sum(sess.run(accuracy,
                                             feed_dict={
                                                 x_lstm: x_test_lstm[this_test * batch_size:(this_test + 1) * batch_size],
                                                 x_stack: x_test_stack[this_test * batch_size:(this_test + 1) * batch_size],
                                                 y: y_test[this_test * batch_size:(this_test + 1) * batch_size]}))

            accuracy_test /= (int(math.floor(test_len / batch_size)) + 1)

            print('accuracy_test', accuracy_test)
            rank_results = []

            whole_count = 0

            while True:

                this_qid = file_ori_vali[whole_count, 1]

                count_same_qid = filr_vali_qid.count(this_qid)

                for i_this_qid in range(0, count_same_qid):
                    rank_results.append([int(this_qid), int(i_this_qid), rank[whole_count+i_this_qid]])

                whole_count += count_same_qid

                if whole_count >= vali_length:
                    break

            np.savetxt('D:/UCL/IRDM/CNN/new/1.txt', rank_results, delimiter=' ', newline='\r\n', fmt='%s')

            # save the model
            if not os.path.exists('./cnn_model/'):
                os.mkdir('./cnn_model/')
            saver.save(sess, "./cnn_model/")

if __name__ == '__main__':
    main()

