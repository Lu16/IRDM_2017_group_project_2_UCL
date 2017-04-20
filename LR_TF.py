import tensorflow as tf
import os
import numpy as np
import math

ROOT_DIR = os.getcwd()
file_ori_train = np.loadtxt(ROOT_DIR+'/all/5/train_feature.txt')
file_ori_vali = np.loadtxt(ROOT_DIR+'/all/5/vali_feature.txt')
whole_length = np.size(file_ori_train, axis=0)
print(whole_length)
file_train = file_ori_train[:, 1:]
file_label = file_ori_train[:, 0]

file_label_one_hot = np.zeros((whole_length, 5))

for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


vali_length = np.size(file_ori_vali, axis=0)
file_vali = file_ori_vali[:, 1:]

file_vali_label = file_ori_train[:, 0]

filr_vali_qid = file_ori_vali[:, 1].tolist()

file_vali_label_one_hot = np.zeros((vali_length, 5))

for i_batch_index in range(vali_length):
    position = file_vali_label[i_batch_index].astype(int)

    file_vali_label_one_hot[i_batch_index, position] = 1

n_epoch=20000
number_of_feature = 137
batch_size = 1024
# learn_rate = 0.1 / batch_size yue 0.00009765625
learn_rate = 0.005
n_classes=5


x = tf.placeholder('float', [None, number_of_feature])
y = tf.placeholder(tf.int64)

W = tf.Variable(tf.random_normal([number_of_feature, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

predsum = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predsum, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

pred = tf.nn.softmax(predsum)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
finalpred =tf.argmax(pred, 1)

saver = tf.train.Saver()

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for indexIter in range(n_epoch):

            choice_batch_index = np.random.choice(whole_length, batch_size)

            batch_x = np.zeros([batch_size, number_of_feature])
            batch_y = np.zeros([batch_size, 5])

            for this_batch_index in range(batch_size):
                batch_x[this_batch_index, :] = file_train[choice_batch_index[this_batch_index]]
                batch_y[this_batch_index, :] = file_label_one_hot[choice_batch_index[this_batch_index], :]

            # to reshape to [batch size X 784 X 1]


            # batch_y = file_label

            _, loss_l = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})

            if indexIter % 5000 == 0:
                accuracy_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})

                print('Iteration %d: loss %.5f Accuracy %.5f' % (indexIter, loss_l, accuracy_train))

        x_vali = file_vali
        y_vali = file_vali_label_one_hot
        # test error
        loss_test = sess.run(loss, feed_dict={x: x_vali, y: y_vali})
        accuracy_test = sess.run(accuracy, feed_dict={x: x_vali, y: y_vali})
        print('Test loss:', loss_test, 'Test accuracy:', accuracy_test)
        rank = sess.run(finalpred, feed_dict={x: x_vali, y: y_vali})

        rank_results = []

        # previous_same_qid = 0

        whole_count = 0

        while True:

            this_qid = file_ori_vali[whole_count, 1]

            count_same_qid = filr_vali_qid.count(this_qid)

            for i_this_qid in range(whole_count, whole_count + count_same_qid):
                rank_results.append([this_qid, i_this_qid, rank[i_this_qid]])

            # previous_same_qid = count_same_qid

            whole_count += count_same_qid

            if whole_count >= vali_length:
                break

        np.savetxt(ROOT_DIR+'/all/5/resultvali5.txt', rank_results, delimiter=' ',
                   newline='\r\n', fmt='%s')

        # save the model
        # if not os.path.exists('./lstm_model_fold_2/'):
        #     os.mkdir('./lstm_model_fold_2/')
        # saver.save(sess, "./lstm_model_fold_2/")