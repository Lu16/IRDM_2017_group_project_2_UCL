# Author: Zekun Yang
# Student Number: 16099795

import tensorflow as tf
import numpy as np
import os
import random

#### MODEL

train_set_size = 137
n_classes = 5
GRU_size = 32
aff2_size = 100

batch_size = 1000

# weights
affine_layer_1 = {'weights':tf.Variable(tf.random_normal([1, GRU_size])),
                        'biases':tf.Variable(tf.random_normal([GRU_size]))}

affine_layer_2 = {'weights':tf.Variable(tf.random_normal([GRU_size, aff2_size])),
                        'biases':tf.Variable(tf.random_normal([aff2_size]))}

affine_layer_3 = {'weights':tf.Variable(tf.random_normal([aff2_size, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

def GRU_model(data):
    # the GRU model

    # data preprocessing
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, 1])

    # affine transformation for matching the dimension of GRU
    data = tf.add(tf.matmul(data, affine_layer_1['weights']), affine_layer_1['biases'])
    data = tf.split(data, train_set_size, 0)
    

    # GRU layer
    GRU_layer = tf.contrib.rnn.GRUCell(GRU_size) #tf.contrib.rnn.GRUCell
    GRU_output, _ = tf.contrib.rnn.static_rnn(GRU_layer, data, dtype=tf.float32)

    # two affine transformations
    affine_transformation_2 = tf.add(tf.matmul(GRU_output[-1], affine_layer_2['weights']), affine_layer_2['biases'])
    affine_transformation_2 = tf.nn.relu(affine_transformation_2)

    output = tf.add(tf.matmul(affine_transformation_2, affine_layer_3['weights']), affine_layer_3['biases'])

    return output

######### train

x = tf.placeholder('float', [None, train_set_size, 1])
y = tf.placeholder(tf.int64)
    
# training the model
prediction_unsoft = GRU_model(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_unsoft, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

prediction = tf.nn.softmax(prediction_unsoft)
correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))
score = tf.argmax(prediction, 1)
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

saver = tf.train.Saver()
    
with tf.device('/gpu:0'):
    with tf.Session() as sess:

        for m in range(1,5+1): # loop for training 5 folds
            # restore model
            print(m)
            saver.restore(sess, './GRU_model_' + str(m) + '/')
            
            # test data file
            file_test = np.loadtxt('MSLR-WEB10K/Fold'+str(m)+'/test_trans.txt')
            
            file_length = np.size(file_test, 0)
            list_score_test = np.zeros(file_length)

            # since test set is too big, we divide it
            an = int(file_length/batch_size)+1
            b = file_length % batch_size

            for i in range(an):
                if i == an-1:
                    test_data = file_test[(i*batch_size):(i*batch_size+b), 1:]
                    test_data = test_data.reshape((b, train_set_size, 1))
                    # convert score to one-hot vector
                    test_label = np.zeros((b, 5))
                    a = file_test[(i*batch_size):(i*batch_size+b), 0].astype(int)
                    test_label[np.arange(b), a] = 1
        
                    # test score
                    score_test = sess.run(score, feed_dict={x: test_data, y: test_label})
                    list_score_test[(i*batch_size):(i*batch_size+b)] = score_test
                else:
                    test_data = file_test[(i*batch_size):((i+1)*batch_size), 1:]
                    test_data = test_data.reshape((batch_size, train_set_size, 1))
                    # convert score to one-hot vector
                    test_label = np.zeros((batch_size, 5))
                    a = file_test[(i*batch_size):((i+1)*batch_size), 0].astype(int)
                    test_label[np.arange(batch_size), a] = 1
        
                    # test score
                    score_test = sess.run(score, feed_dict={x: test_data, y: test_label})
                    list_score_test[(i*batch_size):((i+1)*batch_size)] = score_test
                    
            # write score in file
            file_length = np.size(file_test, 0)

            test_score_file = open('GRU_score_' + str(m) + '.txt', "w", encoding="latin-1")
            
            count = 0
            qid_temp = file_test[0, 1]
            
            for i in range(file_length):
                this_qid = file_test[i, 1]

                if qid_temp != this_qid:
                    qid_temp = this_qid
                    count = 0

                test_score_file.write(str(int(this_qid)) + ' ' + str(int(count)) + ' ' + str(list_score_test[i]) + '\n')
                count = count+1

            test_score_file.close()
                    
                

    



