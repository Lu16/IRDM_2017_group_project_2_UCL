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
accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))

saver = tf.train.Saver()
    
with tf.device('/gpu:0'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for m in range(1,5+1): # loop for training 5 folds

            # train data file
            file_train = np.loadtxt('MSLR-WEB10K/Fold'+str(m)+'/train_trans.txt')
            file_vali = np.loadtxt('MSLR-WEB10K/Fold'+str(m)+'/vali_trans.txt')
            
            for i in range(1,5000+1):
                # train the model on the batch
                file_length = np.size(file_train, 0)
                list_train = random.sample(range(file_length), batch_size)
                
                train_data = file_train[list_train, 1:]
                train_data = train_data.reshape((batch_size, train_set_size, 1))
                # convert score to one-hot vector
                train_label = np.zeros((batch_size, 5))
                a = file_train[list_train, 0].astype(int)
                train_label[np.arange(batch_size), a] = 1

                # train error
                _, loss_train = sess.run([optimizer, cross_entropy], feed_dict={x: train_data, y: train_label})
                accuracy_train = sess.run(accuracy, feed_dict={x: train_data, y: train_label})
                print('Number of batch:', i ,'Train loss:', loss_train,'Train accuracy:',accuracy_train)

                if i % 100 == 0:
                    # test error on validation set
                    # random test batch
                    file_length = np.size(file_vali, 0)
                    list_test = random.sample(range(file_length), batch_size)
                    
                    test_data = file_vali[list_test, 1:]
                    test_data = test_data.reshape((batch_size, train_set_size, 1))
                    # convert score to one-hot vector
                    test_label = np.zeros((batch_size, 5))
                    a = file_vali[list_test, 0].astype(int)
                    test_label[np.arange(batch_size), a] = 1


                    # test error
                    loss_test = sess.run(cross_entropy, feed_dict={x: test_data, y: test_label})
                    accuracy_test = sess.run(accuracy, feed_dict={x: test_data, y: test_label})
                    print('Number of batch:', i ,'Test loss:', loss_test,'Test accuracy:',accuracy_test)
                

                # save model
                if i % 1000 == 0:

                    if not os.path.exists('./GRU_model_' + str(m) + '/'):
                        os.mkdir('./GRU_model_' + str(m) + '/')
                    saver.save(sess, './GRU_model_' + str(m) + '/')


