# Author: Chao Ban
# Student Number: 16042900


import numpy as np

train_file_path = 'D:/DM_sub_data/Folder1/train_feature.txt'
vali_or_test_path = 'D:/DM_sub_data/Folder1/vali_feature.txt'
store_score_path = 'D:/DM_sub_data/Logistic_Regression/score_logistic_regression_137_feature_fold_1_vali.txt'

number_of_feature = 137

file_ori_train = np.loadtxt(train_file_path)

whole_length = np.size(file_ori_train, axis=0)

file_train = file_ori_train[:, 1:]

file_label = file_ori_train[:, 0]

file_label_one_hot = np.zeros((whole_length, 5))

for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1

file_ori_vali = np.loadtxt(vali_or_test_path)
vali_length = np.size(file_ori_vali, axis=0)
file_vali = file_ori_vali[:, 1:]

file_vali_label = file_ori_vali[:, 0]

filr_vali_qid = file_ori_vali[:, 1].tolist()

file_vali_label_one_hot = np.zeros((vali_length, 5))

for i_batch_index in range(vali_length):
    position = file_vali_label[i_batch_index].astype(int)

    file_vali_label_one_hot[i_batch_index, position] = 1


# To obtain the output of the linear layer
def get_layer(x, w, b):
    y = np.matmul(x, w) + b
    return y


# To obtain the output of the softmax
def get_softmax(x):
    L_e = np.exp(x/100000)
    sum_L_e = np.tile(np.sum(L_e, axis=1), (np.size(L_e, axis=1), 1)).T

    return L_e / sum_L_e


# To obtain the output of the softmax-loss
def softmax_cross_entropy_log_loss(x, y_train):
    L_e = np.exp(x/100000)

    sum_L_e = np.tile(np.sum(L_e, axis=1), (np.size(L_e, axis=1), 1)).T

    loss = -sum(sum(y_train * (L_e - np.log(sum_L_e))))

    return loss


# To obtain the gradient of loss
def d_loss(y_prediction, y_train):
    return y_prediction - y_train


# To obtain the gradient of layer
def d_layer(lay_in, d_previous, w):
    dim_x, dim_y = np.shape(d_previous)
    gradient_w = np.matmul(lay_in.T, d_previous)
    gradient_b = np.diag(np.matmul(np.ones((dim_y, dim_x)), d_previous)).T
    gradient_L = np.matmul(d_previous, w.T)
    return gradient_w, gradient_b, gradient_L


def train_model(list_w, list_b, x_train, y_train, layer_num, learn_rate):
    list_output = list()
    list_input = list()
    list_input.append(x_train)

    output_last = get_layer(list_input[-1], list_w[-1], list_b[-1])
    list_output.append(output_last)

    loss = softmax_cross_entropy_log_loss(list_output[-1], y_train)

    list_gradient_w = list()
    list_gradient_b = list()
    list_gradient_L = list()

    dLoss = d_loss(get_softmax(list_output[-1]), y_train)

    gradient_w_last, gradient_b_last, gradient_L_last = d_layer(list_input[-1], dLoss, list_w[-1])
    list_gradient_w.append(gradient_w_last)
    list_gradient_b.append(gradient_b_last)
    list_gradient_L.append(gradient_L_last)

    for index_layer in range(0, layer_num):
        list_w[index_layer] -= learn_rate * list_gradient_w[layer_num - 1 - index_layer]
        list_b[index_layer] -= learn_rate * list_gradient_b[layer_num - 1 - index_layer]

    return loss, list_w, list_b, list_output[-1]


def exam_model(list_w, list_b, x_test):

    list_input = list()

    list_input.append(x_test)

    output_last = get_layer(list_input[-1], list_w[-1], list_b[-1])

    return output_last


def get_accuracy(y_pre, y_true):
    correct_prediction = np.equal(np.argmax(y_pre, 1), np.argmax(y_true, 1))
    correct_prediction = correct_prediction.astype(np.float)
    accuracy = np.mean(correct_prediction)

    return accuracy


def main():

    # To initialize the parameters
    layer_num = 1
    list_num_w = [number_of_feature, 5]

    batch_size = 1024

    # 0.00009765625
    learn_rate = 0.1 / batch_size

    list_w = list()
    list_b = list()
    for index_layer in range(0, layer_num):
        w = np.random.randn(list_num_w[index_layer], list_num_w[index_layer + 1]) / 100000
        b = np.random.randn(list_num_w[index_layer + 1]) / 100000
        list_w.append(w)
        list_b.append(b)

    for indexIter in range(30001):

        choice_batch_index = np.random.choice(whole_length, batch_size)

        batch_x = np.zeros([batch_size, number_of_feature])
        batch_y = np.zeros([batch_size, 5])
        for this_batch_index in range(batch_size):
            batch_x[this_batch_index, :] = file_train[choice_batch_index[this_batch_index]]
            batch_y[this_batch_index, :] = file_label_one_hot[choice_batch_index[this_batch_index], :]

        x_train = batch_x/10000

        y_train = batch_y

        loss, list_w, list_b, out_train = train_model(list_w, list_b, x_train, y_train, layer_num, learn_rate)

        if indexIter % 1000 == 0:
            accuracy_train = get_accuracy(out_train, y_train)

            print('Iteration %d: Accuracy %.5f(train)' % (indexIter, accuracy_train))

    # To save model
    np.savez('logistic_regression.npz', weight=list_w, bias=list_b)

    x_test = file_vali/10000

    y_test = file_vali_label_one_hot

    out_test = exam_model(list_w, list_b, x_test)

    accuracy_test = get_accuracy(out_test, y_test)

    print('Accuracy test', accuracy_test)

    rank = np.argmax(out_test, 1).astype(int)

    print(rank)

    print(np.sum(rank))

    rank_results = []

    whole_count = 0

    while True:

        this_qid = file_ori_vali[whole_count, 1]

        count_same_qid = filr_vali_qid.count(this_qid)

        for i_this_qid in range(whole_count, whole_count + count_same_qid):

            rank_results.append([this_qid.astype(int), i_this_qid, rank[i_this_qid]])

        whole_count += count_same_qid

        if whole_count >= vali_length:
            break

    np.savetxt(store_score_path, rank_results, delimiter=' ', newline='\r\n', fmt='%s')

if __name__ == '__main__':
    main()