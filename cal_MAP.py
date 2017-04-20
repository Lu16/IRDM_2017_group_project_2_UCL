# Author: Chao Ban
# Student Number: 16042900


import numpy as np

test_file_path = 'D:/MSLR/Fold4/test.txt'
store_score_path = 'D:/DM_sub_data/Logistic_Regression/score_logistic_regression_137_feature_fold_1_vali.txt'
result_path = 'D:/rank_lstm_136_feature_128_units_0001_lr_40000_epoch_fold_4_test_MAP.txt'

file_test = np.loadtxt(test_file_path, delimiter=' ', usecols=0, unpack=True)

file_score = np.loadtxt(store_score_path)

test_length = np.size(file_test)
test_rank = np.reshape(file_test, (test_length, 1))

new_score = np.hstack((file_score, test_rank))


def main():

    data_with_same_qid = []
    previous_qid = '-1'
    qid_count = 0
    all_score_list = []

    for i_line in range(test_length):

        qid = new_score[i_line, 0]

        if qid == previous_qid:
            data_with_same_qid.append(new_score[i_line, :])
            if new_score[i_line, 3] > 0:
                qid_count += 1
        else:
            if np.size(data_with_same_qid) != 0:

                if qid_count == 0:
                    print(previous_qid)
                    all_score_list.append([previous_qid, 0])
                else:
                    s = score(data_with_same_qid, qid_count)
                    all_score_list.append([previous_qid, s])
            data_with_same_qid = []
            qid_count = 0
            data_with_same_qid.append(new_score[i_line, :])
            if new_score[i_line, 3] > 0:
                qid_count += 1

        if i_line == test_length-1:
            if qid_count == 0:
                print(previous_qid)
                all_score_list.append([previous_qid, 0])
            else:
                s = score(data_with_same_qid, qid_count)
                all_score_list.append([previous_qid, s])

        previous_qid = qid

    str_map = np.reshape(np.repeat(np.reshape([str('MAP')], [-1, 1]), int(np.size(all_score_list, axis=0)+1)), [-1, 1])

    mean_score = np.mean(all_score_list, axis=0)[1]
    print(mean_score)
    all_score_list.append(['all', mean_score])
    all_score_array = np.asarray(all_score_list)
    print(all_score_list)

    np.savetxt(result_path, np.c_[str_map, all_score_array], fmt='%s', delimiter='    ', newline='\r\n')


def score(data_with_same_qid, qid_count):

    new_sorted_data_raw = np.zeros((np.size(data_with_same_qid, axis=0) + 1, 4))

    for i_line in range(np.size(data_with_same_qid, axis=0)):
        new_sorted_data_raw[i_line, :] = data_with_same_qid[i_line]

    new_sorted_data_raw = (new_sorted_data_raw[new_sorted_data_raw[:, 2].argsort()[::-1]])

    previous_score = -1
    begin_line = 0
    new_sorted_data = np.zeros((np.size(data_with_same_qid, axis=0) + 1, 4))

    for i_line in range(np.size(new_sorted_data_raw, axis=0)):

        this_score = new_sorted_data_raw[i_line, 2]

        if i_line != 0:
            if this_score != previous_score or i_line == np.size(new_sorted_data_raw, axis=0)-1:
                this_block = new_sorted_data_raw[begin_line:i_line, :]
                this_block = (this_block[this_block[:, 1].argsort()])
                new_sorted_data[begin_line:i_line, :] = this_block
                begin_line = i_line

        previous_score = this_score

    count = 0
    ap = 0

    for i_line in range(np.size(new_sorted_data, axis=0)):

        if new_sorted_data[i_line, 3] > 0:
            count += 1
            ap += float(count)/(i_line + 1)

    return ap/qid_count

if __name__ == '__main__':
    main()
