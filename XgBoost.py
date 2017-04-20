import os
import numpy as np
import math

ROOT_DIR = os.getcwd()
file_ori_train = np.loadtxt(ROOT_DIR+'/all/5/trainex.txt')
file_ori_vali = np.loadtxt(ROOT_DIR+'/all/5/valiex.txt')
whole_length = np.size(file_ori_train, axis=0)
print(whole_length)
file_train = file_ori_train[:, 1:]
file_label = file_ori_train[:, 0].astype(int)+1

file_label_one_hot = np.zeros((whole_length, 5))

for i_batch_index in range(whole_length):
    position = file_label[i_batch_index].astype(int)

    file_label_one_hot[i_batch_index, position] = 1


vali_length = np.size(file_ori_vali, axis=0)
file_vali = file_ori_vali[:, 1:]

file_vali_label = file_ori_train[:, 0].astype(int)

file_vali_qid = file_ori_vali[:, 1].tolist()

file_vali_label_one_hot = np.zeros((vali_length, 5))

for i_batch_index in range(vali_length):
    position = file_vali_label[i_batch_index].astype(int)

    file_vali_label_one_hot[i_batch_index, position] = 1

import xgboost as xgb

xgbc = xgb.XGBClassifier(max_depth=6, learning_rate=0.01, n_estimators=10000,
                    objective='multi:softprob', seed=0)
xgbc.fit(file_train, file_label,
eval_set=[(file_train, file_label), (file_vali, file_vali_label)], early_stopping_rounds=30,eval_metric='mlogloss', verbose=True)



print('done')
file_ori_test = np.loadtxt(ROOT_DIR+'/all/5/testex.txt')
test_length = np.size(file_ori_test, axis=0)
file_test = file_ori_test[:, 1:]
file_test_label = file_ori_train[:, 0].astype(int)
file_test_qid = file_ori_test[:, 1].tolist()

rank = xgbc.predict(file_test).reshape( file_test.shape[0], 5 )
rank = np.argmax(rank, 1).astype(int)-1
def get_accuracy(y_pre, y_true):
    correct_prediction = np.equal(np.argmax(y_pre, 1), np.argmax(y_true, 1))
    correct_prediction = correct_prediction.astype(np.float)
    accuracy = np.mean(correct_prediction)

    return accuracy

accuracy_test = get_accuracy(rank, file_test_label)

print('Accuracy test', accuracy_test)
rank_results = []
whole_count = 0
while True:
    this_qid = file_ori_test[whole_count, 1]

    count_same_qid = file_test_qid.count(this_qid)

    for i_this_qid in range(whole_count, whole_count + count_same_qid):
        rank_results.append([this_qid, i_this_qid, rank[i_this_qid]])

    # previous_same_qid = count_same_qid

    whole_count += count_same_qid

    if whole_count >= test_length:
        break

np.savetxt(ROOT_DIR+'/all/5/resultxg.txt', rank_results, delimiter=' ',
           newline='\r\n', fmt='%s')