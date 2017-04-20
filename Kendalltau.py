# Author: Zekun Yang
# Student Number: 16099795

import codecs
import numpy as np
import sys
import itertools


# function for calculate Kendall's tau for two list
def Kendalls_tau(list_test_score_temp, list_model_score_temp):
    # calculate Kendall's tau for this qid
    N = np.size(list_test_score_temp)

    if N > 1:
        # get all pairs and compare
        comp_list_test = np.zeros(int(N*(N-1)/2))
        comp_list_model = np.zeros(int(N*(N-1)/2))

        j = 0
        for a, b in itertools.combinations(list_test_score_temp, 2):
            if a > b:
                comp_list_test[j] = 1
            elif a < b:
                comp_list_test[j] = 0
            else:
                comp_list_test[j] = -10

            j = j+1


        j = 0
        for a, b in itertools.combinations(list_model_score_temp, 2):
            if a > b:
                comp_list_model[j] = 1
            elif a < b:
                comp_list_model[j] = 0
            else:
                comp_list_model[j] = -10

            j = j+1

        # calculation
        a = comp_list_test + comp_list_model;
        concordant = np.count_nonzero(a == 0) + np.count_nonzero(a == 2)
        discordant = np.count_nonzero(a == 1)
        
        Kt = (concordant - discordant)/int(N*(N-1)/2)
    else:
        # if only one element in ranked list
        # always concordant
        Kt = 1

    return Kt


# FILE NAME (input)
#testfile = input("Test label file: ")
#modelscorefile = input("Test model score file: ")
num_file = 5
testfile = 'test' + str(num_file) + '.txt'
modelscorefile = 'result136_' + str(num_file) + '.txt'

# number of qid and docs
num_of_qid = 2000;
num_lines = sum(1 for line in open(testfile, "r",encoding="latin-1"))
print('num of docs:', num_lines)

# store list of Kendall tau for each qid
Kt_list = np.zeros(num_of_qid)

# temp store
list_test_score_temp = []
list_test_qid_temp = []
list_model_score_temp = []
comp_list_test = []
comp_list_model = []

# read test file and model score file
test_ins = open(testfile, "r",encoding="latin-1")
test_lines = test_ins.readlines()
model_ins = open(modelscorefile, "r",encoding="latin-1")
model_lines = model_ins.readlines()

            
# calclulate Kendall's tau for each qid
# report the mean

qid_temp = -1
qid_count = 0

for i in range(num_lines):

    testline = test_lines[i]
    modelline = model_lines[i]

    this_qid = int(testline.split()[1][4:])

    if i == 0:
        qid_temp = this_qid
        
    elif this_qid != qid_temp:
        Kt_list[int(qid_count)] = Kendalls_tau(list_test_score_temp, list_model_score_temp);

        #print(qid_count, Kt_list[qid_count])
        
        qid_count = qid_count+1

   
        # reset
        qid_temp = this_qid
        list_test_score_temp = []
        list_test_qid_temp = []
        list_model_score_temp = []
        comp_list_test = []
        comp_list_model = []

    # append score and qid to list
    list_test_score_temp = np.append(list_test_score_temp, int(testline.split()[0]))
    list_test_qid_temp = np.append(list_test_qid_temp, this_qid)
    list_model_score_temp = np.append(list_model_score_temp, float(modelline.split()[2]))


    if i == num_lines-1:
    
        Kt_list[int(qid_count)] = Kendalls_tau(list_test_score_temp, list_model_score_temp);

        #print(qid_count, Kt_list[qid_count])
        
        qid_count = qid_count+1

    i = i+1

print('Mean Kendall tau (alpha): ', np.mean(Kt_list))
