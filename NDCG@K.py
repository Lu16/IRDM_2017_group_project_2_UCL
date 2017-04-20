import numpy as np
k = 10
txt = 1


def read_file():
    file_0 = np.loadtxt('D:/UCL/IRDM/' + 'myscorefile' + str(txt) + '.txt')
    rel = np.loadtxt('D:/UCL/IRDM/MSLR-WEB10K/Fold'+str(txt)+'/test.txt', delimiter=' ', usecols=0, unpack=True)
    qid = file_0[:, 0]
    score = file_0[:, 2]
    # rel = file_1[:, 0]
    score_rel = np.append(np.reshape(score, [-1, 1]), np.reshape(rel, [-1, 1]), axis=1)
    return qid, score_rel


def cal_dcg(score_rel):
    dcg = 0
    # sort docs list by score
    score_rel = (score_rel[score_rel[:, 0].argsort()[::-1]])
    # calculate gain (the algorithm to cal relevance in Ranklib is to use the power of 2
    gain = np.power(2, score_rel[:, 1]) - 1
    # calculate accumulated dcg
    for i in range(min(len(gain), k)):
        # accumulate dcg of k docs for this query
        dcg += (1/(np.log2(i + 2))) * gain[i]
    return dcg


def cal_ideal_dcg(score_rel):
    ideal_dcg = 0
    # sort docs list by relevance
    score_rel = (score_rel[score_rel[:, 1].argsort()[::-1]])
    # calculate gain (the algorithm to cal relevance in Ranklib is to use the power of 2
    gain = np.power(2, score_rel[:, 1]) - 1
    # calculate accumulated ideal_dcg
    for i in range(min(len(gain), k)):
        # accumulate dcg of k docs for this query
        ideal_dcg += (1/(np.log2(i + 2))) * gain[i]
    return ideal_dcg


def main():
    # read file
    qid, score_rel = read_file()

    # initialize variables
    ndcg = []
    this_q_start = 0
    this_q_end = 0

    '''calculate ndcg for each qeury'''
    for q in range(len(qid)):
        # if this is next query ==> calculate ndcg for previous query
        if qid[q] != qid[this_q_start] or q == len(qid)-1:
            # calculate & store this ndcg
            dcg = cal_dcg(score_rel[this_q_start:this_q_end, :])
            idcg = cal_ideal_dcg(score_rel[this_q_start:this_q_end, :])
            if dcg == 0:
                ndcg = np.append(ndcg, 0)
            else:
                ndcg = np.append(ndcg, dcg/idcg)
            # re-initial variables
            this_q_start = q
            this_q_end = q + 1
        # if we have k docs ==> ignore the rest docs
        elif this_q_end - this_q_start == k:
            this_q_end += 1
            continue
        # store data for this query
        else:
            this_q_end += 1

    str_ndcg = np.repeat(np.reshape([str('NDCG@10')], [-1, 1]), int(len(np.unique(qid))))
    np.savetxt('D:/UCL/IRDM/test_ndcg' + str(txt) + '.txt',  np.c_[str_ndcg, (np.unique(qid)).astype(np.int), ndcg], delimiter='     ', newline='\r\n', fmt='%s')
    # write mean NDCG@k
    fh = open('D:/UCL/IRDM/test_ndcg' + str(txt) + '.txt', 'a')
    fh.write('\r\nall   ' + str(np.mean(ndcg)))
    print('ndcg for txt' + str(txt), np.mean(ndcg))


if __name__ == '__main__':
    main()




