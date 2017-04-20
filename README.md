# IRDM_2017_GroupProject2_Group32_UCL

Code for Deep Learning Models 
---------------------------------------
XgBoost.py 		----xgboost classifier(unable to run due to unfixed bug of xgboost classifier in multiclass case)


LSTM.py ----code for training LSTM and obtaining validation or test score 

GRUmodel.py		----code for training and saving 32 unit GRU model with 137 features including qid

GRUmodel_load&score.py	----code for loading, testing and generating the test relevance score of 32 unit GRU model with 137 features

GRUmodel_noqid.py	----code for training and saving 32 unit GRU model with 136 features not including qid

GRUmodel_noqid_load&score.py----code for loading, testing and generating the test relevance score of 32 unit GRU model with 136 features

CNN_qid.py			----code for training ConvNet with 137 features including qid

CNN_no_qid.py			----code for training ConvNet with 136 features not including qid

Code for Metrics & Logistic Regression
------------------------------------------

Kendalltau.py		----code for calculating mean Kendall's tau score on all queries in the test data given the test model score and the true test score

NDCG@K.py		---- code for calculating mean NDCG@K value of all queries within a specified file

cal_MAP.py  ----code for calculating MAP value of all queries for a score file

LR_TF.py		----logistic regression implemented by tensorflow

LR.py  ----code for logistic regression implemented by numpy

Code for Data Processing
------------------------------------------
implement_pca.py  ----code for implement PCA

obtain_feature_value.java ----code for obtaining feature value of train, validation and test data set

split_data.java ----code for splitting whole data to 1/4 data set
