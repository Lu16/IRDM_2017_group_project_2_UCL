# IRDM_2017_group_project_2_UCL
IRDM 2017 group project 2 at UCL

Code for Deep Learning Models 
---------------------------------------
XgBoost.py 		----xgboost classifier(unable to run due to unfixed bug of xgboost classifier in multiclass case)

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

LR_TF.py		----logistic regression implemented by tensorflow

