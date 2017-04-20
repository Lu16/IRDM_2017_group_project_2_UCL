from sklearn.decomposition import PCA
import numpy as np

file_ori_train = np.loadtxt('D:/DM_sub_data/Folder1/train_feature_1_4_two_id.txt')

file_ori_vali = np.loadtxt('D:/DM_sub_data/Folder1/vali_feature_two_id.txt')

file_train = file_ori_train[:, 2:]

file_vali = file_ori_vali[:, 2:]

pca = PCA(n_components=136)

pca.fit(file_train)

file_train_pca = pca.transform(file_train)

file_vali_pca = pca.transform(file_vali)

file_train_out = np.hstack((file_ori_train[:, 0:2].astype(float), file_train_pca[:, :].astype(float)))

file_vali_out = np.hstack((file_ori_vali[:, 0:2].astype(float), file_vali_pca[:, :].astype(float)))

np.savetxt('D:/DM_sub_data/Folder1/train_feature_1_4_136pca_two_id.txt', file_train_out)

np.savetxt('D:/DM_sub_data/Folder1/vali_feature_136pca_two_id.txt', file_vali_out)



