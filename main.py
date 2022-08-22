import pandas as pd
import os
from normalization import normalize_corpus
from ml import k_means_functions as kf
from collections import Counter
from ml import training_data_feature_constant

# 第一步：读取数据
training_data = pd.read_csv(training_data_feature_constant.TRAINING_DATA_FILE).astype(str)   # 读取文件
content_column = training_data_feature_constant.CONTENT_COLUMN
# training_data = pd.read_csv('data/data.csv')  # 读取文件
# content_column = "content"

# ,sep=",",error_bad_lines=False,engine='python',encoding='utf-8'
print(training_data.head())
print("csv数据", os.linesep, training_data)


data_content = training_data[content_column].tolist()
# data_content = [str(a) for a in training_data[content_column].toList()]

# 第二步：数据载入、分词
norm_data_content = normalize_corpus(data_content)

# 第三步：提取 tf-idf 特征
vectorizer, feature_matrix = kf.build_feature_matrix(norm_data_content,
                                                     feature_type='tfidf',
                                                     min_df=0.2, max_df=0.90,
                                                     ngram_range=(1, 2))
# 查看特征数量
print(feature_matrix.shape)

# 获取特征名字
# 显示所有文本的词汇，列表类型
feature_names = vectorizer.get_feature_names()

# 聚出n个类别
num_clusters = 10
km_obj, clusters = kf.k_means(feature_matrix=feature_matrix,
                              num_clusters=num_clusters)

# 在原先的csv文本中加入一列Cluster后的数字
training_data['Cluster'] = clusters

# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())

# 第六步：查看结果
cluster_data = kf.get_cluster_data(clustering_obj=km_obj,
                                   training_data=training_data,
                                   feature_names=feature_names,
                                   num_clusters=num_clusters,
                                   topn_features=5)

kf.print_cluster_data(cluster_data)

print("finish processing data")
