import pandas as pd
import os
from normalization import normalize_corpus
from ml import k_means_functions as kf
from collections import Counter

# 第一步：读取数据
book_data = pd.read_csv('data/data.csv')  # 读取文件
# ,sep=",",error_bad_lines=False,engine='python',encoding='utf-8'
print(book_data.head())  # 2822 rows x 5 columns
print("csv数据", os.linesep, book_data)

book_content = book_data['content'].tolist()

# 第二步：数据载入、分词
# 返回的是分词后的集合['现代人 内心 流失 的 东西……','在 第一次世界大战 的……'，……]
norm_book_content = normalize_corpus(book_content)

# 第三步：提取 tf-idf 特征
vectorizer, feature_matrix = kf.build_feature_matrix(norm_book_content,
                                                     feature_type='tfidf',
                                                     min_df=0.2, max_df=0.90,
                                                     ngram_range=(1, 2))
# 查看特征数量
# 得到tf-idf矩阵，稀疏矩阵表示法  (2822, 16281) 2822行，16281个词汇（根据16281个词汇建立词的索引）
print(feature_matrix.shape)

# 获取特征名字
# 显示所有文本的词汇，列表类型
feature_names = vectorizer.get_feature_names()

# 设置k=10,聚出10个类别
num_clusters = 10
km_obj, clusters = kf.k_means(feature_matrix=feature_matrix,
                              num_clusters=num_clusters)

# 在原先的csv文本中加入一列Cluster后的数字
book_data['Cluster'] = clusters

# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())

# 第六步：查看结果
cluster_data = kf.get_cluster_data(clustering_obj=km_obj,
                                   book_data=book_data,
                                   feature_names=feature_names,
                                   num_clusters=num_clusters,
                                   topn_features=5)

kf.print_cluster_data(cluster_data)

print("finish processing data")
