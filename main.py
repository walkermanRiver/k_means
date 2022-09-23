import pandas as pd
import os
from normalization import normalize_corpus
from ml import k_means_functions as kf
from collections import Counter
from ml import training_data_feature_constant as const_value

# 第一步：读取数据
training_data = pd.read_csv(const_value.TRAINING_DATA_FILE).astype(str)  # 读取文件

# ,sep=",",error_bad_lines=False,engine='python',encoding='utf-8'
print(training_data.head())
print("csv数据", os.linesep, training_data)


merged_content = training_data[const_value.CONTENT_COLUMN_SUMMARY].map(str). \
        str.cat([training_data[const_value.CONTENT_COLUMN_DESCRIPTION]], sep=' ')
data_content = merged_content.tolist()


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
training_data[const_value.ML_CLUSTER_ID_COLUMN] = clusters

# 获取每个cluster的数量
c = Counter(clusters)
print(c.items())

# 第六步：查看结果
result_data = kf.get_result_with_key_feature(clustering_obj=km_obj,
                                             training_data=training_data,
                                             feature_names=feature_names,
                                             num_clusters=num_clusters,
                                             topn_features=1)

result_data.to_csv("output/output.csv", encoding="utf-8")
# cluster_data = kf.get_cluster_data(clustering_obj=km_obj,
#                                    training_data=training_data,
#                                    feature_names=feature_names,
#                                    num_clusters=num_clusters,
#                                    topn_features=1)

# kf.print_cluster_data(cluster_data)
# kf.print_cluster_data_table(cluster_data)
print("succeed to finish processing data")


