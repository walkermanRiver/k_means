import pandas as pd
import os
from normalization import normalize_corpus

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
# vectorizer, feature_matrix = build_feature_matrix(norm_book_content,
#                                                   feature_type='tfidf',
#                                                   min_df=0.2, max_df=0.90,
#                                                   ngram_range=(1, 2))

print("finish processing data")
