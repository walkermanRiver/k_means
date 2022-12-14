# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import random
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.font_manager import FontProperties

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ml import training_data_feature_constant as const_value


def build_feature_matrix(documents, feature_type='frequency',
                         ngram_range=(1, 1), min_df=0.0, max_df=1.0):
    feature_type = feature_type.lower().strip()  # feature_type为tfidf

    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=min_df,
                                     max_df=max_df, ngram_range=ngram_range)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")

    feature_matrix = vectorizer.fit_transform(documents).astype(float)

    return vectorizer, feature_matrix


# KMeans++
def k_means(feature_matrix, num_clusters=10):
    km = KMeans(n_clusters=num_clusters,
                max_iter=10000)  # km打印结果是KMeans的参数
    km.fit(feature_matrix)
    clusters = km.labels_
    return km, clusters


# def get_cluster_data(clustering_obj, training_data,
#                      feature_names, num_clusters,
#                      topn_features=10):
#     cluster_details = {}
#     # 获取cluster的center
#     ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
#     # 获取每个cluster的关键特征
#     # 获取每个cluster的id
#     for cluster_num in range(num_clusters):
#         cluster_details[cluster_num] = {}
#         cluster_details[cluster_num]['cluster_num'] = cluster_num
#         key_features = [feature_names[index]
#                         for index
#                         in ordered_centroids[cluster_num, :topn_features]]
#         cluster_details[cluster_num]['key_features'] = key_features
#
#         # books = training_data[training_data['Cluster'] == cluster_num]['title'].values.tolist()
#         data_id = training_data[training_data[const_value.ML_CLUSTER_ID_COLUMN] == cluster_num][
#             const_value.ID_COLUMN].values.tolist()
#         cluster_details[cluster_num]['data_id'] = data_id
#
#     return cluster_details


def get_result_with_key_feature(clustering_obj, training_data,
                                feature_names, num_clusters,
                                topn_features=10):
    result_data = training_data
    result_data[const_value.ML_FEATURE_NAME1_COLUMN] = None

    # # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # # 获取每个cluster的关键特征
    # # 获取每个cluster的id
    for cluster_num in range(num_clusters):
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]

        current_cluster_data_index = training_data[training_data[const_value.ML_CLUSTER_ID_COLUMN]
                                                   == cluster_num].index.to_list()
        training_data.loc[current_cluster_data_index, const_value.ML_FEATURE_NAME1_COLUMN] \
            = key_features[0]

    return result_data


# def print_cluster_data(cluster_data):
#     # print ml details
#     for cluster_num, cluster_details in cluster_data.items():
#         print('Cluster {} details:'.format(cluster_num))
#         print('-' * 20)
#         print('Key features:', cluster_details['key_features'])
#         print('data in this ml:')
#         for id_Str in cluster_details['data_id']:
#             print("     " + id_Str)
#         # print(', '.join(cluster_details['data_id']))
#         print('=' * 40)


# def print_cluster_data_table(cluster_data):
#     # print ml details
#     for cluster_num, cluster_details in cluster_data.items():
#         for id_Str in cluster_details['data_id']:
#             print(id_Str + "," + cluster_details['key_features'][0])
#         # print(', '.join(cluster_details['data_id']))


def plot_clusters(feature_matrix,
                  cluster_data, book_data,
                  plot_size=(16, 8)):
    # generate random color for clusters
    def generate_random_color():
        color = '#%06x' % random.randint(0, 0xFFFFFF)
        return color

    # define markers for clusters
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    # build cosine distance matrix
    cosine_distance = 1 - cosine_similarity(feature_matrix)
    # dimensionality reduction using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed",
              random_state=1)
    # get coordinates of clusters in new low-dimensional space
    plot_positions = mds.fit_transform(cosine_distance)
    x_pos, y_pos = plot_positions[:, 0], plot_positions[:, 1]
    # build ml plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign ml features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique ml label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'data_id': book_data['data_id'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each ml using co-ordinates and book titles
    for cluster_num, cluster_frame in grouped_plot_frame:
        marker = markers[cluster_num] if cluster_num < len(markers) \
            else np.random.choice(markers, size=1)[0]
        ax.plot(cluster_frame['x'], cluster_frame['y'],
                marker=marker, linestyle='', ms=12,
                label=cluster_name_map[cluster_num],
                color=cluster_color_map[cluster_num], mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off',
                       labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off',
                       labelleft='off')
    font_properties = FontProperties()
    font_properties.set_size('small')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), fancybox=True,
              shadow=True, ncol=5, numpoints=1, prop=font_properties)
    # add labels as the film titles
    for index in range(len(cluster_plot_frame)):
        ax.text(cluster_plot_frame.ix[index]['x'],
                cluster_plot_frame.ix[index]['y'],
                cluster_plot_frame.ix[index]['data_id'], size=8)
        # show the plot
    plt.show()
