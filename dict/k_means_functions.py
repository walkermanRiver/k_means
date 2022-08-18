# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
# import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import random
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.font_manager import FontProperties

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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
    clusters = km.labels_  # 编号 [4 6 6 ... 2 2 2]
    return km, clusters


def get_cluster_data(clustering_obj, book_data,
                     feature_names, num_clusters,
                     topn_features=10):
    cluster_details = {}
    # 获取cluster的center
    ordered_centroids = clustering_obj.cluster_centers_.argsort()[:, ::-1]
    # 获取每个cluster的关键特征
    # 获取每个cluster的书
    for cluster_num in range(num_clusters):
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster_num'] = cluster_num
        key_features = [feature_names[index]
                        for index
                        in ordered_centroids[cluster_num, :topn_features]]
        cluster_details[cluster_num]['key_features'] = key_features

        books = book_data[book_data['Cluster'] == cluster_num]['title'].values.tolist()
        cluster_details[cluster_num]['books'] = books

    return cluster_details


def print_cluster_data(cluster_data):
    # print cluster details
    for cluster_num, cluster_details in cluster_data.items():
        print('Cluster {} details:'.format(cluster_num))
        print('-' * 20)
        print('Key features:', cluster_details['key_features'])
        print('book in this cluster:')
        print(', '.join(cluster_details['books']))
        print('=' * 40)


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
    # build cluster plotting data
    cluster_color_map = {}
    cluster_name_map = {}
    for cluster_num, cluster_details in cluster_data[0:500].items():
        # assign cluster features to unique label
        cluster_color_map[cluster_num] = generate_random_color()
        cluster_name_map[cluster_num] = ', '.join(cluster_details['key_features'][:5]).strip()
    # map each unique cluster label with its coordinates and books
    cluster_plot_frame = pd.DataFrame({'x': x_pos,
                                       'y': y_pos,
                                       'label': book_data['Cluster'].values.tolist(),
                                       'title': book_data['title'].values.tolist()
                                       })
    grouped_plot_frame = cluster_plot_frame.groupby('label')
    # set plot figure size and axes
    fig, ax = plt.subplots(figsize=plot_size)
    ax.margins(0.05)
    # plot each cluster using co-ordinates and book titles
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
                cluster_plot_frame.ix[index]['title'], size=8)
        # show the plot
    plt.show()
