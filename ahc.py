import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from util import *
from config import *

import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering


results_pca = {
        "pge": {},
        "resnet50": {},
        "vgg16": {}
    }

results_umap = {
    "pge": {},
    "resnet50": {},
    "vgg16": {}
}


def draw_dendrogram(data, cut_distance, title, filename):
    dendrogram = sch.dendrogram(sch.linkage(data, method="ward"))
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.plot([0, sum(range(data.shape[0]))], [cut_distance]*2, 'b--')
    plt.title(title)
    plt.xlabel('Data points')
    plt.ylabel('Euclidean distances')
    plt.savefig(f"{result_path_ahc}/figures/dendrogram/{filename}")
    plt.clf()


def run_ahc():
    for data_name, data_path in data_map.items():
        print(f"Processing data {data_name} in path {data_path}")
        feature_pca, feature_umap, labels_trans, le = load_data(data_path)
        nor_feature_pca = normalize_data(feature_pca)
        nor_feature_umap = normalize_data(feature_umap)

        draw_dendrogram(nor_feature_pca,
                        cut_distance_ahc[data_name]['pca'],
                        f'Dendrogram for {data_name} pca data',
                        f'{data_name}_pca_dendrogram.png')
        draw_dendrogram(nor_feature_umap,
                        cut_distance_ahc[data_name]['umap'],
                        f'Dendrogram for {data_name} umap data',
                        f'{data_name}_umap_dendrogram.png')
        print(f"Done drawing dendrogram for data {data_name}")

        # pca feature
        ahc_pca = AgglomerativeClustering(n_clusters=num_cluster_ahc[data_name]['pca'],
                                          affinity='euclidean',
                                          linkage='ward')
        labels_pred_pca = ahc_pca.fit_predict(nor_feature_pca)

        fig_title_pca = f"AHC clustering results on {data_name} pca features " \
                        f"with {num_cluster_ahc[data_name]['pca']} clusters"
        fig_path_pca = f"{result_path_ahc}/figures/clusters/ahc_{data_name}_pca.png"
        avg_silhouette_score_pca, avg_v_score_pca = evaluate_result(data=nor_feature_pca,
                                                                    labels_pred=labels_pred_pca,
                                                                    labels_trans=labels_trans,
                                                                    le=le,
                                                                    fig_title=fig_title_pca,
                                                                    fig_path=fig_path_pca)

        results_pca[data_name]['optimal_clusters'] = int(num_cluster_ahc[data_name]['pca'])
        results_pca[data_name]['avg_silhouette_score'] = float(avg_silhouette_score_pca)
        results_pca[data_name]['avg_v_score'] = float(avg_v_score_pca)

        # umap feature
        ahc_umap = AgglomerativeClustering(n_clusters=num_cluster_ahc[data_name]['umap'],
                                           affinity='euclidean',
                                           linkage='ward')
        labels_pred_umap = ahc_umap.fit_predict(nor_feature_umap)

        fig_title_umap = f"AHC clustering results on {data_name} umap features " \
                         f"with {num_cluster_ahc[data_name]['umap']} clusters"
        fig_path_umap = f"{result_path_ahc}/figures/clusters/ahc_{data_name}_umap.png"
        avg_silhouette_score_umap, avg_v_score_umap = evaluate_result(data=nor_feature_umap,
                                                                      labels_pred=labels_pred_umap,
                                                                      labels_trans=labels_trans,
                                                                      le=le,
                                                                      fig_title=fig_title_umap,
                                                                      fig_path=fig_path_umap)

        results_umap[data_name]['optimal_clusters'] = int(num_cluster_ahc[data_name]['umap'])
        results_umap[data_name]['avg_silhouette_score'] = float(avg_silhouette_score_umap)
        results_umap[data_name]['avg_v_score'] = float(avg_v_score_umap)
        print(f"Done running AHC on data {data_name}")
        print("==============================================")

    json.dump(results_pca, open(f"{result_path_ahc}/evaluation/pca_feature_results.json", "w"))
    print(f"Done saving results for PCA features data in {result_path_ahc}/evaluation/pca_feature_results.json")
    json.dump(results_umap, open(f"{result_path_ahc}/evaluation/umap_feature_results.json", "w"))
    print(f"Done saving results for PCA features data in {result_path_ahc}/evaluation/umap_feature_results.json")

