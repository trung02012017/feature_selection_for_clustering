import json
import matplotlib.pyplot as plt
import pandas as pd

from config import *
from util import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import v_measure_score


def run_kmeans():

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

    for data_name, data_path in data_map.items():
        print(f"Processing data {data_name} in path {data_path}")
        feature_pca, feature_umap, labels_trans, le = load_data(data_path)
        nor_feature_pca = normalize_data(feature_pca)
        nor_feature_umap = normalize_data(feature_umap)

        # pca feature

        # choose k value by kmeans
        elbow_value_pca = cal_elbow(KMeans(),
                                    nor_feature_pca,
                                    (n_clusters_lower, n_cluster_upper),
                                    f"{result_path}/figures/elbow/{data_name}_pca_elbow.png")

        # run kmeans with chosen k
        kmeans_pca = KMeans(n_clusters=elbow_value_pca, max_iter=max_iter, random_state=random_state)
        labels_pred_pca = kmeans_pca.fit_predict(nor_feature_pca)

        # evaluate and illustrate results
        fig_title_pca = f"kmeans clustering results on {data_name} pca features with {elbow_value_pca} clusters"
        fig_path_pca = f"{result_path}/figures/clusters/kmeans_{data_name}_pca.png"
        avg_silhouette_score_pca, avg_v_score_pca = evaluate_result(data=nor_feature_pca,
                                                                    labels_pred=labels_pred_pca,
                                                                    labels_trans=labels_trans,
                                                                    le=le,
                                                                    fig_title=fig_title_pca,
                                                                    fig_path=fig_path_pca)

        results_pca[data_name]['optimal_clusters'] = int(elbow_value_pca)
        results_pca[data_name]['avg_silhouette_score'] = float(avg_silhouette_score_pca)
        results_pca[data_name]['avg_v_score'] = float(avg_v_score_pca)

        # umap feature

        # choose k value by kmeans
        elbow_value_umap = cal_elbow(KMeans(),
                                     nor_feature_umap,
                                     (n_clusters_lower, n_cluster_upper),
                                     f"{result_path}/figures/elbow/{data_name}_umap_elbow.png")

        # run kmeans with chosen k
        kmeans_umap = KMeans(n_clusters=elbow_value_umap, max_iter=max_iter)
        labels_pred_umap = kmeans_umap.fit_predict(nor_feature_umap)

        fig_title_umap = f"kmeans clustering results on {data_name} umap features with {elbow_value_umap} clusters"
        fig_path_umap = f"{result_path}/figures/clusters/kmeans_{data_name}_umap.png"
        avg_silhouette_score_umap, avg_v_score_umap = evaluate_result(data=nor_feature_umap,
                                                                      labels_pred=labels_pred_umap,
                                                                      labels_trans=labels_trans,
                                                                      le=le,
                                                                      fig_title=fig_title_umap,
                                                                      fig_path=fig_path_umap)

        # evaluate and illustrate results
        results_umap[data_name]['optimal_clusters'] = int(elbow_value_umap)
        results_umap[data_name]['avg_silhouette_score'] = float(avg_silhouette_score_umap)
        results_umap[data_name]['avg_v_score'] = float(avg_v_score_umap)

        print(f"Done running Kmeans on data {data_name}")
        print("==============================================")

    json.dump(results_pca, open(f"{result_path}/evaluation/pca_feature_results.json", "w"))
    print(f"Done saving results for PCA features data in {result_path}/evaluation/pca_feature_results.json")
    json.dump(results_umap, open(f"{result_path}/evaluation/umap_feature_results.json", "w"))
    print(f"Done saving results for PCA features data in {result_path}/evaluation/umap_feature_results.json")


