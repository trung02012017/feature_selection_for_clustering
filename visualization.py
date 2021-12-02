from config import *
from util import *


def run_visualize_data():
    for data_name, data_path in data_map.items():
        feature_pca, feature_umap, labels_trans, le = load_data(data_map["pge"])
        nor_feature_pca = normalize_data(feature_pca)
        nor_feature_umap = normalize_data(feature_umap)

        visualize_data(data=nor_feature_pca,
                       labels=labels_trans,
                       le=le,
                       n_components=2,
                       title=f"{data_name} pca data visualization using t-SNE",
                       filename=f"{data_name}_pca.png",
                       alg='pca')

        visualize_data(data=nor_feature_umap,
                       labels=labels_trans,
                       le=le,
                       n_components=2,
                       title=f"{data_name} umap data visualization using t-SNE",
                       filename=f"{data_name}_umap.png",
                       alg='tsne')


if __name__ == '__main__':
    run_visualize_data()