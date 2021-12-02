import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import v_measure_score

from yellowbrick.cluster import KElbowVisualizer


def load_data(path):
    data_file = h5py.File(path, "r")
    data_pca = np.array(list(data_file[pca_feature]))
    data_umap = np.array(list(data_file[umap_feature]))

    labels = []
    for name in data_file['file_name']:
        label = name.decode("utf-8").split("/")[-2]
        labels.append(label)

    le = LabelEncoder()
    le.fit(labels)

    labels_trans = le.transform(labels)

    return data_pca, data_umap, labels_trans, le


def visualize_data(data, labels, le, n_components, title, filename, alg):

    true_labels = le.inverse_transform(labels)

    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data_nor = (data - data_min) / (data_max - data_min)

    if alg == "pca":
        pca = PCA(n_components=n_components)
        pca.fit(data_nor)
        data_pc = pca.transform(data_nor)

    if alg == "tsne":
        tsne = TSNE(n_components=n_components, perplexity=40)
        data_pc = tsne.fit_transform(data_nor)

    df = pd.DataFrame(dict(pc0=data_pc[:, 0], pc1=data_pc[:, 1], labels=true_labels))

    ax = sns.scatterplot('pc0', 'pc1', data=df, hue='labels', legend="brief")
    plt.title(title)
    plt.savefig(f"results/data_visualization/{filename}")
    print(f"Done saving figure to results/data_visualization/{filename}")
    plt.clf()


def normalize_data(data):
    data_min = np.amin(data, axis=0)
    data_max = np.amax(data, axis=0)
    data_nor = (data - data_min) / (data_max - data_min)

    return data_nor


def cal_elbow(model, x, k_range, file_name):
    visualizer = KElbowVisualizer(model,
                                  k=k_range,
                                  metric='calinski_harabasz',
                                  timings=True,
                                  locate_elbow=True)
    visualizer.fit(x)
    visualizer.show(outpath=file_name, clear_figure=True)

    return visualizer.elbow_value_


def evaluate_result(data,
                    labels_pred,
                    labels_trans,
                    le,
                    fig_title,
                    fig_path):
    avg_silhouette_score = silhouette_score(data, labels_pred)
    avg_v_score = v_measure_score(labels_trans, labels_pred)

    result_df = pd.DataFrame({"clusters": labels_pred, "labels": le.inverse_transform(labels_trans)})

    count_label_df = result_df.groupby(['clusters', 'labels']).agg({'labels': 'count'})
    prob_label_df = count_label_df.groupby(level=0)["labels"].apply(lambda x: 100 * x / float(x.sum()))
    ax = prob_label_df.unstack().plot(kind='bar', stacked=True, colormap=plt.get_cmap('tab10'))
    ax.legend(loc="lower right")
    ax.set_ylabel("Tissue Abundance Percentage")
    ax.set_title(fig_title)
    plt.savefig(fig_path)
    plt.clf()

    return avg_silhouette_score, avg_v_score
