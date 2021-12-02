inceptionv3_feature_path = "data/colon_nct_feature/inceptionv3_dim_reduced_feature.h5"
pge_feature_path = "data/colon_nct_feature/pge_dim_reduced_feature.h5"
resnet50_feature_path = "data/colon_nct_feature/resnet50_dim_reduced_feature.h5"
vgg16_feature_path = "data/colon_nct_feature/vgg16_dim_reduced_feature.h5"


data_map = {
        "pge": pge_feature_path,
        "resnet50": resnet50_feature_path,
        "vgg16": vgg16_feature_path
    }

pca_feature = 'pca_feature'
umap_feature = 'umap_feature'

n_clusters_lower = 9
n_cluster_upper = 25

max_iter = 5000
result_path = "results/kmeans"

result_path_ahc = "results/ahc"
cut_distance_ahc = {
    "pge": {
        "pca": 8,
        "umap": 15
    },
    "resnet50":  {
        "pca": 10.5,
        "umap": 19
    },
    "vgg16":  {
        "pca": 9,
        "umap": 23
    }
}

num_cluster_ahc = {
    "pge": {
        "pca": 9,
        "umap": 14
    },
    "resnet50":  {
        "pca": 10,
        "umap": 11
    },
    "vgg16":  {
        "pca": 10,
        "umap": 10
    }
}

random_state = 2021