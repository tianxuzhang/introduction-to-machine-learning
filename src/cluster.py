from sklearn.datasets import make_blobs
from sklearn.cluster import (
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    KMeans,
    MeanShift,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)

# 生成合成数据集
X, _ = make_blobs(n_samples=1000, centers=4, random_state=42)

models = [
    AgglomerativeClustering(),
    Birch(),
    DBSCAN(eps=0.5, min_samples=5),
    KMeans(n_clusters=4),
    MeanShift(),
    MiniBatchKMeans(n_clusters=4),
    OPTICS(),
    SpectralClustering(n_clusters=4),
]

for model in models:
    # 进行聚类
    y_pred = model.fit_predict(X)

    # 打印每个簇的样本数量
    cluster_counts = {}
    for label in y_pred:
        if label in cluster_counts:
            cluster_counts[label] += 1
        else:
            cluster_counts[label] = 1
    print(f"{model.__class__.__name__}: Cluster Counts: {cluster_counts}")