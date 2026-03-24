"""
Clustering utilities.
"""

def simple_kmeans(data, k=3, max_iter=100):
    """Simple K-means clustering."""
    import numpy as np
    centroids = data[:k]
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters if cluster]
        if len(new_centroids) == k:
            centroids = new_centroids
    return centroids, clusters
