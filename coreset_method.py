import time
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import pandas as pd

# Parameters
CORESET_RATIO = 0.2
NUM_CLUSTERS = 10

def random_coreset(dataset, ratio=CORESET_RATIO):
    start = time.time()
    indices = np.random.choice(len(dataset), int(len(dataset) * ratio), replace=False)
    subset_time = time.time() - start
    return torch.utils.data.Subset(dataset, indices), subset_time

def clustering_coreset(dataset, ratio=CORESET_RATIO, num_clusters=NUM_CLUSTERS):
    start = time.time()
    # Cluster based on user interaction count (better than ID; for representation)
    user_counts = pd.Series(dataset.users.numpy()).value_counts().sort_index()
    features = user_counts.values.reshape(-1, 1)  # Use count as feature
    kmeans = KMeans(n_clusters=num_clusters)
    labels = kmeans.fit_predict(features)
    selected = []
    for c in range(num_clusters):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) > 0:
            num_select = int(len(cluster_indices) * ratio)
            selected.extend(np.random.choice(cluster_indices, min(num_select, len(cluster_indices)), replace=False))
    subset_time = time.time() - start
    return torch.utils.data.Subset(dataset, selected), subset_time

def gradient_coreset(dataset, model, ratio=CORESET_RATIO):
    start = time.time()
    gradients = []
    criterion = nn.BCELoss()
    for i in range(len(dataset)):
        u, it, l = dataset[i]
        u, it, l = u.unsqueeze(0), it.unsqueeze(0), l.unsqueeze(0)
        pred = model(u, it)
        loss = criterion(pred, l)
        grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        grad_norm = sum(g.norm() for g in grads if g is not None)
        gradients.append(grad_norm.item())
    indices = np.argsort(gradients)[-int(len(dataset) * ratio):]  # High gradients
    subset_time = time.time() - start
    return torch.utils.data.Subset(dataset, indices), subset_time