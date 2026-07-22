import math
import torch

def get_metrics(targets, results, device, k):
    metrics = torch.zeros([2]).to(device)
    hits, batch = hit_at_k(targets, results, k)
    ndcg, _ = ndcg_at_k(targets, results, k)
    metrics[0] = hits
    metrics[1] = ndcg
    return metrics, batch

def hit_at_k(labels, results, k):
    hit = 0.0
    batch = results.shape[0]
    for i in range(batch):
        if labels[i] in results[i, :k]: hit += 1
    return hit, batch

def ndcg_at_k(labels, results, k):
    ndcg = 0.0
    batch = results.shape[0]
    for i in range(batch):
        rel = torch.where(results[i, :k] == labels[i], 1, 0)
        for j in range(len(rel)):
            ndcg += rel[j] / math.log(j+2, 2)
    return ndcg, batch
