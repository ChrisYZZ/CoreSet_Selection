import numpy as np
import random
import torch
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.stats import entropy, kendalltau
from torch.utils.data import DataLoader
import torch.nn as nn

# Parameters
K = 10
BATCH_SIZE = 256

def evaluate(model, test_ds, num_items, user_interacted, k=K, num_neg_eval=99):
    model.eval()
    hr_scores, ndcg_scores = [], []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            u, i, _ = test_ds[idx]
            u_id = u.item()
            pos_i = i.item()

            # Sample num_neg_eval negatives (not interacted)
            negatives = []
            while len(negatives) < num_neg_eval:
                cand = random.randint(0, num_items - 1)
                if cand not in user_interacted.get(u_id, set()):
                    negatives.append(cand)
            items_to_rank = [pos_i] + negatives
            u_tensor = torch.full((len(items_to_rank),), u_id, dtype=torch.long)
            i_tensor = torch.tensor(items_to_rank, dtype=torch.long)
            scores = model(u_tensor, i_tensor).numpy()

            # Rank: sort by score desc
            ranked_indices = np.argsort(-scores)  # Descending
            rank = np.where(ranked_indices == 0)[0][0] + 1  # Pos_i is index 0, rank 1-based

            # HR@K (== Recall@K for single positive)
            hr = 1 if rank <= k else 0
            hr_scores.append(hr)

            # NDCG@K (manual, since single relevant)
            if rank <= k:
                dcg = 1 / np.log2(rank + 1)
            else:
                dcg = 0
            idcg = 1 / np.log2(2)  # Ideal: rank 1
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

    return np.mean(hr_scores), np.mean(ndcg_scores)

class CoresetEvaluator:
    def __init__(self):
        pass

    def _get_subset_data(self, coreset, full_dataset):
        """Helper to extract users/items from Subset or Dataset"""
        if isinstance(coreset, torch.utils.data.Subset):
            users = coreset.dataset.users[coreset.indices].numpy()
            items = coreset.dataset.items[coreset.indices].numpy()
        else:
            users = coreset.users.numpy()
            items = coreset.items.numpy()
        full_users = full_dataset.users.numpy()
        full_items = full_dataset.items.numpy()
        return users, items, full_users, full_items

    def evaluate_q1_value_metrics(self, coreset, full_dataset, model):
        """Q1: 评估交互价值指标"""
        metrics = {}
        # 梯度影响力 (average norm)
        metrics['gradient_norm'] = self.compute_gradient_norms(coreset, model)
        # 预测不确定性 (variance of preds)
        metrics['prediction_variance'] = self.compute_prediction_variance(coreset, model)
        # 交互稀有度 (average 1/freq of items)
        metrics['rarity_score'] = self.compute_rarity_scores(coreset, full_dataset)
        return metrics

    def compute_gradient_norms(self, coreset, model):
        model.eval()
        grads = []
        criterion = nn.BCELoss()
        loader = DataLoader(coreset, batch_size=BATCH_SIZE)
        for u, i, l in loader:
            pred = model(u, i)
            loss = criterion(pred, l)
            # Use autograd.grad to avoid backward warning
            model_grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grad_norm = sum(g.norm() for g in model_grads if g is not None)
            grads.append(grad_norm.item())
            # Reset grads to break cycle
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = None
        return np.mean(grads) if grads else 0

    def compute_prediction_variance(self, coreset, model):
        model.eval()
        preds = []
        loader = DataLoader(coreset, batch_size=BATCH_SIZE)
        for u, i, _ in loader:
            pred = model(u, i)
            preds.extend(pred.detach().numpy())
        return np.var(preds) if preds else 0

    def compute_rarity_scores(self, coreset, full_dataset):
        _, items, _, full_items = self._get_subset_data(coreset, full_dataset)
        item_counts = np.bincount(full_items, minlength=max(full_items)+1)
        rarity = [1 / item_counts[it] if item_counts[it] > 0 else 0 for it in items]
        return np.mean(rarity)

    def evaluate_q2_representativeness(self, coreset, full_dataset):
        """Q2: 量化代表性"""
        users, _, full_users, _ = self._get_subset_data(coreset, full_dataset)
        # KL散度 (user dist)
        kl_div = self.compute_kl_divergence(users, full_users)
        # 嵌入空间覆盖 (unique ratio for users)
        coverage = self.compute_embedding_coverage(users, full_users)  # User-based
        # MMD距离 (with sampling)
        mmd = self.compute_mmd(users.reshape(-1, 1), full_users.reshape(-1, 1))
        return {'kl_divergence': kl_div, 'coverage': coverage, 'mmd': mmd}

    def compute_kl_divergence(self, subset_data, full_data):
        hist_full, bins = np.histogram(full_data, bins=100, density=True)
        hist_sub, _ = np.histogram(subset_data, bins=bins, density=True)
        hist_full += 1e-10  # Avoid zero
        hist_sub += 1e-10
        return entropy(hist_full, hist_sub)

    def compute_embedding_coverage(self, subset_data, full_data):
        # Simple: ratio of unique in sub / full (for users/items)
        return len(np.unique(subset_data)) / len(np.unique(full_data)) if len(np.unique(full_data)) > 0 else 0

    def compute_mmd(self, subset_data, full_data):
        # RBF kernel MMD with sampling to avoid memory error
        sample_size = 1000
        if len(subset_data) > sample_size:
            sub_indices = np.random.choice(len(subset_data), sample_size, replace=False)
            subset_data = subset_data[sub_indices]
        if len(full_data) > sample_size:
            full_indices = np.random.choice(len(full_data), sample_size, replace=False)
            full_data = full_data[full_indices]
        gamma = 1.0 / subset_data.shape[1]
        xx = pairwise_kernels(subset_data, subset_data, metric='rbf', gamma=gamma)
        yy = pairwise_kernels(full_data, full_data, metric='rbf', gamma=gamma)
        xy = pairwise_kernels(subset_data, full_data, metric='rbf', gamma=gamma)
        return np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)

    def evaluate_q3_importance(self, coreset, full_dataset):
        """Q3: 用户/物品重要性分析"""
        users, items, full_users, full_items = self._get_subset_data(coreset, full_dataset)
        # 计算保留的用户/物品比例
        user_coverage = len(set(users)) / len(set(full_users)) if len(set(full_users)) > 0 else 0
        item_coverage = len(set(items)) / len(set(full_items)) if len(set(full_items)) > 0 else 0
        # 分析活跃度分布 (correlation of interaction counts)
        activity_correlation = self.analyze_activity_correlation(users, full_users)
        return {'user_coverage': user_coverage,
                'item_coverage': item_coverage,
                'activity_correlation': activity_correlation}

    def analyze_activity_correlation(self, subset_users, full_users):
        sub_counts = np.bincount(subset_users, minlength=max(full_users)+1)
        full_counts = np.bincount(full_users, minlength=max(full_users)+1)
        min_len = min(len(sub_counts), len(full_counts))
        return np.corrcoef(sub_counts[:min_len], full_counts[:min_len])[0, 1]

    def evaluate_q4_diversity(self, coreset, model):
        """Q4: 多样性与冗余度量 (need model for embeds)"""
        if isinstance(coreset, torch.utils.data.Subset):
            users = coreset.dataset.users[coreset.indices]
            items = coreset.dataset.items[coreset.indices]
        else:
            users = coreset.users
            items = coreset.items
        # Sample for large coreset to avoid memory
        sample_size = 1000
        if len(users) > sample_size:
            indices = np.random.choice(len(users), sample_size, replace=False)
            users = users[indices]
            items = items[indices]
        embeds = model.get_user_item_embeds(users, items)
        # 计算成对相似度
        pairwise_sim = cosine_similarity(embeds)
        # 计算多样性指标（如DPP似然 approx det(gram))
        gram = pairwise_sim + 1e-5 * np.eye(len(pairwise_sim))  # Stabilize
        diversity_score = np.linalg.det(gram)
        return {'redundancy': np.mean(pairwise_sim), 'diversity': diversity_score}

    def evaluate_q5_estimation_quality(self, heuristic_scores, model_scores):
        """Q5: 评估启发式vs模型驱动估计"""
        # 计算相关性
        correlation = np.corrcoef(heuristic_scores, model_scores)[0, 1]
        # 计算排序一致性（Kendall's tau）
        tau, _ = kendalltau(heuristic_scores, model_scores)
        return {'correlation': correlation, 'kendall_tau': tau}