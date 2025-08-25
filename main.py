from data import load_movielens
from model import NCF
from coreset_methods import random_coreset, clustering_coreset, gradient_coreset
from train import train_model
from evaluate import evaluate, CoresetEvaluator
from torch.utils.data import DataLoader
import numpy as np

# Parameters
BATCH_SIZE = 256
K = 10

# Load data
train_ds, val_ds, test_ds, num_users, num_items, user_interacted = load_movielens()
evaluator = CoresetEvaluator()

# Full dataset baseline
model = NCF(num_users, num_items)
full_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
model, train_time = train_model(model, full_loader, num_items)
recall, ndcg = evaluate(model, test_ds, num_items, user_interacted, K)
total_time = train_time  # No selection time for full
print(f'Full Dataset - Recall@{K}: {recall}, NDCG@{K}: {ndcg}, Train Time: {train_time}s, Total Time: {total_time}s')

# Evaluate full as 'coreset' for baseline (use train_ds as coreset for full)
q1_metrics = evaluator.evaluate_q1_value_metrics(train_ds, train_ds, model)
q2_metrics = evaluator.evaluate_q2_representativeness(train_ds, train_ds)
q3_metrics = evaluator.evaluate_q3_importance(train_ds, train_ds)
q4_metrics = evaluator.evaluate_q4_diversity(train_ds, model)
# For Q5, example: heuristic = rarity per sample, model = random grads (replace with actual)
heuristic_scores = np.random.rand(len(train_ds))  # Dummy; compute actual rarity per sample
model_scores = np.random.rand(len(train_ds))  # Dummy; compute grad per sample
q5_metrics = evaluator.evaluate_q5_estimation_quality(heuristic_scores, model_scores)
print(f'Full Q1: {q1_metrics}, Q2: {q2_metrics}, Q3: {q3_metrics}, Q4: {q4_metrics}, Q5: {q5_metrics}')

# Example: Random coreset
coreset, select_time = random_coreset(train_ds)
model = NCF(num_users, num_items)  # Reset model
coreset_loader = DataLoader(coreset, batch_size=BATCH_SIZE, shuffle=True)
model, train_time = train_model(model, coreset_loader, num_items)
recall, ndcg = evaluate(model, test_ds, num_items, user_interacted, K)
total_time = select_time + train_time
print(f'Random Coreset - Recall@{K}: {recall}, NDCG@{K}: {ndcg}, Select Time: {select_time}s, Train Time: {train_time}s, Total Time: {total_time}s')

# Evaluate random coreset
q1_metrics = evaluator.evaluate_q1_value_metrics(coreset, train_ds, model)
q2_metrics = evaluator.evaluate_q2_representativeness(coreset, train_ds)
q3_metrics = evaluator.evaluate_q3_importance(coreset, train_ds)
q4_metrics = evaluator.evaluate_q4_diversity(coreset, model)
# For Q5, similar dummy
heuristic_scores = np.random.rand(len(coreset))
model_scores = np.random.rand(len(coreset))
q5_metrics = evaluator.evaluate_q5_estimation_quality(heuristic_scores, model_scores)
print(f'Random Q1: {q1_metrics}, Q2: {q2_metrics}, Q3: {q3_metrics}, Q4: {q4_metrics}, Q5: {q5_metrics}')

# TODO: Add similar for clustering/gradient