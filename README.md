# Coreset Selection for Efficient Recommendation

## Project Overview
This mini-project investigates coreset selection techniques for efficient training of Neural Collaborative Filtering (NCF) models on recommender systems. Based on the proposal "Coreset Selection for Efficient Recommendation," it implements baseline NCF on MovieLens-1M, with coreset strategies (random, clustering, gradient), evaluation metrics (Recall@10, NDCG@10), and efficiency measurements. Additional evaluators analyze coreset quality per solution design questions.

Key features:
- Follows NCF paper protocols: implicit feedback, leave-one-out evaluation, negative sampling.
- Coreset strategies: Random, Clustering (user activity), Gradient-based.
- Evaluators: For value, representativeness, importance, diversity, estimation quality.
- Modular structure for easy extension.

## Directory Structure
- `data.py`: Data loading and MovieLensDataset class.
- `model.py`: NCF (NeuMF) model definition.
- `coreset_methods.py`: Coreset selection functions (random, clustering, gradient).
- `train.py`: Model training with negative sampling.
- `evaluate.py`: Recommendation evaluation (Recall/NDCG) and CoresetEvaluator class.
- `main.py`: Main script to run experiments and evaluations.
- `ml-1m/ratings.dat`: Dataset (download from MovieLens if missing).

## Requirements
- Python 3.10+
- Libraries: torch, pandas, numpy, scikit-learn, scipy
- Install: `pip install torch pandas numpy scikit-learn scipy`

## Usage
1. Place `ratings.dat` in `ml-1m/` directory.
2. Run: `python main.py`
   - Outputs: Performance (Recall/NDCG), times, and evaluator metrics for full and random coreset.
   - Extend: Add calls for clustering/gradient in main.py.
3. Customize parameters in main.py (e.g., EPOCHS, CORESET_RATIO).

## Notes
- Full dataset training may take ~15-20 mins on CPU.
- Evaluators use sampling for large data to avoid memory issues.
- For new coreset methods: Add to coreset_methods.py and call in main.py.

## License
MIT License.
