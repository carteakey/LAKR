import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k

class RecommendationSystem:
    def __init__(self, factors=100, regularization=0.01, alpha=1.0, iterations=50):
        self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, 
                                             alpha=alpha, iterations=iterations)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.item_user_data = None
        self.item_factors = None
        
    def fit(self, df):
        users = self.user_encoder.fit_transform(df['user_id'])
        items = self.item_encoder.fit_transform(df['parent_asin'])
        ratings = df['rating'].values
        
        self.item_user_data = csr_matrix((ratings, (items, users)))
        self.model.fit(self.item_user_data)
        self.item_factors = self.model.item_factors
    
    def recommend(self, user_id, n=10):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            recommendations = self.model.recommend(user_idx, self.item_user_data, N=n)
        except ValueError:
            # For new users, recommend most popular items
            item_popularity = np.asarray(self.item_user_data.sum(axis=1)).flatten()
            recommendations = sorted(enumerate(item_popularity), key=lambda x: -x[1])[:n]
        
        return [self.item_encoder.inverse_transform([item])[0] for item, _ in recommendations]

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values(['user_id', 'timestamp'])
    return df

def train_model(train_df):
    model = RecommendationSystem()
    model.fit(train_df)
    return model

def evaluate_model(model, test_df, k=10):
    true_items = test_df['parent_asin'].values
    user_ids = test_df['user_id'].values
    
    predictions = []
    for user_id in user_ids:
        predictions.append(model.recommend(user_id, n=k))
    
    precision = precision_at_k(predictions, true_items, k=k)
    map_score = mean_average_precision_at_k(predictions, true_items, k=k)
    ndcg = ndcg_at_k(predictions, true_items, k=k)
    
    return {
        f'precision@{k}': precision,
        f'MAP@{k}': map_score,
        f'NDCG@{k}': ndcg
    }

def main():
    # Load and preprocess data
    train_file = '/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.train.csv'
    valid_file = '/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.valid.csv'
    test_file = '/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.test.csv'
    
    train_df = load_and_preprocess_data(train_file)
    valid_df = load_and_preprocess_data(valid_file)
    test_df = load_and_preprocess_data(test_file)
    
    # Train the model
    model = train_model(train_df)
    
    # Evaluate the model
    validation_metrics = evaluate_model(model, valid_df)
    test_metrics = evaluate_model(model, test_df)
    
    print("Validation Metrics:")
    print(validation_metrics)
    print("\nTest Metrics:")
    print(test_metrics)

if __name__ == "__main__":
    main()