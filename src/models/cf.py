import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, mean_average_precision_at_k, ndcg_at_k
from scipy.sparse import csr_matrix
from tqdm import tqdm
from itertools import product


class RecommendationSystem:
    def __init__(self, factors=1000, regularization=0.01, alpha=1.0, iterations=50):
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
        )
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.item_user_data = None
        self.item_factors = None

    def fit(self, df):
        print("Encoding users and items...")
        users = self.user_encoder.fit_transform(df["user_id"])
        items = self.item_encoder.fit_transform(df["parent_asin"])
        ratings = df["rating"].values

        print("Creating item-user matrix...")
        self.item_user_data = csr_matrix((ratings, (items, users)))

        print("Fitting the model...")
        self.model.fit(self.item_user_data)
        self.item_factors = self.model.item_factors

    def recommend(self, user_id, n=10):
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            recommendations = self.model.recommend(user_idx, self.item_user_data, N=n)
        except ValueError:
            # For new users, recommend most popular items
            item_popularity = np.asarray(self.item_user_data.sum(axis=1)).flatten()
            recommendations = sorted(enumerate(item_popularity), key=lambda x: -x[1])[
                :n
            ]

        return [
            self.item_encoder.inverse_transform([item])[0]
            for item, _ in recommendations
        ]

    def evaluate(self, test_df, k=10):
        # Create a user-item interaction matrix for the test set
        test_user_item = csr_matrix(
            (
                test_df["rating"].values,
                (
                    self.item_encoder.transform(test_df["parent_asin"]),
                    self.user_encoder.transform(test_df["user_id"]),
                ),
            )
        )

        # Calculate metrics
        precision = precision_at_k(
            self.model, self.item_user_data, test_user_item, k, show_progress=True
        )
        map_score = mean_average_precision_at_k(
            self.model, self.item_user_data, test_user_item, k, show_progress=True
        )
        ndcg = ndcg_at_k(
            self.model, self.item_user_data, test_user_item, k, show_progress=True
        )

        # round the metrics to 4 decimal places
        precision = round(precision, 4)
        map_score = round(map_score, 4)
        ndcg = round(ndcg, 4)
        return {f"precision@{k}": precision, f"MAP@{k}": map_score, f"NDCG@{k}": ndcg}


def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Preprocessing data...")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values(["user_id", "timestamp"])
    return df


def train_model(train_df, factors=100, regularization=0.01, alpha=1.0):
    print("Training the model...")
    model = RecommendationSystem(
        factors=factors, regularization=regularization, alpha=alpha
    )
    model.fit(train_df)
    return model


def filter_seen_items(train_df, test_df):
    seen_items = set(train_df["parent_asin"])
    return test_df[test_df["parent_asin"].isin(seen_items)]


def main():
    # Load and preprocess data
    train_file = "/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.train.csv"
    valid_file = "/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.valid.csv"
    test_file = "/home/kchauhan/repos/mds-tmu-mrp/datasets/Video_Games.test.csv"

    train_df = load_and_preprocess_data(train_file)
    valid_df = load_and_preprocess_data(valid_file)
    test_df = load_and_preprocess_data(test_file)

    # Define hyperparameters grid
    hyperparameters_grid = {
        "factors": [10, 50, 100, 1000, 2000],
        "regularization": [0.01],
        "alpha": [1.0],
    }

    best_validation_score = float("-inf")
    best_hyperparameters = {}
    best_model = None
    k = 10

    # Perform grid search
    for combination in product(*hyperparameters_grid.values()):
        hyperparameters = dict(zip(hyperparameters_grid.keys(), combination))
        print(f"Training with hyperparameters: {hyperparameters}")

        model = train_model(train_df, **hyperparameters)

        valid_df_filtered = filter_seen_items(train_df, valid_df)
        validation_metrics = model.evaluate(valid_df_filtered, k=k)

        if validation_metrics[f"precision@{k}"] > best_validation_score:
            best_validation_score = validation_metrics[f"precision@{k}"]
            best_hyperparameters = hyperparameters
            best_model = model

    # Evaluate the best model on the test set
    test_df_filtered = filter_seen_items(train_df, test_df)
    test_metrics = best_model.evaluate(test_df_filtered)

    print(f"Best Hyperparameters: {best_hyperparameters}")
    print("\nValidation Metrics of Best Model:")
    print(
        validation_metrics
    )  # This should be the metrics of the best model on the validation set
    print("\nTest Metrics of Best Model:")
    print(test_metrics)


if __name__ == "__main__":
    main()
