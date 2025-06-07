import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================Calculate similarity matrix==========================================================
def pairwise_pearson_matrix(users, movies):
    # users: (n_users, dim), movies: (n_movies, dim)
    users = users.to_numpy() if isinstance(users, pd.DataFrame) else users
    movies = movies.to_numpy() if isinstance(movies, pd.DataFrame) else movies
    
    users_centered = users - users.mean(axis=1, keepdims=True)
    movies_centered = movies - movies.mean(axis=1, keepdims=True)

    users_norm = np.linalg.norm(users_centered, axis=1, keepdims=True)  # (n_users, 1)
    movies_norm = np.linalg.norm(movies_centered, axis=1, keepdims=True)  # (n_movies, 1)

    users_normalized = users_centered / (users_norm + 1e-8)
    movies_normalized = movies_centered / (movies_norm + 1e-8)

    return users_normalized @ movies_normalized.T 

def calculate_similarity_matrix(user_embeddings, movie_embeddings, method = 'cosine'):
    if method == 'cosine':
        user_movie_sim = cosine_similarity(user_embeddings, movie_embeddings)
    elif method == 'pearson':
        user_movie_sim = pairwise_pearson_matrix(user_embeddings, movie_embeddings)
    else: 
        raise ValueError("Method must be 'cosine' or 'pearson' atm.")
    
    return user_movie_sim

# ==========================================================Calculate metrics from similarity matrix==========================================================
def sim_descriptive(df_user_movie):
    all_values = df_user_movie.values.flatten()

    # Remove NaNs if any
    all_values = all_values[~np.isnan(all_values)]

    global_stats = {
        'count': len(all_values),
        'mean': np.mean(all_values),
        'std': np.std(all_values),
        'min': np.min(all_values),
        '25%': np.percentile(all_values, 25),
        '50%': np.percentile(all_values, 50),
        '75%': np.percentile(all_values, 75),
        'max': np.max(all_values)
    }

    display(pd.DataFrame(global_stats, index=['Overall']))

def calculate_precision_recall_from_similarity(user_sim, user_test_ratings, k, threshold=3.5):
    """
    Calculate precision and recall at k using similarity scores for a specific user.

    """
    n_ratings = len(user_test_ratings)

    if n_ratings == 0:
        return 0.0, 0.0
    
    # Create a set of relevant movies (true_ratings > threshold)
    relevant_movies = set(user_test_ratings[user_test_ratings.iloc[:, 2] >= threshold].iloc[:, 1].astype(int))

    n_rel = len(relevant_movies)
    if n_rel == 0:
        return 0.0, 0.0

    # Taking top similarity for top_k_recs, AS WE TAKE SUBSET OF USER_SIM FROM MOVIEID USER HAVE RATED IN TEST SET
    movieId_user_rated_test = [int(id) for id in user_test_ratings.iloc[:, 1]]
    user_sim = user_sim[user_sim.index.astype(int).isin(movieId_user_rated_test)]

    top_k_recs = set(int(x) for x in user_sim.index[:k])

    # Find intersection of relevant and recommended items
    n_rel_and_rec_k = len(list(top_k_recs & relevant_movies))

    precision = n_rel_and_rec_k / k
    recall = n_rel_and_rec_k / n_rel

    return precision, recall

def precision_recall_at_k(df_user_movie, rating_test, k, threshold=3.5):
    """
    Calculate precision and recall at k for all users in the test set.

    """
    userIds = rating_test['userId'].unique()
    userIds_mapping = {userId: idx for idx, userId in enumerate(userIds)}

    n_users = len(rating_test['userId'].unique())

    precisions = np.zeros(n_users)
    recalls = np.zeros(n_users)

    for userId in userIds:
        user_test_ratings = rating_test[rating_test['userId'] == userId]
        user_sim = df_user_movie.loc[userId].sort_values(ascending=False)

        precisions[userIds_mapping[userId]], recalls[userIds_mapping[userId]] = calculate_precision_recall_from_similarity(user_sim, user_test_ratings, k, threshold)

    precision = sum(prec for prec in precisions) / n_users if n_users > 0 else 0.0
    recall = sum(rec for rec in recalls) / n_users if n_users > 0 else 0.0

    print(f"Precision@{k}: {precision:.5f} - Recall@{k}: {recall:.5f}")

    return precision, recall

def calculate_ndcg_from_similarity(user_sim, user_test_ratings, k):
    """
    Calculate NDCG@K using similarity scores for a specific user.

    """
    if len(user_test_ratings) == 0:
        return 0.0

    # movie_ratings = {movie_id : rating}
    movie_ratings = dict(zip(user_test_ratings.iloc[:, 1].astype(int), user_test_ratings.iloc[:, 2]))

    # Similarly, we take only subset of user_sim from movieId user have rated in test set
    movieId_user_rated_test = [int(id) for id in user_test_ratings.iloc[:, 1]]
    user_sim = user_sim[user_sim.index.astype(int).isin(movieId_user_rated_test)]

    top_k_recs = set(int(x) for x in user_sim.index[:k])

    # DCG@K
    dcg = 0.0
    for i, movie_id in enumerate(top_k_recs):
        if movie_id in movie_ratings:
            # Using the actual rating as relevance score
            rel = movie_ratings[movie_id]

            dcg += rel / np.log2(i + 2)

    # Sort movies by their ratings for ideal ranking
    ideal_ranking = sorted(movie_ratings.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_ranking))):
        idcg += ideal_ranking[i] / np.log2(i + 2)

    # NDCG
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0

    return ndcg

def ndcg_at_k(df_user_movie, rating_test, k):
    """
    Calculate average NDCG@K for all users in the test set.

    """
    userIds = rating_test['userId'].unique()
    n_users = len(userIds)

    ndcg_sum = 0.0

    for userId in userIds:
        user_test_ratings = rating_test[rating_test['userId'] == userId]
        user_sim = df_user_movie.loc[userId].sort_values(ascending=False)

        ndcg_sum += calculate_ndcg_from_similarity(user_sim, user_test_ratings, userId, k)

    avg_ndcg = ndcg_sum / n_users if n_users > 0 else 0.0

    print(f"NDCG@{k}: {avg_ndcg:.5f}")

    return avg_ndcg