import numpy as np
from numba import njit


@njit
def _predict(x_id, y_id, x_rated_y, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 2)) # sim and rating
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in x_rated_y:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (sim, r) in k_neighbors:
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * r
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_mean(x_id, y_id, x_rated_y, mu, S, k, k_min):
    """Predict rating of user x for item y (if iiCF).
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        mu (ndarray): List of mean ratings of all user (if iiCF, or all item if uuCF).
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3)) # user/item ids for mean (mu)
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in x_rated_y:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((x2, sim, rating))

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            sum_ratings += sim * (r - mu[nb])
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est = sum_ratings / sum_sim

    return est


@njit
def _predict_baseline(x_id, y_id, x_rated_y, S, k, k_min, global_mean, bx, by):
    """Predict rating of user x for item y (if iiCF) using baseline estimate.
    Args:
        x_id (int): users Id    (if iiCF)
        y_id (int): items Id    (if iiCF)
        x_rated_y (ndarray): List where element `i` is ndarray of `(xs, rating)` where `xs` is all x that rated y and the ratings.
        X (ndarray): the training set with size (|TRAINSET|, 3)
        S (ndarray): similarity matrix
        k (int): number of k-nearest neighbors
        k_min (int): number of minimum k
        global_mean (float): mean ratings in training set
        bx (ndarray): user biases   (if iiCF)
        by (ndarray): item biases   (if iiCF)
    Returns:
        pred (float): predicted rating of user x for item y     (if iiCF)
    """

    k_neighbors = np.zeros((k, 3))
    k_neighbors[:, 1] = -1          # All similarity degree is default to -1

    for x2, rating in x_rated_y:
        if int(x2) == x_id:
            continue       # Bo qua item dang xet
        sim = S[int(x2), x_id]
        argmin = np.argmin(k_neighbors[:, 1])
        if sim > k_neighbors[argmin, 1]:
            k_neighbors[argmin] = np.array((x2, sim, rating))

    est = global_mean + bx[x_id] + by[y_id]

    # Compute weighted average
    sum_sim = sum_ratings = actual_k = 0
    for (nb, sim, r) in k_neighbors:
        nb = int(nb)
        if sim > 0:
            sum_sim += sim
            nb_bsl = global_mean + bx[nb] + by[y_id]
            sum_ratings += sim * (r - nb_bsl)
            actual_k += 1

    if actual_k < k_min:
        sum_ratings = 0

    if sum_sim:
        est += sum_ratings / sum_sim

    return est

@njit
def _calculate_precision_recall(user_ratings, k, threshold, top_k_ranking_metric):
    """Calculate the precision and recall at k metric for the user based on his/her obversed rating and his/her predicted rating.

    Args:
        user_ratings (ndarray): An array contains the predicted rating in the first column and the obversed rating in the second column.
        k (int): the k metric.
        threshold (float): relevant threshold.
        top_k_ranking_metric (bool): use k (instead of n_rec_k) for precision.
    Returns:
        (precision, recall): the precision and recall score for the user.
    """
    n_ratings = len(user_ratings)
    if n_ratings == 0:
        return 0.0, 0.0
        
    pred_ratings = np.zeros(n_ratings)
    true_ratings = np.zeros(n_ratings)
    
    for i in range(n_ratings):
        pred_ratings[i] = user_ratings[i][0]
        true_ratings[i] = user_ratings[i][1]
    
    # Sort by predicted ratings
    sort_idx = np.argsort(pred_ratings)[::-1]
    pred_ratings = pred_ratings[sort_idx]
    true_ratings = true_ratings[sort_idx]

    # Number of relevant items
    n_rel = np.sum(true_ratings[:min(k, n_ratings)] >= threshold)
    n_rel_top_k = np.sum(true_ratings >= threshold) # for top k ranking, this is n_rel on whole test set

    # Number of recommended items
    n_rec_k = np.sum(pred_ratings[:min(k, n_ratings)] >= threshold)

    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = 0
    if top_k_ranking_metric:
        n_rel_and_rec_k = n_rel # = true_ratings >= threshold in first k rec by predicted ratings
    else:
        n_rel_and_rec_k = np.sum((true_ratings[:min(k, n_ratings)] >= threshold) & \
                         (pred_ratings[:min(k, n_ratings)] >= threshold))

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.
    if n_rec_k != 0:
        if top_k_ranking_metric:
            precision = n_rel_and_rec_k / k
        else:
            precision = n_rel_and_rec_k / n_rec_k
    else:
        precision = 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.
    if n_rel != 0:
        if top_k_ranking_metric:
            recall = n_rel_and_rec_k / n_rel_top_k
        else:
            recall = n_rel_and_rec_k / n_rel
    else:
        recall = 0

    return precision, recall

@njit
def _calculate_ndcg(user_true_ratings, user_est_ratings, k):
    """Calculate the NDCG at k metric for the user based on his/her obversed rating and his/her predicted rating.

    Args:
        user_true_ratings (ndarray): An array contains the predicted rating on the test set.
        user_est_ratings (ndarray): An array contains the obversed rating on the test set.
        k (int): the k metric.

    Returns:
        ndcg: the ndcg score for the user.
    """
    n_ratings = len(user_true_ratings)
    if n_ratings == 0:
        return 0.0
        
    # Sort ratings by true and predicted values
    true_order = np.argsort(user_true_ratings)[::-1][:min(k, n_ratings)]
    est_order = np.argsort(user_est_ratings)[::-1][:min(k, n_ratings)]

    # Calculate DCG for both true and predicted ratings
    true_dcg = dcg(user_true_ratings, true_order)
    est_dcg = dcg(user_est_ratings, est_order)

    # Avoid division by zero
    if true_dcg == 0:
        return 0.0
        
    ndcg = est_dcg / true_dcg
    return ndcg

@njit
def dcg(ratings, order, top_k_ranking_metric = False):
    """ Calculate discounted cumulative gain.

    Args:
        ratings (ndarray): the rating of the user on the test set.
        order (ndarray): list of item id, sorted by the rating.

    Returns:
        float: the discounted cumulative gain of the user.
    """
    dcg_score = 0.0
    for ith, item in enumerate(order):
        dcg_score += (np.power(2, ratings[item]) - 1) / np.log2(ith + 2)
    return dcg_score
