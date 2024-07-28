import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_at_k_1(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    missing_users_indices = []

    for idx, u in enumerate(user_ids):
        if u not in train_user_dict:
            missing_users_indices.append(idx)
            continue

        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    if missing_users_indices:
        print(f"Number of missing users in train_user_dict: {len(missing_users_indices)}")

    try:
        _, rank_indices = torch.sort(torch.tensor(cf_scores).cuda(), descending=True)
    except:
        _, rank_indices = torch.sort(torch.tensor(cf_scores), descending=True)
    rank_indices = rank_indices.cpu().numpy()

    binary_hit = test_pos_item_binary[np.arange(len(user_ids))[:, None], rank_indices]
    
    # Remove missing users
    if missing_users_indices:
        binary_hit = np.delete(binary_hit, missing_users_indices, axis=0)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall']    = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg']      = ndcg_at_k_batch(binary_hit, k)
    
    return metrics_dict

def calc_metrics_at_k(top_k_items, train_user_dict, test_user_dict, user_ids, Ks):
    max_k = max(Ks)
    result = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}
    
    for i, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict.get(u, [])
        test_pos_item_list = test_user_dict.get(u, [])
        
        # Remove train items from top-k items
        top_k = top_k_items[i]
        top_k = [item for item in top_k if item not in train_pos_item_list][:max_k]
        
        hits = np.isin(top_k, test_pos_item_list)
        
        for k in Ks:
            hit_k = hits[:k]
            num_hits_k = np.sum(hit_k)
            
            precision_k = num_hits_k / k
            recall_k = num_hits_k / len(test_pos_item_list) if test_pos_item_list else 0
            ndcg_k = ndcg_at_k(hit_k, k)
            
            result[k]['precision'].append(precision_k)
            result[k]['recall'].append(recall_k)
            result[k]['ndcg'].append(ndcg_k)
    
    return result

def ndcg_at_k(hit, k):
    dcg = np.sum(hit / np.log2(np.arange(2, len(hit) + 2)))
    idcg = np.sum(1 / np.log2(np.arange(2, min(k, np.sum(hit)) + 2)))
    return dcg / idcg if idcg > 0 else 0