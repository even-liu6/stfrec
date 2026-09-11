import numpy as np
import torch
import utility.tools as tools


def testing(model, args, dataset, device):
    model = model.eval()
    topK = eval(args.top_K)
    model_results = {'recall': np.zeros(len(topK)), 'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        test_users = list(dataset.test_dict.keys())
        user_list, true_list, rating_list = [], [], []
        num_batch = len(test_users) // int(args.test_batch_size) + 1

        for batch_users in tools.mini_batch(test_users, batch_size=int(args.test_batch_size)):
            exclude_users, exclude_items = [], []
            test_batch_pos = [dataset.test_dict[u] for u in batch_users]

            for i, u in enumerate(batch_users):
                if u in dataset.train_dict:
                    exclude_users.extend([i] * len(dataset.train_dict[u]))
                    exclude_items.extend(dataset.train_dict[u])

            batch_users_device = torch.Tensor(batch_users).long().to(device)
            rating = model.get_rating_for_test(batch_users_device)
            rating[exclude_users, exclude_items] = -1
            _, rating_k = torch.topk(rating, k=max(topK))
            rating = rating.cpu()
            del rating

            user_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            true_list.append(test_batch_pos)

        assert num_batch == len(user_list)

        results = []
        for single_list in zip(rating_list, true_list):
            results.append(test_single_batch(single_list, topK))

        for idx, result in enumerate(results, start=1):
            model_results['recall'] += result['recall']
            model_results['ndcg'] += result['ndcg']
            print('\t Steps %d/%d: recall = %.3f, ndcg = %.3f' %
                  (idx, num_batch, float(result['recall'][1]), float(result['ndcg'][1])), end='\r')

        model_results['recall'] /= float(len(test_users))
        model_results['ndcg'] /= float(len(test_users))

        if int(getattr(args, 'pop_eval', 0)) == 1 and hasattr(model, 'item_group'):
            model_results['pop'] = popularity_metrics(
                rating_list=rating_list,
                true_list=true_list,
                topK=topK,
                item_group=model.item_group.detach().cpu().numpy(),
                item_degree=model.item_degree.detach().cpu().numpy(),
                num_items=dataset.num_items,
            )

        return model_results


def popularity_metrics(rating_list, true_list, topK, item_group, item_degree, num_items):
    """Popularity-bias metrics at @20 by default.

    These metrics do not affect training. They let you verify whether the new
    loss actually changes long-tail behavior rather than only Recall/NDCG.
    """
    if 20 in topK:
        k = 20
    else:
        k = topK[1] if len(topK) > 1 else topK[0]

    all_pred = []
    group_hits = np.zeros(3, dtype=np.float64)
    group_truth = np.zeros(3, dtype=np.float64)
    group_ndcg_sum = np.zeros(3, dtype=np.float64)
    group_user_count = np.zeros(3, dtype=np.float64)

    discount = 1.0 / np.log2(np.arange(2, k + 2))

    for pred_batch, truth_batch in zip(rating_list, true_list):
        pred_items = pred_batch.numpy()[:, :k]
        all_pred.append(pred_items.reshape(-1))
        for row_idx, true_items in enumerate(truth_batch):
            pred = pred_items[row_idx]
            pred_set = set(pred.tolist())
            true_items = list(true_items)
            for g in (0, 1, 2):
                true_g = [it for it in true_items if item_group[it] == g]
                if len(true_g) == 0:
                    continue
                true_g_set = set(true_g)
                hits = sum(1 for it in pred if it in true_g_set)
                group_hits[g] += hits
                group_truth[g] += len(true_g)

                labels = np.array([1.0 if it in true_g_set else 0.0 for it in pred], dtype=np.float64)
                dcg = np.sum(labels * discount)
                ideal_len = min(k, len(true_g))
                idcg = np.sum(discount[:ideal_len]) if ideal_len > 0 else 1.0
                group_ndcg_sum[g] += dcg / max(idcg, 1e-12)
                group_user_count[g] += 1.0

    all_pred = np.concatenate(all_pred) if len(all_pred) > 0 else np.array([], dtype=np.int64)
    coverage = len(np.unique(all_pred)) / float(num_items) if num_items > 0 else 0.0
    avg_pop = float(np.mean(item_degree[all_pred])) if all_pred.size > 0 else 0.0
    avg_log_pop = float(np.mean(np.log1p(item_degree[all_pred]))) if all_pred.size > 0 else 0.0

    group_recall = group_hits / np.maximum(group_truth, 1.0)
    group_ndcg = group_ndcg_sum / np.maximum(group_user_count, 1.0)

    return {
        'k': k,
        'coverage': coverage,
        'avg_pop': avg_pop,
        'avg_log_pop': avg_log_pop,
        'tail_recall': float(group_recall[0]),
        'middle_recall': float(group_recall[1]),
        'head_recall': float(group_recall[2]),
        'tail_ndcg': float(group_ndcg[0]),
        'middle_ndcg': float(group_ndcg[1]),
        'head_ndcg': float(group_ndcg[2]),
    }


def testing_group(model, args, dataset, device, my_dicts):
    model = model.eval()
    topK = eval(args.top_K)
    model_results = {'recall': np.zeros(len(topK)), 'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        test_users = list(my_dicts.keys())
        user_list, true_list, rating_list = [], [], []
        num_batch = len(test_users) // int(args.test_batch_size) + 1

        for batch_users in tools.mini_batch(test_users, batch_size=int(args.test_batch_size)):
            exclude_users, exclude_items = [], []
            test_batch_pos = [my_dicts[u] for u in batch_users]

            for i, u in enumerate(batch_users):
                if u in dataset.train_dict:
                    exclude_users.extend([i] * len(dataset.train_dict[u]))
                    exclude_items.extend(dataset.train_dict[u])
            batch_users_device = torch.Tensor(batch_users).long().to(device)
            rating = model.get_rating_for_test(batch_users_device)
            rating[exclude_users, exclude_items] = -1
            _, rating_k = torch.topk(rating, k=max(topK))
            rating = rating.cpu()
            del rating

            user_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            true_list.append(test_batch_pos)

        assert num_batch == len(user_list)
        results = []
        for single_list in zip(rating_list, true_list):
            results.append(test_single_batch(single_list, topK))

        for result in results:
            model_results['recall'] += result['recall']
            model_results['ndcg'] += result['ndcg']
        model_results['recall'] /= float(len(test_users))
        model_results['ndcg'] /= float(len(test_users))
        return model_results


def test_single_batch(single_list, topK):
    pred_items = single_list[0].numpy()
    true_items = single_list[1]
    pred_item_label = pred_to_label(pred_items, true_items)

    recall, ndcg = [], []
    for k_size in topK:
        recall.append(recall_k(pred_item_label, k_size, true_items))
        ndcg.append(ndcg_k(pred_item_label, k_size, true_items))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg)}


def pred_to_label(pred_items, true_items):
    pred_item_label = []
    for i in range(len(true_items)):
        true_item = true_items[i]
        pred_item = pred_items[i]
        pred = list(map(lambda x: x in true_item, pred_item))
        pred = np.array(pred).astype("float")
        pred_item_label.append(pred)
    return np.array(pred_item_label).astype("float")


def recall_k(pred, k_size, true):
    pred_k = pred[:, : k_size].sum(1)
    recall_num = np.array([len(true[i]) for i in range(len(true))])
    recall = np.sum(pred_k / recall_num)
    return recall


def ndcg_k(pred, k_size, true):
    assert len(pred) == len(true)
    pred_matrix = pred[:, :k_size]
    true_matrix = np.zeros((len(pred_matrix), k_size))
    for i, items in enumerate(true):
        length = k_size if k_size <= len(items) else len(items)
        true_matrix[i, :length] = 1

    dcg = np.sum(pred_matrix * (1. / np.log2(np.arange(2, k_size + 2))), axis=1)
    idcg = np.sum(true_matrix * (1. / np.log2(np.arange(2, k_size + 2))), axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def sparsity_test(dataset, args, model, device):
    sparsity_results = []
    model = model.eval()
    topK = eval(args.top_K)

    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {'recall': np.zeros(len(topK)), 'ndcg': np.zeros(len(topK))}
            users_list, rating_list, ground_true_list = [], [], []
            num_batch = len(users) // int(args.test_batch_size) + 1

            for batch_users in tools.mini_batch(users, batch_size=int(args.test_batch_size)):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items(batch_users)
                ground_true = [dataset.test_dict[u] for u in batch_users]
                batch_users_device = torch.Tensor(batch_users).long().to(device)
                rating = model.get_rating_for_test(batch_users_device)

                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -1
                _, rating_k = torch.topk(rating, k=max(topK))
                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            results = []
            for single_list in zip(rating_list, ground_true_list):
                results.append(test_single_batch(single_list, topK))

            for result in results:
                model_results['recall'] += result['recall']
                model_results['ndcg'] += result['ndcg']

            model_results['recall'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))
            sparsity_results.append(model_results)

    return sparsity_results
