import random
import numpy as np
import torch


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get("indices", False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError("Inputs to shuffles must have the same length")

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices

    return result


def mini_batch(*arrays, **kwargs):
    batch_size = kwargs.get("batch_size", 2048)

    if len(arrays) == 1:
        for i in range(0, len(arrays[0]), batch_size):
            yield arrays[0][i:i + batch_size]
    else:
        for i in range(0, len(arrays[0]), batch_size):
            yield tuple(array[i:i + batch_size] for array in arrays)


def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)

    row = torch.LongTensor(coo.row)
    col = torch.LongTensor(coo.col)
    index = torch.stack([row, col], dim=0)
    value = torch.FloatTensor(coo.data)

    sp_tensor = torch.sparse_coo_tensor(
        index,
        value,
        torch.Size(coo.shape)
    )

    return sp_tensor.coalesce()
