import numpy as np
import torch
from scipy.spatial.distance import cdist
import os.path as osp


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=False):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()
        indices = []
        for i in range(label.max() + 1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = int(train_prop / (label.max() + 1) * len(label))
        val_lb = int(valid_prop * len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    return split_idx


def ndarry_to_torch_sparse_tensor(mx):
    row, col = mx.nonzero()
    indices = torch.from_numpy(np.vstack((row, col))).long()
    values = torch.ones(row.shape[0])
    shape = torch.Size(mx.shape)

    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.int)


def ExtractV2E(data):
    # Assume edge_index = [V|E;E|V]
    edge_index = data.edge_index
    #     First, ensure the sorting is correct (increasing along edge_index[0])
    _, sorted_idx = torch.sort(edge_index[0])
    edge_index = edge_index[:, sorted_idx].type(torch.LongTensor)

    num_nodes = data.n_x
    num_hyperedges = data.num_hyperedges
    if not ((num_nodes + num_hyperedges - 1) == data.edge_index[0].max().item()):
        print('num_hyperedges does not match! 1')
        return
    cidx = torch.where(edge_index[0] == num_nodes)[
        0].min()  # cidx: [V...|cidx E...]
    data.edge_index = edge_index[:, :cidx].type(torch.LongTensor)
    return data


def ConstructH(data, override=True):
    """
    Construct incidence matrix H of size (num_nodes,num_hyperedges) from edge_index = [V;E]
    """
    #     ipdb.set_trace()
    edge_index = np.array(data.edge_index)
    # Don't use edge_index[0].max()+1, as some nodes maybe isolated
    num_nodes = data.x.shape[0]
    num_hyperedges = np.max(edge_index[1]) - np.min(edge_index[1]) + 1
    H = np.zeros((num_nodes, num_hyperedges))
    cur_idx = 0
    for he in np.unique(edge_index[1]):
        nodes_in_he = edge_index[0][edge_index[1] == he]
        H[nodes_in_he, cur_idx] = 1.
        cur_idx += 1

    # row, col = H.nonzero()
    #
    # indices = torch.from_numpy(np.vstack((row, col))).long()
    # values = torch.ones(row.shape[0])
    # shape = torch.Size((num_nodes, num_nodes))
    #
    # data.H = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.int)
    if override:
        data.edge_index = H
    else:
        data.H = H
    return data


def GenerateDist(data, file_path, TYPE="euclidean"):
    X = data.x.numpy()
    distances = cdist(X, X, TYPE)

    if TYPE == "cosine":
        distances = (distances + 1) / 2
    distances += 1e-5
    # distances = np.ones((X.shape[0], X.shape[0]))
    # sorted_indices = np.argsort(distances, axis=1)
    # kth_smallest_values = distances[np.arange(X.shape[0]), sorted_indices[:, K - 1]]
    # kth_smallest_values += 1e-5
    data.dists = distances

    return data


def preprocess_data(args, data):
    print(f"Dataset: {args.dname}")
    if args.method in ["AdE"]:
        data = ExtractV2E(data)
        data = ConstructH(data)
        data.edge_index = ndarry_to_torch_sparse_tensor(data.edge_index)
        file_path = osp.join(args.root_dir, args.preprocessed_dir, f"Dist_{args.dname}_{args.Distance_Method}.txt")
        data = GenerateDist(data, file_path, TYPE=args.Distance_Method)
    else:
        raise ValueError("Unrecognized model name")
    return data
