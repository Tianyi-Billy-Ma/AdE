import random, os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import yaml
import os.path as osp

def load_yaml(file_dir):
    file_path = osp.join(file_dir, "config.yaml")
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def set_parameters(args):
    hyperparameters = load_yaml(args.root_dir)
    if args.dname in hyperparameters.keys():
        parameters = hyperparameters[args.dname]
        args.lr = parameters["lr"]
        args.wd = parameters["wd"]
        args.dropout = parameters["dropout"]
        args.hidden_dim = parameters["hidden_dim"]
        args.MLP_hidden = parameters["MLP_hidden"]
    else:
        raise ValueError("The dataset does not have stored parameters")
    return args

def cosine_similarity(X, rerange=True):
    X = F.normalize(X, dim=1)
    sim = torch.mm(X, X.t())
    return (sim + 1) / 2 if rerange else sim


def euclidean_distance(X):
    #     differences = X.unsqueeze(0) - X.unsqueeze(1)
    #     return torch.sqrt(differences.pow(2).sum(-1))
    sum_X =  torch.sum(X * X, dim=1)
    dist = sum_X.unsqueeze(1) + sum_X.unsqueeze(0) - 2 * X @ X.T
    # Ensure numerical stability and take the square root
    dist = torch.sqrt(torch.clamp(dist, min=0))
    return dist

def fix_seed(seed=37):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def evaluate(y_pred, y_true, split_idx, eval_func):
    train_acc = eval_func(
        y_true[split_idx['train']], y_pred[split_idx['train']])
    valid_acc = eval_func(
        y_true[split_idx['valid']], y_pred[split_idx['valid']])
    test_acc = eval_func(
        y_true[split_idx['test']], y_pred[split_idx['test']])

    #     Also keep track of losses
    train_loss = F.nll_loss(
        y_pred[split_idx['train']], y_true[split_idx['train']])
    valid_loss = F.nll_loss(
        y_pred[split_idx['valid']], y_true[split_idx['valid']])
    test_loss = F.nll_loss(
        y_pred[split_idx['test']], y_true[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, y_pred


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            print(f'\nRun {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[result[:, 1].argmax().item(), 0]:.2f}')
            print(f'   Final Test: {result[result[:, 2].argmax().item(), 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            best_epoch = []
            for r in result:
                index = np.argmax(r[:, 1]).item()
                best_epoch.append(index)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 2].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            print("best epoch:", best_epoch)
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

    def plot_result(self, run=None):
        plt.style.use('seaborn')
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            x = torch.arange(result.shape[0])
            plt.figure()
            print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
        else:
            result = 100 * torch.tensor(self.results[0])
            x = torch.arange(result.shape[0])
            plt.figure()
            #             print(f'Run {run + 1:02d}:')
            plt.plot(x, result[:, 0], x, result[:, 1], x, result[:, 2])
            plt.legend(['Train', 'Valid', 'Test'])
