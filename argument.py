import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=os.getcwd())
    parser.add_argument('--preprocessed_dir', default="data/preprocessed/", type=str)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--train_prop', type=float, default=0.4)
    parser.add_argument('--valid_prop', type=float, default=0.3)
    parser.add_argument('--dname', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--cuda', default=0, choices=[-1, 0, 1], type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=0.01, type=float)

    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    parser.add_argument("--MLP_num_layers", default=2, type=int)
    parser.add_argument('--MLP_hidden', default=32, type=int)

    parser.add_argument('--feature_noise', default='1', type=str)
    parser.add_argument('--num_features', default=0, type=int)  # Placeholder
    parser.add_argument('--num_classes', default=0, type=int)  # Placeholder

    parser.add_argument('--Distance_Method', default="euclidean", type=str,
                        help="Options are euclidean, minkowski, cityblock, cosine, correlation, hamming, jaccard")
    parser.add_argument('--threshold', default=0.0, type=float)
    parser.add_argument('--backbone', default='GAT', type=str,
                        help="The backbone GNN model")
    parser.add_argument("--aggregate", default="mean", choices=["mean", "max", "sum"])


    parser.add_argument('--display_step', type=int, default=-1)

    parser.set_defaults(dname="citeseer")
    parser.set_defaults(method="AdE")
    parser.set_defaults(threshold=0.0)
    parser.set_defaults(backbone="GCN")

    args = parser.parse_args()
    return args