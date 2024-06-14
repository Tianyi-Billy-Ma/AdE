import copy

import numpy as np
from tqdm import tqdm
import time, os
import os.path as osp
from utils import *
from argument import parse_args
from models import parse_model
import torch
import torch.nn.functional as F


def run(args):
    fix_seed(args.seed)
    logger = Logger(args.runs, args)

    data, args = load_data(args)
    data = preprocess_data(args, data)

    model = parse_model(args, data)

    if args.cuda in [0, 1, 2, 3]:
        device = torch.device('cuda:' + str(args.cuda)
                              if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # torch.set_default_device(device)

    model = model.to(device)
    data = data.to(device)
    # data_pre = copy.deepcopy(data)

    activation, loss_func, eval_func = get_functions()
    num_params = count_parameters(model)

    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)

    runtime_list = []

    for run in tqdm(range(args.runs)):

        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        for epoch in range(args.epochs):

            #         Training part
            model.train()
            optimizer.zero_grad()

            out = model(data)
            logit = activation(out, dim=1)
            loss = loss_func(logit[train_idx], data.y[train_idx])

            loss.backward()

            optimizer.step()
            model.eval()
            with torch.no_grad():

                out = model(data)
                logit = F.log_softmax(out, dim=1)

                result = evaluate(y_true=data.y, y_pred=logit, split_idx=split_idx, eval_func=eval_func)
                logger.add_result(run, result[:3])


            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%'
                      )

        end_time = time.time()
        runtime_list.append(end_time - start_time)

    # emb_file_path = osp.join(args.root_dir, "ablations", "TSNE", "data", f"emb_{args.dname}_{args.method}.txt")
    # acc_file_path = osp.join(args.root_dir, "ablations", "TSNE", "data", f"acc_{args.dname}_{args.method}.txt")
    # np.savetxt(emb_file_path, best_embedding.cpu().detach().numpy())
    # np.savetxt(acc_file_path, accs)
        # logger.print_statistics(run)


    ### Save results ###

    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)


    best_val, best_test = logger.print_statistics()
    res_root = osp.join(args.root_dir, 'hyperparameter_tunning')
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    print(f"Saving results to {filename}")
    with open(filename, 'a+') as write_obj:
        cur_line = f'{args.method}_{args.lr}_{args.wd}'
        cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
        cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
        cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
        cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    if args.method == "model":
        experiments_file = f'{res_root}/{args.method}.csv'
        with open(experiments_file, 'a+') as f:
            cur_line = f'{args.dname}'
            cur_line += f',{args.lr}'
            cur_line += f',{args.wd}'
            cur_line += f',{args.dropout}'
            cur_line += f',{args.hidden_dim}'
            cur_line += f',{args.MLP_hidden}'
            cur_line += f',{best_val.mean():.3f} ± {best_val.std():.3f}'
            cur_line += f',{best_test.mean():.3f} ± {best_test.std():.3f}'
            cur_line += f',{num_params}, {avg_time:.2f}s, {std_time:.2f}s'
            cur_line += f',{avg_time // 60}min{(avg_time % 60):.2f}s'
            cur_line += f'\n'
            f.write(cur_line)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    run(args)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
