import numpy as np
import torch.nn.functional as F


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()

    #     ipdb.set_trace()
    #     for i in range(y_true.shape[1]):
    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def get_activation_function():
    return F.log_softmax


def get_loss_function():
    return F.nll_loss


def get_evaluation_function():
    return eval_acc


def get_functions():
    activation = get_activation_function()
    loss_func = get_loss_function()
    eval_func = get_evaluation_function()
    return activation, loss_func, eval_func
