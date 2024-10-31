import os

import torch


def compute_dict_mean(epoch_dicts: list):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d: dict):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def add_prefix(d: dict, prefix: str, additional_dict=None):
    new_d = dict()
    for k, v in d.items():
        new_d[f'{prefix}{k}'] = v
    if additional_dict is not None:
        new_d.update(additional_dict)
    return new_d


def save_checkpoint(policy, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
    }, path)


def prepare_dirs(cfg):
    if not os.path.exists(cfg.base_dir):
        os.mkdir(cfg.base_dir)
    for dir_name in ('ckpts', 'data', 'videos'):
        path = os.path.join(cfg.base_dir, dir_name)
        if not os.path.exists(path):
            os.mkdir(path)
