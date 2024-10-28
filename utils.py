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


def add_prefix(d: dict, prefix: str):
    new_d = dict()
    for k, v in d.items():
        new_d[f'{prefix}{k}'] = v.detach()
    return new_d
