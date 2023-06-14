import torch
import torch.nn.functional as F


def flatten_list(x):
    return [item for sublist in x for item in sublist]


def multi_task_loss(inputs, targets, logscale=True):
    if isinstance(targets[0], list):
        targets = flatten_list(targets)

    if isinstance(inputs[0], tuple):
        inputs = flatten_list(inputs)

    losses = []
    for input, target in zip(inputs, targets):
        if isinstance(input, list):
            assert (len(input) - target.size(1))<=1, (len(input), target.size(1))
            for input_i, target_i in zip(input, target.T):  # for perception, don't feed dangerous as a target
                if not logscale:
                    input_i = torch.log(input_i)
                losses.append(F.nll_loss(input_i, target_i))
        else:
            if not logscale:
                input = torch.log(input)
            losses.append(F.nll_loss(input, target))
    assert not torch.isnan(sum(losses))
    return sum(losses) / len(losses), losses


def multi_task_metric(inputs, targets, return_supports=False):
    if isinstance(targets[0], list):
        targets = flatten_list(targets)

    if isinstance(inputs[0], tuple):
        inputs = flatten_list(inputs)

    metrics = []
    for input, target in zip(inputs, targets):
        if isinstance(input, list):
            assert (len(input) - target.size(1))<=1, (len(input), target.size(1))
            for input_i, target_i in zip(input, target.T):  # for perception, don't feed dangerous as a target
                metrics.append((input_i.argmax(1) == target_i).float().mean().item())
        else:
            metrics.append((input.argmax(1) == target).float().mean().item())

    n_supports = inputs[0].size(0)
    if return_supports:
        return metrics, n_supports
    return metrics

