from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def make_distributed_dataloader(
    dataset,
    dataloader_cfg: dict,
    num_processes: int,
    rank: int,
    seed: int = 42,
    shuffle_default: bool = True,
    drop_last_default: bool = True,
):
    """Build a DataLoader where `dataloader_cfg['batch_size']` is the GLOBAL batch summed
    across processes. In multi-GPU mode the dataset is sharded via DistributedSampler;
    in single-GPU mode the loader is constructed directly.

    Returns (DataLoader, DistributedSampler | None). `set_epoch` should be called on the
    returned sampler each epoch when shuffling is enabled.
    """
    cfg = dict(dataloader_cfg)
    global_bs = cfg["batch_size"]
    assert global_bs % num_processes == 0, (
        f"batch_size ({global_bs}) must be divisible by num GPUs ({num_processes})"
    )
    cfg["batch_size"] = global_bs // num_processes
    if num_processes > 1:
        shuffle = cfg.pop("shuffle", shuffle_default)
        drop_last = cfg.pop("drop_last", drop_last_default)
        sampler = DistributedSampler(
            dataset, num_replicas=num_processes, rank=rank, shuffle=shuffle, seed=seed
        )
        return DataLoader(dataset, sampler=sampler, drop_last=drop_last, **cfg), sampler
    return DataLoader(dataset, **cfg), None

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def dict_apply_with_exclude(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor], 
        ignore_keys: List[str] = []
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if key in ignore_keys:
            result[key] = value
        elif isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer
