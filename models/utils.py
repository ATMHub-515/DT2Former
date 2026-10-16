import torch
import copy
import math
import torch.distributed
import torch.nn as nn


def get_activation_layer(activation: str):
    if activation == "ReLU":
        return nn.ReLU(True)
    elif activation == "GELU":
        return nn.GELU()
    else:
        raise ValueError(f"Do not support activation layer: {activation}")


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])


def pos_to_pos_embed(
    pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi
):
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * torch.div(dim_i, 2, rounding_mode="trunc") / num_pos_feats)
    pos_embed = pos[..., None] / dim_i
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def logits_to_scores(logits: torch.Tensor):
    return logits.sigmoid()
