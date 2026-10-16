import torch


def build_model(config):
    if config["MODEL"] != "memotr":
        raise ValueError("Only the MeMOTR reference model is included in this partial release")
    from .memotr import build
    return build(config).to(torch.device(config["DEVICE"]))


def build_criterion(config):
    if config["MODEL"] != "memotr":
        raise ValueError("Only the MeMOTR reference model is included in this partial release")
    from .criterion import build
    return build(config)
