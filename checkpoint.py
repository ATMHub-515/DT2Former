import os
import random
from pathlib import Path
import numpy as np
import torch


def read_checkpoint(path):
    result = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(result, dict) or "model" not in result:
        raise ValueError("Expected a trusted MeMOTR-style checkpoint containing model state")
    return result


def initialize_model(model, path, load_class_head=False):
    source = read_checkpoint(path)["model"]
    target = model.state_dict()
    loaded, skipped = ({}, [])
    for original, value in source.items():
        key = original.removeprefix("module.")
        if key.startswith("backbone.0.body."):
            key = "backbone.backbone.backbone." + key[len("backbone.0.body.") :]
        if key.startswith("input_proj."):
            key = "feature_projs." + key[len("input_proj.") :]
        key = {
            "query_embed.weight": "det_query_embed",
            "tgt_embed.weight": "det_query_embed",
            "refpoint_embed.weight": "det_anchor",
        }.get(key, key)
        candidates = [key]
        matched = False
        for candidate in candidates:
            if "class_embed" in candidate and (not load_class_head):
                continue
            if candidate in target and target[candidate].shape == value.shape:
                loaded[candidate] = value
                matched = True
        if not matched:
            skipped.append(original)
    if not loaded:
        raise ValueError("No compatible parameters were found in the checkpoint")
    incompatible = model.load_state_dict(loaded, strict=False)
    return {
        "loaded_keys": sorted(loaded),
        "skipped_source_keys": skipped,
        "new_target_keys": incompatible.missing_keys,
    }


def atomic_save(payload, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])
