from pathlib import Path
import yaml


def load_config(path, overrides=()):
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator or key not in config:
            raise ValueError(f"Unknown configuration override: {item}")
        config[key] = yaml.safe_load(value)
    validate_config(config)
    return config


def validate_config(c):
    if c["MODEL"] != "memotr" or c["DATASET"] not in ("AirMot", "UA-DETRAC"):
        raise ValueError("This partial release supports MeMOTR and the AirMot/UA-DETRAC datasets")
    for key in (
        "BATCH_SIZE",
        "ACCUMULATION_STEPS",
        "EPOCHS",
        "SAVE_PERIOD",
        "LOG_INTERVAL",
        "NUM_DET_QUERIES",
    ):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["NUM_WORKERS"] < 0 or c["NUM_DEC_LAYERS"] < 2:
        raise ValueError("NUM_WORKERS >= 0 and NUM_DEC_LAYERS >= 2 are required")
    if c["NUM_ENC_LAYERS"] < 1 or c["NUM_HEADS"] < 1 or c["HIDDEN_DIM"] < 32:
        raise ValueError("Invalid encoder or embedding dimensions")
    if c["NUM_FEATURE_LEVELS"] != 4 or c["CHECKPOINT_LEVEL"] not in (1, 2, 3):
        raise ValueError("This release uses four feature levels and checkpoint levels 1, 2 or 3")
    if not c["AUX_LOSS"] or not c["RETURN_INTER_DEC"]:
        raise ValueError("MeMOTR query propagation requires intermediate decoder outputs")
    if c["VISUALIZE"] or c["EXTRA_TRACK_ATTN"]:
        raise ValueError("Visualization and extra track attention are not included in this release")
    if c["HIDDEN_DIM"] % 32 or c["HIDDEN_DIM"] % c["NUM_HEADS"]:
        raise ValueError("HIDDEN_DIM must be divisible by 32 and NUM_HEADS")
    if len(c["AUX_LOSS_WEIGHT"]) != c["NUM_DEC_LAYERS"] - 1:
        raise ValueError("AUX_LOSS_WEIGHT must have NUM_DEC_LAYERS - 1 entries")
    if not 0 <= c["MERGE_DET_TRACK_LAYER"] < c["NUM_DEC_LAYERS"]:
        raise ValueError("Invalid MERGE_DET_TRACK_LAYER")
    if c["SAMPLE_STEPS"] != sorted(set(c["SAMPLE_STEPS"])):
        raise ValueError("SAMPLE_STEPS must be strictly increasing")
    if len(c["SAMPLE_LENGTHS"]) != len(c["SAMPLE_STEPS"]) + 1 or len(c["SAMPLE_INTERVALS"]) != len(
        c["SAMPLE_LENGTHS"]
    ):
        raise ValueError("Sampling stages have inconsistent lengths")
    if min(c["SAMPLE_LENGTHS"]) < 2 or min(c["SAMPLE_INTERVALS"]) < 1:
        raise ValueError("Clips require at least two frames and positive intervals")
    for key in ("DET_SCORE_THRESH", "TRACK_SCORE_THRESH", "RESULT_SCORE_THRESH", "UPDATE_THRESH"):
        if not 0 <= c[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
