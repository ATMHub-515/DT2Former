# DT²Former

Source utilities for **DT²Former: A Detection–Tracking Decoupled Transformer for Heterogeneous Airport Surface Multi-Object Tracking**.

**Release status — partial.** This release contains the MeMOTR-based reference pipeline, dataset adapters, and tracking evaluation utilities. **The DT²Former model, decoupled decoder, auxiliary-training implementation, and trained weights are not released yet.** This is not a reproduction package for the results reported in the paper. We will release the remaining code once the paper is accepted.

## Contents

- `models/`, `structures/`, `utils/`, `models/ops/`: MeMOTR reference model and supporting operators (DT²Former-specific components excluded).
- `data/`, `third_party/trackeval/`, `evaluate.py`: AirMOT and converted UA-DETRAC data loading and class-wise MOT evaluation.
- `main.py`, `train.py`, `infer.py`, `run.sh`, `configs/airmot.yaml`: reference-model training and sequence inference.

## Setup

Tested during package preparation with Linux, Python 3.12, PyTorch 2.5.1, and torchvision 0.20.1. Full training, CUDA compilation, and multi-GPU execution of this staged release have **not** been validated.

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Training and inference require the CUDA deformable-attention extension, a compatible NVIDIA driver, a CUDA toolkit with `nvcc`, and a C++ compiler:

```bash
python -m pip install --no-build-isolation ./models/ops
```

## Data and predictions

Supply your own data in the converted MOT layout. An example validation sequence:

```text
datasets/
  AirMot/
    train_seqmap.txt
    train/<sequence>/img1/000000.PNG
    train/<sequence>/gt/gt.txt
    train/<sequence>/seqinfo.ini
```

`train_seqmap.txt` contains a `name` header followed by sequence names. `seqinfo.ini` must define `[Sequence]` and `seqLength`. Frames are one-based; AirMOT image filenames start at `000000.PNG`. Converted UA-DETRAC images start at `000001.jpg`. Raw UA-DETRAC XML is not accepted.

Each GT/prediction row contains eight whitespace-separated values:

```text
frame_id track_id class_id left top width height score
```

AirMOT class IDs: `0 airplane`, `1 person`, `2 baggage_tug`, `3 follow_me_vehicle`. UA-DETRAC: `0 car`, `1 van`, `2 bus`, `3 others`. Prediction files are named `<sequence>.txt`, one per evaluated sequence. Dataset files and redistribution rights are **not** provided by this repository.

## Evaluation

Given a folder containing predictions for all sequences in the requested split:

```bash
python main.py evaluate \
  --dataset AirMot \
  --data-root /path/to/datasets \
  --predictions /path/to/tracker \
  --split val \
  --output outputs/airmot_eval
```

Outputs include `summary.csv` and native TrackEval summaries. HOTA, CLEAR, and Identity metrics are reported by class; TrackEval also produces class-averaged and detection-averaged combinations. The latter is **not** the arithmetic mean of class-wise HOTA values.

## Reference model (MeMOTR only)

The bundled `configs/airmot.yaml` selects `MODEL: memotr`. To run the reference pipeline with your own converted data and trusted weights:

```bash
GPUS=0 bash run.sh train \
  --config configs/airmot.yaml \
  --data-root /path/to/datasets \
  --pretrained /path/to/trusted_detector_checkpoint.pth \
  --output runs/memotr_airmot
```

This configuration is provided as an editable reference, **not** as a claim of reproducing any published result. Do not load untrusted PyTorch checkpoints.

## Attribution and license

The reference pipeline is derived from [MeMOTR](https://github.com/MCG-NJU/MeMOTR) and related open-source tracking work; the evaluation subset is adapted from [TrackEval](https://github.com/JonathonLuiten/TrackEval). Retain `LICENSE`, `THIRD_PARTY_NOTICES.txt`, `models/ops/LICENSE`, and `third_party/trackeval/LICENSE` when redistributing applicable components. The CUDA operators have separate Apache-2.0 terms.
