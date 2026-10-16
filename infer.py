from pathlib import Path
import torch
from torch.utils.data import DataLoader
from checkpoint import read_checkpoint
from data.dataset import SequenceDataset, sequence_names
from models import build_model
from models.runtime_tracker import RuntimeTracker
from structures.track_instances import TrackInstances
from utils.box_ops import box_cxcywh_to_xyxy
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.utils import distributed_rank, distributed_world_size, is_distributed


@torch.no_grad()
def infer(config, checkpoint, output, split):
    model = build_model(config).eval()
    model.load_state_dict(read_checkpoint(checkpoint)["model"], strict=True)
    root = Path(config["DATA_ROOT"]) / config["DATASET"]
    names = sequence_names(root, split)
    destination = Path(output) / "tracker"
    destination.mkdir(exist_ok=True)
    device = next(model.parameters()).device
    for name in names[distributed_rank() :: distributed_world_size()]:
        tracker = RuntimeTracker(
            det_score_thresh=config["DET_SCORE_THRESH"],
            track_score_thresh=config["TRACK_SCORE_THRESH"],
            miss_tolerance=config["MISS_TOLERANCE"],
            use_motion=False,
            use_dab=config["USE_DAB"],
        )
        tracks = [
            TrackInstances(
                hidden_dim=model.hidden_dim, num_classes=model.num_classes, use_dab=model.use_dab
            ).to(device)
        ]
        dataset = SequenceDataset(
            root / split / name, config["DATASET"], config["TEST_SIZE"], config["MAX_SIZE"]
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=config["NUM_WORKERS"], shuffle=False)
        target = destination / f"{name}.txt"
        temporary = target.with_suffix(".txt.partial")
        with temporary.open("x") as stream:
            for images, frames, widths, heights in loader:
                frame = tensor_list_to_nested_tensor([images[0]]).to(device)
                result = model(frame, tracks)
                previous, new = tracker.update(result, tracks)
                tracks = model.postprocess_single_frame(previous, new, None)
                current = tracks[0].to("cpu")
                if len(current) == 0:
                    continue
                score = current.scores.max(-1).values
                label = current.labels
                width, height = (int(widths[0]), int(heights[0]))
                area = current.boxes[:, 2] * width * current.boxes[:, 3] * height
                keep = (
                    (score > config["RESULT_SCORE_THRESH"])
                    & (area > config["MIN_BOX_AREA"])
                    & (current.ids >= 0)
                )
                boxes = box_cxcywh_to_xyxy(current.boxes[keep]) * torch.tensor(
                    [width, height, width, height]
                )
                for identity, category, confidence, box in zip(
                    current.ids[keep], label[keep], score[keep], boxes
                ):
                    x1, y1, x2, y2 = box.tolist()
                    stream.write(
                        f"{int(frames[0])} {int(identity)} {int(category)} {x1:.6f} {y1:.6f} {x2 - x1:.6f} {y2 - y1:.6f} {float(confidence):.8f}\n"
                    )
        temporary.rename(target)
        print(f"Completed {name}: {target}", flush=True)
    if is_distributed():
        torch.distributed.barrier()
