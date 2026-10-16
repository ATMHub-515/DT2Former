import configparser
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from . import transforms as T

CLASSES = {
    "AirMot": ("airplane", "person", "baggage_tug", "follow_me_vehicle"),
    "UA-DETRAC": ("car", "van", "bus", "others"),
}


def sequence_names(root, split):
    root = Path(root)
    seqmap = root / f"{split}_seqmap.txt"
    if seqmap.exists():
        names = [s.strip() for s in seqmap.read_text().splitlines() if s.strip()]
        if names and names[0].lower() == "name":
            names = names[1:]
    else:
        names = sorted((p.name for p in (root / split).iterdir() if p.is_dir()))
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Empty or duplicate sequence list: {root}, {split}")
    for name in names:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Invalid sequence name: {name}")
        if not (root / split / name).is_dir():
            raise FileNotFoundError(root / split / name)
    return names


def sequence_length(sequence):
    path = Path(sequence) / "seqinfo.ini"
    parser = configparser.ConfigParser()
    if not parser.read(path):
        raise FileNotFoundError(path)
    length = parser.getint("Sequence", "seqLength")
    if length < 1:
        raise ValueError(f"Invalid sequence length: {path}")
    return length


def image_path(sequence, dataset, frame):
    filename = f"{frame - 1:06d}.PNG" if dataset == "AirMot" else f"{frame:06d}.jpg"
    return Path(sequence) / "img1" / filename


def read_rows(path, length, num_classes=4):
    seen = set()
    with Path(path).open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                if len(fields) != 8:
                    raise ValueError("expected 8 fields")
                frame, identity, category = map(int, fields[:3])
                x, y, width, height, score = map(float, fields[3:])
                if not all((math.isfinite(v) for v in (x, y, width, height, score))):
                    raise ValueError("non-finite value")
                if not 1 <= frame <= length or identity < 0 or (not 0 <= category < num_classes):
                    raise ValueError("frame, identity or class outside allowed range")
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive box size")
                if (frame, identity) in seen:
                    raise ValueError("duplicate identity within frame")
            except ValueError as error:
                raise ValueError(f"{path}:{line_no}: {error}") from error
            seen.add((frame, identity))
            yield (frame, identity, category, x, y, width, height, score)


def training_transform(config):
    scales = config["TRAIN_SCALES"]
    return T.MultiCompose(
        [
            T.MultiRandomHorizontalFlip(),
            T.MultiRandomSelect(
                T.MultiRandomResize(scales, max_size=config["MAX_SIZE"]),
                T.MultiCompose(
                    [
                        T.MultiRandomResize([800, 1000, 1200]),
                        T.MultiRandomCrop(800, 1200),
                        T.MultiRandomResize(scales, max_size=config["MAX_SIZE"]),
                    ]
                ),
            ),
            T.MultiHSV(),
            T.MultiToTensor(),
            T.MultiNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TrackingDataset(Dataset):

    def __init__(self, config, split="train", transform=None):
        self.config = config
        self.dataset = config["DATASET"]
        self.root = Path(config["DATA_ROOT"]) / self.dataset
        self.split = split
        self.transform = transform
        self.names = sequence_names(self.root, split)
        self.lengths = {}
        self.gts = {}
        for name in self.names:
            sequence = self.root / split / name
            length = sequence_length(sequence)
            self.lengths[name] = length
            rows = defaultdict(list)
            for frame, identity, label, x, y, w, h, _ in read_rows(sequence / "gt/gt.txt", length):
                rows[frame].append((identity, label, x, y, x + w, y + h))
            self.gts[name] = rows
        self.set_epoch(0)

    def set_epoch(self, epoch):
        stage = sum((epoch >= step for step in self.config["SAMPLE_STEPS"]))
        self.clip_length = self.config["SAMPLE_LENGTHS"][stage]
        self.max_interval = self.config["SAMPLE_INTERVALS"][stage]
        self.clips = [
            (name, frame)
            for name in self.names
            for frame in range(1, self.lengths[name] - self.clip_length + 2)
        ]
        if not self.clips:
            raise ValueError("No sequence is long enough for the configured clip length")

    def get_frame(self, name, frame):
        path = image_path(self.root / self.split / name, self.dataset, frame)
        with Image.open(path) as source:
            image = source.convert("RGB")
        rows = self.gts[name][frame]
        boxes = torch.tensor([r[2:] for r in rows], dtype=torch.float32).reshape(-1, 4)
        info = {
            "boxes": boxes,
            "ids": torch.tensor([r[0] for r in rows], dtype=torch.long),
            "labels": torch.tensor([r[1] for r in rows], dtype=torch.long),
            "areas": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "frame_idx": torch.tensor(frame),
        }
        return (image, info)

    def __getitem__(self, index):
        name, begin = self.clips[index]
        limit = min(self.max_interval, (self.lengths[name] - begin) // (self.clip_length - 1))
        interval = random.randint(1, limit)
        images, infos = zip(
            *(self.get_frame(name, begin + i * interval) for i in range(self.clip_length))
        )
        if self.transform is not None:
            images, infos = self.transform(list(images), list(infos))
        return {"imgs": list(images), "infos": list(infos)}

    def __len__(self):
        return len(self.clips)


class SequenceDataset(Dataset):

    def __init__(self, sequence, dataset, short_size=800, max_size=1536):
        self.sequence = Path(sequence)
        self.dataset = dataset
        self.length = sequence_length(sequence)
        self.short_size = short_size
        self.max_size = max_size

    def __getitem__(self, index):
        frame = index + 1
        with Image.open(image_path(self.sequence, self.dataset, frame)) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = min(self.short_size / min(width, height), self.max_size / max(width, height))
        image = F.resize(image, [max(1, int(height * scale)), max(1, int(width * scale))])
        image = F.normalize(F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return (image, frame, width, height)

    def __len__(self):
        return self.length


def collate_fn(batch):
    return {key: [sample[key] for sample in batch] for key in ("imgs", "infos")}
