import csv
import json
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from checkpoint import atomic_save, initialize_model, restore_rng, rng_state
from data.dataset import TrackingDataset, collate_fn, training_transform
from models import build_model, build_criterion
from structures.track_instances import TrackInstances
from utils.nested_tensor import tensor_list_to_nested_tensor
from utils.utils import distributed_rank, distributed_world_size, is_distributed


class ClipObjective(nn.Module):

    def __init__(self, tracker, criterion):
        super().__init__()
        self.tracker = tracker
        self.criterion = criterion

    def forward(self, batch):
        model = self.tracker
        device = next(model.parameters()).device
        tracks = TrackInstances.init_tracks(
            batch, model.hidden_dim, model.num_classes, device=device, use_dab=model.use_dab
        )
        self.criterion.init_a_clip(batch, model.hidden_dim, model.num_classes, device)
        count = len(batch["imgs"][0])
        for frame_index in range(count):
            frames = [images[frame_index] for images in batch["imgs"]]
            output = model(tensor_list_to_nested_tensor(frames).to(device), tracks)
            previous, new, unmatched = self.criterion.process_single_frame(
                output, tracks, frame_index
            )
            if frame_index + 1 < count:
                tracks = model.postprocess_single_frame(previous, new, unmatched)
        losses, _ = self.criterion.get_mean_by_n_gts()
        total = self.criterion.get_sum_loss_dict(losses)
        return (total, {k: v.detach() for k, v in losses.items()})


def parameter_groups(model, c):
    groups = {key: [] for key in ("LR", "LR_BACKBONE", "LR_POINTS")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "LR_BACKBONE"
            if name.startswith("backbone.")
            else (
                "LR_POINTS"
                if any((s in name for s in ("reference_points", "sampling_offsets")))
                else "LR"
            )
        )
        groups[key].append(parameter)
    return [{"params": params, "lr": c[key]} for key, params in groups.items() if params]


def train(config, output, resume=None):
    rank, world = (distributed_rank(), distributed_world_size())
    output = Path(output)
    model = build_model(config)
    if resume is None and config["PRETRAINED_MODEL"]:
        report = initialize_model(
            model, config["PRETRAINED_MODEL"], config["PRETRAINED_CLASS_HEAD"]
        )
        if rank == 0:
            (output / "initialization.json").write_text(json.dumps(report, indent=2))
            print(
                f"Initialized {len(report['loaded_keys'])} tensors; see initialization.json",
                flush=True,
            )
    optimizer = torch.optim.AdamW(
        parameter_groups(model, config), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"]
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config["LR_DROP_MILESTONES"], gamma=config["LR_DROP_RATE"]
    )
    start_epoch, global_step = (0, 0)
    if resume is not None:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch = resume["states"]["start_epoch"]
        global_step = resume["states"]["global_iters"]
    objective = ClipObjective(model, build_criterion(config))
    if is_distributed():
        objective = DistributedDataParallel(
            objective, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )
    dataset = TrackingDataset(config, transform=training_transform(config))
    if resume is not None:
        states = resume.get("rng_states")
        if states is None or len(states) != world:
            raise ValueError("Resume requires saved RNG states and the original world size")
        restore_rng(states[rank])
    if start_epoch >= config["EPOCHS"]:
        raise ValueError("Checkpoint has already reached EPOCHS")
    print(
        f"rank={rank}, world={world}, batch_per_rank={config['BATCH_SIZE']}, accumulation={config['ACCUMULATION_STEPS']}, nominal_effective_batch={world * config['BATCH_SIZE'] * config['ACCUMULATION_STEPS']}",
        flush=True,
    )
    for epoch in range(start_epoch, config["EPOCHS"]):
        objective.train()
        dataset.set_epoch(epoch)
        sampler = (
            DistributedSampler(dataset, shuffle=True, seed=config["SEED"])
            if is_distributed()
            else RandomSampler(dataset)
        )
        if is_distributed():
            sampler.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"],
            collate_fn=collate_fn,
            pin_memory=True,
        )
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        totals = {}
        for index, batch in enumerate(loader):
            total, losses = objective(batch)
            finite = torch.isfinite(total.detach()).to(torch.int32)
            if is_distributed():
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {index}")
            accumulation = config["ACCUMULATION_STEPS"]
            group_start = index // accumulation * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            (total / group_size).backward()
            if (index + 1) % accumulation == 0 or index + 1 == len(loader):
                if config["CLIP_MAX_NORM"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["CLIP_MAX_NORM"], error_if_nonfinite=True
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            values = {"total_loss": total.detach(), **losses}
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            global_step += 1
            if rank == 0 and index % config["LOG_INTERVAL"] == 0:
                print(
                    f"epoch={epoch} step={index}/{len(loader)} loss={total.item():.6f}", flush=True
                )
        used_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        keys = sorted(totals)
        mean = torch.tensor(
            [totals[key] / len(loader) for key in keys], device=next(model.parameters()).device
        )
        if is_distributed():
            torch.distributed.all_reduce(mean)
            mean /= world
        current_rng = rng_state()
        all_rng = [current_rng]
        if is_distributed():
            all_rng = [None] * world
            torch.distributed.all_gather_object(all_rng, current_rng)
        if rank == 0:
            row = {
                "epoch": epoch,
                "seconds": time.monotonic() - started,
                "lr": used_lr,
                **dict(zip(keys, mean.tolist())),
            }
            csv_path = output / "results.csv"
            exists = csv_path.exists()
            with csv_path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "states": {"start_epoch": epoch + 1, "global_iters": global_step},
                "config": config,
                "rng_states": all_rng,
                "world_size": world,
            }
            atomic_save(state, output / "last.pth")
            if (epoch + 1) % config["SAVE_PERIOD"] == 0:
                atomic_save(state, output / f"checkpoint_{epoch:03d}.pth")
            print(f"Completed epoch {epoch}; saved {output / 'last.pth'}", flush=True)
        if is_distributed():
            torch.distributed.barrier()
