import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path
import torch
import yaml
from config import load_config, validate_config


class Tee:

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, text):
        self.terminal.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def environment():
    packages = ("torch", "torchvision", "numpy", "scipy", "PyYAML", "Pillow", "opencv-python")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_build": torch.version.cuda,
        "packages": {key: importlib.metadata.version(key) for key in packages},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def prepare_output(output, resume, rank, world):
    status = [None]
    if rank == 0:
        try:
            Path(output).mkdir(parents=True, exist_ok=resume)
            status[0] = {"ok": True}
        except Exception as error:
            status[0] = {"error": str(error)}
    if world > 1:
        torch.distributed.broadcast_object_list(status)
    if "error" in status[0]:
        raise RuntimeError(status[0]["error"])


def parser():
    root = argparse.ArgumentParser(description="MeMOTR reference training and tracking evaluation (partial release)")
    commands = root.add_subparsers(dest="command", required=True)
    for mode in ("train", "infer", "check"):
        sub = commands.add_parser(mode)
        sub.add_argument("--config", default=str(Path(__file__).parent / "configs/airmot.yaml"))
        sub.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        sub.add_argument("--data-root")
        if mode != "check":
            sub.add_argument("--output", required=mode == "infer")
        if mode == "train":
            group = sub.add_mutually_exclusive_group()
            group.add_argument("--pretrained")
            group.add_argument("--resume")
        if mode == "infer":
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--split", default="val", choices=("train", "val", "test"))
    sub = commands.add_parser("evaluate")
    sub.add_argument("--dataset", default="AirMot", choices=("AirMot", "UA-DETRAC"))
    sub.add_argument("--data-root", required=True)
    sub.add_argument("--predictions", required=True)
    sub.add_argument("--split", default="val", choices=("train", "val", "test"))
    sub.add_argument("--output", required=True)
    sub.add_argument("--workers", type=int, default=1)
    return root


def main():
    args = parser().parse_args()
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if args.command in ("check", "evaluate") and world != 1:
        raise ValueError("check/evaluate must run as a single process")
    if args.command == "check":
        from models import build_model, build_criterion

        config = load_config(args.config, args.set)
        config["DEVICE"] = "cpu"
        config["BACKBONE_PRETRAINED"] = False
        model = build_model(config)
        build_criterion(config)
        print(
            json.dumps(
                {
                    "environment": environment(),
                    "model": type(model).__name__,
                    "parameters": sum((p.numel() for p in model.parameters())),
                    "checks": "Config, imports, CPU model/criterion construction; no model forward or GPU execution",
                },
                indent=2,
            )
        )
        return
    resume = None
    if args.command == "train" and args.resume:
        from checkpoint import read_checkpoint

        resume = read_checkpoint(args.resume)
        for key in ("optimizer", "scheduler", "states", "config", "rng_states"):
            if resume.get(key) is None:
                raise ValueError(f"Checkpoint is not resumable: missing {key}")
        config = dict(resume["config"])
        for item in args.set:
            key, separator, value = item.partition("=")
            if not separator or key not in ("EPOCHS", "NUM_WORKERS"):
                raise ValueError("Resume only allows EPOCHS and NUM_WORKERS overrides")
            config[key] = yaml.safe_load(value)
        args.output = args.output or str(Path(args.resume).resolve().parent)
        if Path(args.output).resolve() != Path(args.resume).resolve().parent:
            raise ValueError("Resume must use the original run directory")
    elif args.command != "evaluate":
        config = load_config(args.config, args.set)
    if args.command != "evaluate":
        if args.data_root:
            config["DATA_ROOT"] = str(Path(args.data_root).resolve())
        validate_config(config)
        if args.command == "train":
            if not args.output:
                raise ValueError("--output is required for a new training run")
            if args.pretrained:
                config["PRETRAINED_MODEL"] = str(Path(args.pretrained).resolve())
        if not str(config["DEVICE"]).startswith("cuda"):
            raise ValueError("Training/inference require the compiled CUDA attention operator")
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config["DEVICE"] = "cuda"
        if world > 1:
            torch.distributed.init_process_group("nccl")
        from utils.utils import set_seed

        set_seed(config["SEED"])
    elif args.workers < 1:
        raise ValueError("--workers must be positive")
    output = Path(args.output).resolve()
    try:
        prepare_output(output, resume is not None, rank, world)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        with (output / f"console_{stamp}_rank{rank}.log").open("x") as stream:
            with (
                contextlib.redirect_stdout(Tee(sys.stdout, stream)),
                contextlib.redirect_stderr(Tee(sys.stderr, stream)),
            ):
                try:
                    print(json.dumps(environment(), indent=2), flush=True)
                    if rank == 0:
                        (output / f"launch_{stamp}.json").write_text(
                            json.dumps(
                                {"arguments": vars(args), "environment": environment()}, indent=2
                            )
                        )
                        if args.command != "evaluate":
                            name = f"config_resume_{stamp}.yaml" if resume else "config.yaml"
                            (output / name).write_text(yaml.safe_dump(config, sort_keys=False))
                    if args.command == "train":
                        from train import train

                        train(config, output, resume)
                    elif args.command == "infer":
                        from infer import infer

                        infer(config, args.checkpoint, output, args.split)
                    else:
                        from evaluate import evaluate

                        evaluate(
                            args.dataset,
                            args.data_root,
                            args.predictions,
                            output,
                            args.split,
                            args.workers,
                        )
                except Exception:
                    traceback.print_exc()
                    raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
