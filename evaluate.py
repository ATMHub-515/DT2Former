import csv
import json
import sys
from pathlib import Path
from data.dataset import CLASSES, read_rows, sequence_length, sequence_names


def evaluate(dataset, data_root, predictions, output, split="val", workers=1):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party"))
    import trackeval

    root = Path(data_root).resolve() / dataset
    predictions = Path(predictions).resolve()
    output = Path(output).resolve()
    names = sequence_names(root, split)
    lengths = {name: sequence_length(root / split / name) for name in names}
    expected = {f"{name}.txt" for name in names}
    actual = {p.name for p in predictions.glob("*.txt")}
    if expected != actual:
        raise ValueError(
            f"Prediction files mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for name in names:
        for path in (root / split / name / "gt/gt.txt", predictions / f"{name}.txt"):
            for _ in read_rows(path, lengths[name], len(CLASSES[dataset])):
                pass
    dataset_class = (
        trackeval.datasets.AirMot if dataset == "AirMot" else trackeval.datasets.UADETRAC
    )
    adapter = dataset_class(
        {
            "GT_FOLDER": str(root),
            "SPLIT_TO_EVAL": split,
            "TRACKERS_FOLDER": str(predictions.parent),
            "TRACKERS_TO_EVAL": [predictions.name],
            "TRACKER_SUB_FOLDER": "",
            "OUTPUT_FOLDER": str(output / "native"),
            "OUTPUT_SUB_FOLDER": "",
            "SEQ_INFO": lengths,
            "CLASSES_TO_EVAL": list(CLASSES[dataset]),
            "PRINT_CONFIG": False,
        }
    )
    engine = trackeval.Evaluator(
        {
            "USE_PARALLEL": workers > 1,
            "NUM_PARALLEL_CORES": workers,
            "PRINT_ONLY_COMBINED": True,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "PLOT_CURVES": False,
            "BREAK_ON_ERROR": True,
            "LOG_ON_ERROR": str(output / "errors.log"),
        }
    )
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
        trackeval.metrics.Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}),
    ]
    results, status = engine.evaluate([adapter], metrics)
    if status[adapter.get_name()][predictions.name] != "Success":
        raise RuntimeError(status)
    rows = []
    for path in sorted((output / "native" / predictions.name).glob("*_summary.txt")):
        lines = path.read_text().splitlines()
        rows.append(
            {
                "class": path.name.removesuffix("_summary.txt"),
                **dict(zip(lines[0].split(), lines[1].split())),
            }
        )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "dataset": dataset,
        "split": split,
        "sequences": names,
        "classes": CLASSES[dataset],
        "metrics": ["HOTA", "CLEAR", "Identity"],
        "clear_identity_iou": 0.5,
        "predictions": str(predictions),
        "aggregation": "Native TrackEval sequence combination, followed by class/detection-averaged summaries",
    }
    (output / "evaluation.json").write_text(json.dumps(metadata, indent=2))
    return results
