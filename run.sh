set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -eq 0 ]]; then
    echo 'Usage: GPUS=0,1 PYTHON=/path/to/python bash run.sh train|infer|evaluate|check ...' >&2
    exit 2
fi
PYTHON="${PYTHON:-python}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export CUDA_VISIBLE_DEVICES="${GPUS:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ "$1" == train || "$1" == infer ]]; then
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    declare -A selected=()
    for device in "${devices[@]}"; do
        [[ "$device" =~ ^[0-9]+$ ]] || { echo 'GPUS must contain comma-separated physical GPU indices' >&2; exit 2; }
        [[ ! -v "selected[$device]" ]] || { echo "Duplicate GPU index: $device" >&2; exit 2; }
        selected[$device]=1
    done
    if (( ${#devices[@]} > 1 )); then
        exec "$PYTHON" -s -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#devices[@]}" main.py "$@"
    fi
fi
exec "$PYTHON" -s main.py "$@"
