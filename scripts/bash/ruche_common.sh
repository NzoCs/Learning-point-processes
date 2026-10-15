#!/bin/bash
# Shared by Ruche entry points; installation happens before submission.

ltpp_launch() {
    local device=$1
    shift
    local count=$(( ${#experiments[@]} * ${#datasets[@]} ))
    local phase=${LTPP_PHASE:-all}
    case "$phase" in
        train|test|predict|all) ;;
        *) echo "Invalid LTPP_PHASE: $phase" >&2; return 2 ;;
    esac
    local -a indices command
    local index model dataset output
    local environment=${UV_PROJECT_ENVIRONMENT:-$repo_dir/.venv}
    local python=$environment/bin/python
    if [[ ${1:-} == --dry-run && $# == 1 ]]; then
        mapfile -t indices < <(seq 0 "$((count - 1))")
    elif (( $# == 0 )); then
        index=${SLURM_ARRAY_TASK_ID:-0}
        if [[ ! $index =~ ^[0-9]+$ ]] || (( ${#index} > 6 )); then
            echo "Invalid Slurm array index: $index" >&2; return 2
        fi
        index=$((10#$index))
        if (( index >= count )); then
            echo "Slurm array index $index outside 0-$((count - 1))" >&2; return 2
        fi
        indices=("$index")
        if [[ -z ${SLURM_JOB_ID:-} ]]; then
            echo "Run with sbatch on a compute node, or use --dry-run." >&2; return 2
        fi
    else
        echo "Usage: bash <launcher> [--dry-run]" >&2; return 2
    fi
    if [[ -n ${LTPP_EPOCHS:-} ]] && { [[ ! $LTPP_EPOCHS =~ ^[1-9][0-9]*$ ]] || (( ${#LTPP_EPOCHS} > 6 )); }; then
        echo "LTPP_EPOCHS must be a positive integer." >&2; return 2
    fi
    for index in "${indices[@]}"; do
        model=${experiments[$((index / ${#datasets[@]}))]}
        dataset=${datasets[$((index % ${#datasets[@]}))]}
        output=${LTPP_OUTPUT_ROOT:-$repo_dir/artifacts/ruche}/$device/job-${SLURM_JOB_ID:-pending}-task-$index
        command=("$python" -m scripts.cli run
            --config "$repo_dir/yaml_configs/configs.yaml"
            --model "$model" --dataset-id "$dataset"
            --phase "$phase" --save-dir "$output")
        if [[ -n ${LTPP_EPOCHS:-} ]]; then
            command+=(--epochs "$LTPP_EPOCHS")
        fi
        printf 'task=%s device=%s ' "$index" "$device"
        printf '%q ' "${command[@]}"
        printf '\n'
        if [[ ${1:-} == --dry-run ]]; then
            continue
        fi
        cd -- "$repo_dir"
        if [[ ! -f uv.lock || ! -x $python ]]; then
            echo "Missing uv.lock or Python environment at $environment. Install with uv sync --frozen before submission." >&2; return 2
        fi
        if [[ $device == cpu ]]; then
            export CUDA_VISIBLE_DEVICES=""
        fi
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
        export MPLBACKEND=Agg
        export LTPP_RUN_ID="job-${SLURM_JOB_ID}-task-${index}"
        srun "$python" -m scripts.check_backend --device "$device"
        srun "${command[@]}"
    done
}
