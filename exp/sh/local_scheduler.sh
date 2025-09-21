#!/usr/bin/env bash


: "${LOCK_FILE:?LOCK_FILE must be set to the path of the GPU pool file}"
LOCK_PATH="$LOCK_FILE"          # file that stores available GPU ids (one per line)
LOCK_GUARD="$LOCK_FILE.lock"    # separate file used only for locking


acquire_gpu() {
    while true; do
        gpu_id=$(
            flock -x $LOCK_GUARD
            line=$(head -n 1 $LOCK_PATH)
            if [ -n "$line" ]; then
                if [ "$(uname)" == "Darwin" ]; then
                    sed -i '' '1d' $LOCK_PATH
                else
                    sed -i '1d' $LOCK_PATH
                fi
            fi
            printf "%s" $line
        )
        if [ -n "${gpu_id:-}" ]; then
            echo $gpu_id
            return 0
        fi
        sleep 1
    done
}

release_gpu() {
    local id="${1:?gpu id required}"
    (
        flock -x $LOCK_GUARD
        printf "%s\n" $id >> $LOCK_PATH
    )
}

run_command_with_params_on_gpu() {
    set +x
    if [ $# -eq 0 ]; then
        echo "Error: No command provided to run_command_with_params_on_gpu"
        return 1
    fi
    
    gpu_id=$(acquire_gpu)
    echo "Acquired GPU $gpu_id"
    
    # Set CUDA_VISIBLE_DEVICES to the acquired GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Execute the command with all provided arguments
    echo "Running command on GPU $gpu_id: $@"
    "$@"
    local exit_code=$?
    
    # Release the GPU after command completion
    release_gpu $gpu_id
    echo "Released GPU $gpu_id"
    set -x

    return $exit_code
}