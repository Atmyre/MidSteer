#!/usr/bin/env bash

acquire_gpu() {
    while true; do
        gpu_id=$(
            flock 99
            line=$(head -n 1 $LOCK_FILE)
            if [ -n "$line" ]; then
                if [ "$(uname)" == "Darwin" ]; then
                    sed -i '' '1d' $LOCK_FILE
                else
                    sed -i '1d' $LOCK_FILE
                fi
            fi
            echo $line
        ) 99> $LOCK_FILE
        if [ -n "$gpu_id" ]; then
            break
        fi
        sleep 1
    done
    echo $gpu_id
}

release_gpu() {
    (
        flock 99
        echo $1 >> $LOCK_FILE
    ) 99> $LOCK_FILE
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