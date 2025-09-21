#!/usr/bin/env bash

acquire_gpu() {
    while true; do
        gpu_id=$(flock $LOCK_FILE ./exp/sh/get_line_from_file_and_remove.sh $LOCK_FILE)
        if [ -n "$gpu_id" ]; then
            break
        fi
        sleep 1
    done
    echo $gpu_id
}

release_gpu() {
    flock $LOCK_FILE ./exp/sh/add_line_to_file.sh $1 $LOCK_FILE
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