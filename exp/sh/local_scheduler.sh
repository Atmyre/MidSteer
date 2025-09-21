#!/usr/bin/env bash
set -u  # fail on unset vars

: "${LOCK_FILE:?LOCK_FILE must be set to the path of the GPU pool file}"
LOCK_GUARD="${LOCK_FILE}.lock"   # separate file only for locking

acquire_gpu() {
  while true; do
    local gpu_id
    gpu_id="$(
      # Take an exclusive lock, then read first line and delete it with sed
      flock -x "$LOCK_GUARD" bash -c '
        file="$1"
        # read first line if available
        if IFS= read -r line <"$file"; then
          # remove first line with sed, keeping your original logic
          if [ "$(uname)" = "Darwin" ]; then
            sed -i "" "1d" "$file"
          else
            sed -i "1d" "$file"
          fi
          printf "%s" "$line"
        fi
      ' _ "$LOCK_FILE"
    )"

    if [ -n "${gpu_id:-}" ]; then
      echo "$gpu_id"
      return 0
    fi
    sleep 1
  done
}

release_gpu() {
  local id="${1:?gpu id required}"
  # Append the id back while holding the same lock
  flock -x "$LOCK_GUARD" bash -c '
    printf "%s\n" "$1" >> "$2"
  ' _ "$id" "$LOCK_FILE"
}

run_command_with_params_on_gpu() {
  set +x
  if [ $# -eq 0 ]; then
    echo "Error: No command provided to run_command_with_params_on_gpu" >&2
    return 1
  fi

  local gpu_id
  gpu_id="$(acquire_gpu)" || return 1
  echo "Acquired GPU $gpu_id"

  echo "Running command on GPU $gpu_id: $*"
  CUDA_VISIBLE_DEVICES=$gpu_id "$@"
  local exit_code=$?

  release_gpu "$gpu_id"
  echo "Released GPU $gpu_id"
  set -x

  return $exit_code
}