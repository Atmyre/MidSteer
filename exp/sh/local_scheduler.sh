#!/usr/bin/env bash
set -u  # fail on unset vars

: "${LOCK_FILE:?LOCK_FILE must be set to the path of the GPU pool file}"
LOCK_PATH="$LOCK_FILE"          # file that stores available GPU ids (one per line)
LOCK_GUARD="$LOCK_FILE.lock"    # separate file used only for locking

acquire_gpu() {
  while true; do
    gpu_id="$(
      flock -x "$LOCK_GUARD" -c '
        # read first line if present
        if IFS= read -r line <"$LOCK_PATH"; then
          # remove first line atomically-ish: write tail to temp then move
          tmp="${LOCK_PATH}.tmp.$$"
          tail -n +2 "$LOCK_PATH" >"$tmp" || exit 1
          mv -f "$tmp" "$LOCK_PATH" || exit 1
          printf "%s" "$line"
        fi
      ' 2>/dev/null
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
  flock -x "$LOCK_GUARD" -c '
    printf "%s\n" "'"$id"'" >> "'"$LOCK_PATH"'"
  ' 2>/dev/null
}

run_command_with_params_on_gpu() {
  if [ $# -eq 0 ]; then
    echo "Error: No command provided to run_command_with_params_on_gpu" >&2
    return 1
  fi

  local gpu_id
  gpu_id="$(acquire_gpu)" || return 1
  echo "Acquired GPU $gpu_id"

  export CUDA_VISIBLE_DEVICES="$gpu_id"
  echo "Running command on GPU $gpu_id: $*"
  "$@"
  local exit_code=$?

  release_gpu "$gpu_id"
  echo "Released GPU $gpu_id"

  return $exit_code
}