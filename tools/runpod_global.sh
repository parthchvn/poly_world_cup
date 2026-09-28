#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -u "$repo_dir/tools/runpod_actor_experiment.py" --variant global "$@"
