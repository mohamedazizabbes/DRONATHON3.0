#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
MODE="${1:-gpu}"
SEED="${2:-}"
VENV="${SKYFORGE_VENV:-$HOME/skyforge-models/.venv}"
XA="${XAUTHORITY:-$HOME/.Xauthority}"
mkdir -p runs
docker rm -f gzsim >/dev/null 2>&1 || true
ARGS=(--name gzsim -d --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia
      -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2
      --network host --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
      -e DISPLAY="${DISPLAY:-:0}" -e XAUTHORITY=/tmp/.Xauth -v "$XA:/tmp/.Xauth:ro" -v /tmp/.X11-unix:/tmp/.X11-unix:ro
      -v "$PWD/runs:/ws/runs" -v "$PWD/models:/models:ro")
SETUP="source /ws/install/setup.bash"
if [ "$MODE" = "gpu" ] && [ -d "$VENV" ]; then
  ARGS+=(-v "$VENV:$VENV:ro")
  SETUP="$SETUP && export PYTHONPATH=$VENV/lib/python3.12/site-packages:\$PYTHONPATH"
  JURY=/models/jury.pt; TEAM=/models/team.pt
else
  JURY=/models/jury.onnx; TEAM=/models/team.onnx
fi
LAUNCH="ros2 launch skyforge_sim sim.launch.py gui:=true jury_model:=$JURY team_model:=$TEAM fault_script:=none"
[ -n "$SEED" ] && LAUNCH="$LAUNCH seed:=$SEED"
docker run "${ARGS[@]}" skyforge:jazzy-sim bash -c "$SETUP && $LAUNCH" >/dev/null
echo "simulation started ($MODE vision${SEED:+, seed $SEED}); stop with: docker rm -f gzsim"
