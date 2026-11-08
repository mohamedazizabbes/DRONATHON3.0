#!/usr/bin/env bash
# Install the SKYFORGE comm bridge on a Raspberry Pi (Raspberry Pi OS / Ubuntu, Python >= 3.10).
# Only the Python standard library is required; psutil (accurate CPU/RAM) and PyYAML (config file)
# are installed when available.
#
#   scp -r DRONATHON3.0 pi@<pi-ip>:~/    then on the Pi:
#   cd ~/DRONATHON3.0 && sudo ./deploy/raspberry_pi/install.sh [--station <laptop-ip>] [--brain <laptop-ip>]
set -euo pipefail

STATION_IP="127.0.0.1"
BRAIN_IP="127.0.0.1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --station) STATION_IP="$2"; shift 2 ;;
    --brain) BRAIN_IP="$2"; shift 2 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
done

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DEST=/opt/skyforge
RUN_USER="${SUDO_USER:-pi}"

echo "[skyforge] installing to $DEST (user $RUN_USER), station=$STATION_IP brain=$BRAIN_IP"
apt-get update -qq || true
apt-get install -y -qq python3 python3-psutil python3-yaml || echo "[skyforge] apt optional packages skipped"

mkdir -p "$DEST" "$DEST/logs"
rm -rf "$DEST/skyforge_core"
cp -r "$REPO/skyforge_ws/src/skyforge_core/skyforge_core" "$DEST/"
rm -rf "$DEST/skyforge_core/sim" "$DEST/skyforge_core/brain" "$DEST/skyforge_core/perception" \
       "$DEST/skyforge_core/planning" "$DEST/skyforge_core/reflex"   # comm-only footprint on the Pi
cp "$REPO/skyforge_ws/src/skyforge_bringup/config/skyforge.yaml" "$DEST/skyforge.yaml"
cat > "$DEST/bridge.env" <<EOF
SKYFORGE_STATION=${STATION_IP}:14601
SKYFORGE_BRAIN_REPLY=${BRAIN_IP}:14701
EOF
chown -R "$RUN_USER" "$DEST"

sed "s/__USER__/$RUN_USER/" "$REPO/deploy/raspberry_pi/skyforge-comm-bridge.service" > /etc/systemd/system/skyforge-comm-bridge.service
systemctl daemon-reload
systemctl enable --now skyforge-comm-bridge.service
sleep 1
systemctl --no-pager status skyforge-comm-bridge.service | head -5
echo "[skyforge] done. Logs: journalctl -u skyforge-comm-bridge -f ; metrics: $DEST/logs/bridge_metrics.jsonl"
echo "[skyforge] benchmark (F12): cd $DEST && python3 -m skyforge_core.comm.pi_bench --duration 60 --outage 30"
