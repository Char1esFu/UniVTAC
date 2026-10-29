#!/usr/bin/env bash
# Forward the cluster inference server to 127.0.0.1:5001 on this workstation.
#
# The compute node has no public address, but the login node reaches it on the
# internal network and ssh already jumps there (~/.ssh/config: alex via csnhr).
# So a plain local forward is enough -- no proxy, no reverse tunnel.
#
# Usage:  bash tools/tunnel.sh            # auto-detect the running job's node
#         bash tools/tunnel.sh a0633      # or name it
set -euo pipefail

HOST="${DREAMTAC_SSH_HOST:-alex}"
PORT="${DREAMTAC_PORT:-5001}"
NODE="${1:-}"

if [ -z "$NODE" ]; then
    echo "looking up your running job on $HOST..."
    NODE=$(ssh "$HOST" "squeue --me -h -t RUNNING -o '%N' | head -1" | tr -d '[:space:]')
    [ -n "$NODE" ] || { echo "no running job found; sbatch the server first" >&2; exit 1; }
fi

echo "forwarding 127.0.0.1:$PORT -> $NODE:$PORT via $HOST  (Ctrl-C to stop)"
echo "check it with: curl -s http://127.0.0.1:$PORT/health"
exec ssh -N -L "${PORT}:${NODE}:${PORT}" "$HOST"
