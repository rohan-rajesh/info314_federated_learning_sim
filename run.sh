#!/usr/bin/env bash
# Normal demo: 1 parameter server + 1 coordinator + 3 clients, 5 rounds.
# Usage: bash run.sh
cd "$(dirname "$0")"

cleanup() {
    echo "=== Shutting down background processes ==="
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Starting parameter server ==="
python3 parameter_server.py &

sleep 0.4

echo "=== Starting coordinator (min 3 clients, 5 rounds, 5s deadline) ==="
python3 coordinator.py --min-clients 3 --num-rounds 5 &
COORD_PID=$!

sleep 0.4

echo "=== Starting 3 clients ==="
python3 client.py client_a --dataset-size 300 &
python3 client.py client_b --dataset-size 500 &
python3 client.py client_c --dataset-size 200 &

echo "=== Waiting for training to complete ==="
wait $COORD_PID
echo "=== Training complete ==="
