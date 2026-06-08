import subprocess
import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def start(args):
    return subprocess.Popen([PYTHON] + args, cwd=SCRIPT_DIR)

def main():
    procs = []

    print("[demo] Starting parameter server")
    ps = start(["parameter_server.py"])
    procs.append(ps)
    time.sleep(0.4)

    # short deadline + low miss threshold so the demo finishes quickly
    print("=== [demo] Starting coordinator "
          "(3 clients, 10 rounds, 3s deadline, max 2 missed) ===")
    coord = start([
        "coordinator.py",
        "--min-clients", "3",
        "--num-rounds", "10",
        "--deadline", "3",
        "--max-missed-rounds", "2",
    ])
    procs.append(coord)
    time.sleep(0.4)

    print("[demo] Starting clients")
    procs.append(start(["client.py", "client_a", "--dataset-size", "300"]))
    procs.append(start(["client.py", "client_b", "--dataset-size", "500"]))
    client_c = start(["client.py", "client_c", "--dataset-size", "200"])
    procs.append(client_c)

    # kill client_c after ~2 rounds (3s deadline * 2 + startup buffer)
    kill_after = 10
    print(f"\n[demo] Running for {kill_after}s, "
          f"then killing client_c to simulate a crash ===\n")
    time.sleep(kill_after)

    print("\n[demo] KILLING client_c — "
          "training should continue and eventually mark it INACTIVE <<<\n")
    client_c.terminate()

    try:
        coord.wait(timeout=90)
        print("[demo] Training complete")
    except subprocess.TimeoutExpired:
        print("[demo] Coordinator timed out — something may have stalled")
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

if __name__ == "__main__":
    main()
