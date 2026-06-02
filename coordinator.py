# resources for retrying logic and patterns:
# https://krython.com/tutorial/python/connection-retry-logic-handling-failures/

# explains how to reuse the same address in python sockets: 
# https://gist.github.com/webgtx/a24722d5cbb8849351591b6c006b76a5.

import socket
import threading
import time
import argparse
from protocol import send_msg, recv_msgs, PROTOCOL_MSGS

COORD_ID = "coordinator"

def run(host, port, ps_host, ps_client_port, ps_coord_port, min_clients, num_rounds, deadline_sec, max_missed_rounds):
    clients = {}
    lock = threading.Lock()
    ready_event = threading.Event()

    # register client and reply with the PS address        
    def handle_client(sock):
        for msg in recv_msgs(sock):
            if msg["type"] == PROTOCOL_MSGS["CLIENT_READY"]:
                cid = msg["client_id"]
                ps_addr = f"{ps_host}:{ps_client_port}"
                send_msg(sock, COORD_ID, PROTOCOL_MSGS["READY_ACK"],
                         client_id=cid, status="accepted",
                         parameter_server_addr=ps_addr)

                # unblock main thread when enough clients are registered                         
                with lock:
                    clients[cid] = {"sock": sock, "dataset_size": msg["dataset_size"]}
                    n = len(clients)
                print(f"[coordinator] registered {cid} ({msg['dataset_size']} samples) [{n}/{min_clients}]")
                with lock:
                    if len(clients) >= min_clients:
                        ready_event.set()

    # connect to PS coord port
    ps_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(10):
        try:
            ps_sock.connect((ps_host, ps_coord_port))
            break
        except ConnectionRefusedError:
            if attempt == 9:
                raise
            time.sleep(0.5)
    print(f"[coordinator] connected to PS at {ps_host}:{ps_coord_port}")
    ps_msgs = recv_msgs(ps_sock)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(20)
    print(f"[coordinator] listening on {host}:{port}")
    def accept_loop():
        while True:
            sock, _ = srv.accept()
            threading.Thread(target=handle_client, args=(sock,), daemon=True).start()
    threading.Thread(target=accept_loop, daemon=True).start()
    print(f"[coordinator] waiting for {min_clients} clients...")
    ready_event.wait()
    with lock:
        selected = list(clients.keys())
    print(f"[coordinator] all clients ready: {selected}")

    # track consecutive missed rounds per client; once a client crosses the
    # threshold it's added to `inactive` and we stop including it in rounds
    misses = {cid: 0 for cid in selected}
    inactive = set()

    for round_id in range(1, num_rounds + 1):
        active = [c for c in selected if c not in inactive]
        if not active:
            print(f"[coordinator] no active clients left, stopping at round {round_id}")
            break
        print(f"[coordinator] starting round {round_id} with {active}")
        # tell server to open the round and push GLOBAL_MODEL to clients first
        send_msg(ps_sock, COORD_ID, PROTOCOL_MSGS["ROUND_START"],
                 round_id=round_id, participating_clients=active)
        for cid in active:
            try:
                send_msg(clients[cid]["sock"], COORD_ID, PROTOCOL_MSGS["START_ROUND"],
                         round_id=round_id, deadline_ms=int(deadline_sec * 1000),
                         local_epochs=5, learning_rate=0.1)
            except (BrokenPipeError, ConnectionResetError, OSError) as e:
                # client socket is dead; mark inactive right away
                print(f"[coordinator] send to {cid} failed ({e}), marking INACTIVE")
                inactive.add(cid)
        # wait for the deadline to pass, then close the round at parameter server
        time.sleep(deadline_sec)
        send_msg(ps_sock, COORD_ID, PROTOCOL_MSGS["PROCEED"],
                 round_id=round_id, participating_clients=active)
        for msg in ps_msgs:
            if msg["type"] == PROTOCOL_MSGS["ROUND_COMPLETE"]:
                participated = msg.get("participating_clients", [])
                skipped = [c for c in active if c not in participated]
                # reset miss counts for clients that showed up; bump it for the rest
                for cid in active:
                    if cid in participated:
                        misses[cid] = 0
                    else:
                        misses[cid] += 1
                        if misses[cid] >= max_missed_rounds and cid not in inactive:
                            inactive.add(cid)
                            print(f"[coordinator] marking {cid} INACTIVE "
                                  f"({misses[cid]} consecutive misses)")
                print(f"[coordinator] round {round_id} done: "
                      f"model_v{msg['new_model_version']}, "
                      f"participated={participated}, skipped={skipped}")
                break
    for cid in selected:
        if cid in inactive:
            continue
        try:
            send_msg(clients[cid]["sock"], COORD_ID, PROTOCOL_MSGS["GOODBYE"], reason="training_complete")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
    send_msg(ps_sock, COORD_ID, PROTOCOL_MSGS["GOODBYE"], reason="training_complete")
    print("[coordinator] training complete")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--ps-host", default="127.0.0.1")
    ap.add_argument("--ps-client-port", type=int, default=9100)
    ap.add_argument("--ps-coord-port", type=int, default=9101)
    ap.add_argument("--min-clients", type=int, default=3)
    ap.add_argument("--num-rounds", type=int, default=10)
    ap.add_argument("--deadline", type=float, default=5.0,
                    help="seconds to wait before triggering aggregation")
    ap.add_argument("--max-missed-rounds", type=int, default=3,
                    help="mark a client inactive after this many consecutive missed rounds")
    args = ap.parse_args()
    run(args.host, args.port, args.ps_host, args.ps_client_port, args.ps_coord_port,
        args.min_clients, args.num_rounds, args.deadline, args.max_missed_rounds)