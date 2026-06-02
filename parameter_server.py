# following resources were used to developing server logic & network communication:
# https://docs.python.org/3/howto/sockets.html
# https://docs.python.org/3/library/socket.html
# https://oneuptime.com/blog/post/2026-03-20-socket-errors-exceptions-python-ipv4/view

import socket
import threading
import argparse
from protocol import send_msg, recv_msgs, PROTOCOL_MSGS
from model import init_weights, fedavg

PS_ID = "ps"
# parameter server logic
def run(host, client_port, coord_port):
    state = {"weights": init_weights(), "version": 0}
    client_socks = []
    socks_lock = threading.Lock()
    updates = {} # client updates
    updates_lock = threading.Lock()
    expected = [] # expected client ids
    done_event = threading.Event()

    # collect weight update from one client; signal when all expected updates are in
    def handle_client(sock):
        for msg in recv_msgs(sock):
            if msg["type"] == PROTOCOL_MSGS["WEIGHT_UPDATE"]:
                cid = msg["client_id"]
                loss = msg.get("local_loss")
                loss_str = f"{loss:.4f}" if loss is not None else "?"
                print(f"[ps] update from {cid} round={msg['round_id']} loss={loss_str}")
                with updates_lock:
                    updates[cid] = (msg["weight_delta"], msg["dataset_size"])
                    ready = set(updates.keys()) >= set(expected)
                if ready:
                    done_event.set()
            elif msg["type"] == PROTOCOL_MSGS["GOODBYE"]:
                break

    # accept client connections
    def accept_clients():
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, client_port))
        srv.listen(20)
        print(f"[ps] client port {host}:{client_port}")
        while True:
            sock, _ = srv.accept()
            with socks_lock:
                client_socks.append(sock)
            threading.Thread(target=handle_client, args=(sock,), daemon=True).start()
    threading.Thread(target=accept_clients, daemon=True).start()

    # coordinator listener: single connection, blocks until coordinator connects
    coord_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    coord_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    coord_srv.bind((host, coord_port))
    coord_srv.listen(1)
    print(f"[ps] coord port {host}:{coord_port}")
    coord_sock, _ = coord_srv.accept()
    print("[ps] coordinator connected")

    for msg in recv_msgs(coord_sock):
        if msg["type"] == PROTOCOL_MSGS["PROCEED"]:
            round_id = msg["round_id"]
            participating = msg["participating_clients"]
            print(f"[ps] round {round_id} — clients: {participating}")
            # reset per-round state before broadcasting
            expected.clear()
            expected.extend(participating)
            updates.clear()
            done_event.clear()
            # push current global model to all connected clients
            with socks_lock:
                snapped = list(client_socks)
            for sock in snapped:
                send_msg(sock, PS_ID, PROTOCOL_MSGS["GLOBAL_MODEL"],
                         round_id=round_id, model_version=state["version"],
                         weights=state["weights"])
            # wait for all clients to submit updates
            done_event.wait()
            with updates_lock:
                batch = list(updates.values())
            # aggregate updates & bump model version
            state["weights"] = fedavg(state["weights"], batch)
            state["version"] += 1
            print(f"[ps] aggregated → model v{state['version']}")
            # notify coordinator when aggregation is done
            send_msg(coord_sock, PS_ID, PROTOCOL_MSGS["ROUND_COMPLETE"],
                     round_id=round_id, new_model_version=state["version"],
                     clients_used=len(batch))
        # coordinator is shutting down
        elif msg["type"] == PROTOCOL_MSGS["GOODBYE"]:
            print("[ps] coordinator goodbye, shutting down")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=9100)
    ap.add_argument("--coord-port", type=int, default=9101)
    args = ap.parse_args()
    run(args.host, args.client_port, args.coord_port)
