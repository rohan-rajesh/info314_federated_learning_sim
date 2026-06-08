# following resources were used to developing server logic & network communication:
# https://docs.python.org/3/howto/sockets.html
# https://docs.python.org/3/library/socket.html
# https://oneuptime.com/blog/post/2026-03-20-socket-errors-exceptions-python-ipv4/view

import socket
import threading
import argparse
from protocol import send_msg, recv_msgs, PROTOCOL_MSGS
from model import init_weights, fedavg, make_dataset, mse_loss

PS_ID = "ps"

# fixed validation set size used to track global loss across rounds
VAL_SIZE = 200

def run(host, client_port, coord_port):
    state = {"weights": init_weights(), "version": 0, "round": 0}
    client_socks = []
    socks_lock = threading.Lock()
    updates = {}  # client_id -> (delta, dataset_size), cleared each round
    updates_lock = threading.Lock()

    # held-out validation set for global loss tracking (never used for training)
    val_X, val_y = make_dataset("ps_validation", VAL_SIZE)

    def send_global_model(sock):
        send_msg(sock, PS_ID, PROTOCOL_MSGS["GLOBAL_MODEL"],
                 round_id=state["round"], model_version=state["version"],
                 weights=state["weights"])

    # one thread per client; routes WEIGHT_UPDATE into the updates dict
    def handle_client(sock):
        for msg in recv_msgs(sock):
            if msg["type"] == PROTOCOL_MSGS["GOODBYE"]:
                break
            if msg["type"] != PROTOCOL_MSGS["WEIGHT_UPDATE"]:
                continue
            cid = msg["client_id"]
            rid = msg["round_id"]
            # drop updates from a closed round
            if rid != state["round"]:
                print(f"[ps] stale update from {cid} for round {rid} (current={state['round']})")
                send_msg(sock, PS_ID, PROTOCOL_MSGS["ERROR"],
                         error_code="STALE_UPDATE",
                         error_message=f"round {rid} is closed",
                         in_reply_to=msg.get("msg_id"), round_id=rid)
                continue
            with updates_lock:
                if cid in updates:
                    print(f"[ps] duplicate update from {cid}, ignoring")
                    continue
                updates[cid] = (msg["weight_delta"], msg["dataset_size"])
            loss = msg.get("local_loss")
            loss_str = f"{loss:.4f}" if loss is not None else "n/a"
            print(f"[ps] update from {cid} round={rid} loss={loss_str}")

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
            # push current model if a client connects mid-round
            if state["round"] > 0:
                send_global_model(sock)
            threading.Thread(target=handle_client, args=(sock,), daemon=True).start()
    threading.Thread(target=accept_clients, daemon=True).start()

    coord_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    coord_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    coord_srv.bind((host, coord_port))
    coord_srv.listen(1)
    print(f"[ps] coord port {host}:{coord_port}")

    # outer loop supports coordinator crash-and-reconnect: after an unexpected
    # disconnect the PS keeps its model state and waits for a new coordinator
    while True:
        coord_sock, _ = coord_srv.accept()
        print("[ps] coordinator connected")

        for msg in recv_msgs(coord_sock):
            mtype = msg["type"]
            if mtype == PROTOCOL_MSGS["ROUND_START"]:
                # open new round: clear updates, bump round, broadcast current model
                round_id = msg["round_id"]
                print(f"[ps] ROUND_START {round_id} — clients: {msg['participating_clients']}")
                with updates_lock:
                    updates.clear()
                state["round"] = round_id
                with socks_lock:
                    snapped = list(client_socks)
                for sock in snapped:
                    try:
                        send_global_model(sock)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        with socks_lock:
                            if sock in client_socks:
                                client_socks.remove(sock)

            elif mtype == PROTOCOL_MSGS["PROCEED"]:
                # deadline passed: aggregate and report back
                round_id = msg["round_id"]
                if round_id != state["round"]:
                    print(f"[ps] PROCEED for round {round_id} but current is {state['round']}, ignoring")
                    continue
                with updates_lock:
                    batch = list(updates.values())
                    participants = list(updates.keys())
                if batch:
                    state["weights"] = fedavg(state["weights"], batch)
                    state["version"] += 1
                    global_loss = mse_loss(state["weights"], val_X, val_y)
                    print(f"[ps] aggregated {len(batch)} updates → "
                          f"model v{state['version']} global_loss={global_loss:.4f}")
                else:
                    global_loss = None
                    print(f"[ps] round {round_id} had zero updates, "
                          f"keeping model v{state['version']}")
                send_msg(coord_sock, PS_ID, PROTOCOL_MSGS["ROUND_COMPLETE"],
                         round_id=round_id, new_model_version=state["version"],
                         clients_used=len(batch), participating_clients=participants,
                         global_loss=global_loss)

            elif mtype == PROTOCOL_MSGS["GOODBYE"]:
                print("[ps] coordinator goodbye, shutting down")
                break

        else:
            # for/else: loop exhausted without break = coordinator dropped unexpectedly
            print("[ps] coordinator disconnected unexpectedly, waiting for reconnect...")
            continue

        # reached via break on GOODBYE = clean shutdown
        break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--client-port", type=int, default=9100)
    ap.add_argument("--coord-port", type=int, default=9101)
    args = ap.parse_args()
    run(args.host, args.client_port, args.coord_port)
