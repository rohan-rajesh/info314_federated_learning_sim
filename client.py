import socket
import threading
import queue
import time
import argparse
from protocol import send_msg, recv_msgs, parse_addr, PROTOCOL_MSGS
from model import make_dataset, local_train, subtract, mse_loss


def _register(client_id, dataset_size, coord_host, coord_port):
    """Connect to coordinator, send CLIENT_READY, return (sock, msg_generator, ps_addr)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(20):
        try:
            sock.connect((coord_host, coord_port))
            break
        except ConnectionRefusedError:
            if attempt == 19:
                raise
            time.sleep(1.0)
    send_msg(sock, client_id, PROTOCOL_MSGS["CLIENT_READY"],
             client_id=client_id, dataset_size=dataset_size, protocol_version="1.0")
    print(f"[{client_id}] sent CLIENT_READY ({dataset_size} samples)")
    gen = recv_msgs(sock)
    for msg in gen:
        if msg["type"] == PROTOCOL_MSGS["READY_ACK"]:
            if msg["status"] != "accepted":
                raise RuntimeError(f"rejected by coordinator: {msg.get('reason', '?')}")
            print(f"[{client_id}] accepted, PS at {msg['parameter_server_addr']}")
            return sock, gen, msg["parameter_server_addr"]
    raise RuntimeError("coordinator closed before READY_ACK")


def run(client_id, dataset_size, coord_addr):
    coord_host, coord_port = parse_addr(coord_addr)
    X, y = make_dataset(client_id, dataset_size)

    coord_sock, coord_gen, ps_addr = _register(client_id, dataset_size, coord_host, coord_port)
    ps_host, ps_port = parse_addr(ps_addr)
    ps_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ps_sock.connect((ps_host, ps_port))

    coord_q = queue.Queue()
    ps_q = queue.Queue()

    def read_coord(gen):
        # put None sentinel when the connection closes so the main loop can detect it
        try:
            for msg in gen:
                coord_q.put(msg)
        except OSError:
            pass
        finally:
            coord_q.put(None)

    def read_ps(sock):
        try:
            for msg in recv_msgs(sock):
                ps_q.put(msg)
        except OSError:
            pass

    threading.Thread(target=read_coord, args=(coord_gen,), daemon=True).start()
    threading.Thread(target=read_ps, args=(ps_sock,), daemon=True).start()

    def reconnect(reason):
        """Re-register with coordinator; reconnect to PS if the address changed.
        Returns True on success, False after exhausting retries."""
        nonlocal coord_sock, coord_gen, ps_sock, ps_host, ps_port
        print(f"[{client_id}] reconnecting (reason={reason!r})...")
        for attempt in range(20):
            try:
                coord_sock, coord_gen, new_ps_addr = _register(
                    client_id, dataset_size, coord_host, coord_port)
                new_ps_host, new_ps_port = parse_addr(new_ps_addr)
                if (new_ps_host, new_ps_port) != (ps_host, ps_port):
                    ps_sock.close()
                    ps_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    ps_sock.connect((new_ps_host, new_ps_port))
                    ps_host, ps_port = new_ps_host, new_ps_port
                    threading.Thread(target=read_ps, args=(ps_sock,), daemon=True).start()
                threading.Thread(target=read_coord, args=(coord_gen,), daemon=True).start()
                return True
            except Exception as e:
                print(f"[{client_id}] reconnect attempt {attempt + 1} failed: {e}")
                if attempt == 19:
                    return False
                time.sleep(2.0)
        return False

    while True:
        msg = coord_q.get()

        if msg is None:
            # coordinator connection dropped unexpectedly
            print(f"[{client_id}] coordinator connection lost")
            if not reconnect("connection_lost"):
                print(f"[{client_id}] giving up after failed reconnects")
                break
            continue

        if msg["type"] == PROTOCOL_MSGS["GOODBYE"]:
            reason = msg.get("reason", "")
            if reason == "training_complete":
                print(f"[{client_id}] training complete")
                break
            # coordinator marked this client inactive; pause then re-register
            print(f"[{client_id}] got GOODBYE (reason={reason!r}), will re-register...")
            time.sleep(1.0)
            if not reconnect(reason):
                print(f"[{client_id}] giving up after failed re-registration")
                break
            continue

        if msg["type"] != PROTOCOL_MSGS["START_ROUND"]:
            continue

        round_id = msg["round_id"]
        epochs = msg.get("local_epochs", 5)
        lr = msg.get("learning_rate", 0.1)
        print(f"[{client_id}] round {round_id} started")

        # wait for the matching global model for this round
        while True:
            ps_msg = ps_q.get()
            if ps_msg["type"] == PROTOCOL_MSGS["GLOBAL_MODEL"] and ps_msg["round_id"] == round_id:
                weights = ps_msg["weights"]
                print(f"[{client_id}] received model v{ps_msg['model_version']}")
                break

        trained = local_train(weights, X, y, epochs=epochs, lr=lr)
        delta = subtract(trained, weights)
        loss = mse_loss(trained, X, y)

        send_msg(ps_sock, client_id, PROTOCOL_MSGS["WEIGHT_UPDATE"],
                 round_id=round_id, client_id=client_id,
                 weight_delta=delta, dataset_size=dataset_size, local_loss=loss)
        print(f"[{client_id}] sent update round={round_id} loss={loss:.4f}")

    coord_sock.close()
    ps_sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("client_id")
    ap.add_argument("--dataset-size", type=int, default=500)
    ap.add_argument("--coord", default="127.0.0.1:9000")
    args = ap.parse_args()
    run(args.client_id, args.dataset_size, args.coord)
