import socket
import threading
import queue
import time
import argparse
from protocol import send_msg, recv_msgs, parse_addr, PROTOCOL_MSGS
from model import make_dataset, local_train, subtract, mse_loss

def run(client_id, dataset_size, coord_addr):
    coord_host, coord_port = parse_addr(coord_addr)

    # connect to coordinator 
    coord_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for attempt in range(10):
        try:
            coord_sock.connect((coord_host, coord_port))
            break
        except ConnectionRefusedError:
            if attempt == 9:
                raise
            time.sleep(0.5)

    send_msg(coord_sock, client_id, PROTOCOL_MSGS["CLIENT_READY"],
             client_id=client_id, dataset_size=dataset_size, protocol_version="1.0")
    print(f"[{client_id}] sent CLIENT_READY ({dataset_size} samples)")

    # single generator for the coord socket
    coord_msgs = recv_msgs(coord_sock)

    # extract parameter server address from response
    ps_addr = None
    for msg in coord_msgs:
        if msg["type"] == PROTOCOL_MSGS["READY_ACK"]:
            if msg["status"] != "accepted":
                print(f"[{client_id}] rejected: {msg.get('reason', '?')}")
                return
            ps_addr = msg["parameter_server_addr"]
            print(f"[{client_id}] accepted, PS at {ps_addr}")
            break

    ps_host, ps_port = parse_addr(ps_addr)
    ps_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ps_sock.connect((ps_host, ps_port))

    # generate local training data seeded by client id
    X, y = make_dataset(client_id, dataset_size)
    # reader threads feed data to queues 
    coord_q = queue.Queue()
    ps_q = queue.Queue()
    # hand off the existing generator to the reader thread
    def read_coord():
        for msg in coord_msgs:
            coord_q.put(msg)

    def read_ps():
        for msg in recv_msgs(ps_sock):
            ps_q.put(msg)

    threading.Thread(target=read_coord, daemon=True).start()
    threading.Thread(target=read_ps, daemon=True).start()

    while True:
        msg = coord_q.get()
        if msg["type"] == PROTOCOL_MSGS["GOODBYE"]:
            print(f"[{client_id}] training complete")
            break
        if msg["type"] != PROTOCOL_MSGS["START_ROUND"]:
            continue

        round_id = msg["round_id"]
        epochs = msg.get("local_epochs", 5)
        lr = msg.get("learning_rate", 0.1)
        print(f"[{client_id}] round {round_id} started")

        # wait for the matching global model for this round
        while True:
            msg = ps_q.get()
            if msg["type"] == PROTOCOL_MSGS["GLOBAL_MODEL"] and msg["round_id"] == round_id:
                weights = msg["weights"]
                print(f"[{client_id}] received model v{msg['model_version']}")
                break

        # train locally and compute the weight delta against the global model
        trained = local_train(weights, X, y, epochs=epochs, lr=lr)
        delta = subtract(trained, weights)
        loss = mse_loss(trained, X, y)

        # send weight delta back to parameter server
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
