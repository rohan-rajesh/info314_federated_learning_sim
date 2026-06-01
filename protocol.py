import json
import uuid

# all the message types used in our protocol
PROTOCOL_MSGS = {
    "CLIENT_READY": "CLIENT_READY",
    "READY_ACK": "READY_ACK",
    "START_ROUND": "START_ROUND",
    "GLOBAL_MODEL": "GLOBAL_MODEL",
    "WEIGHT_UPDATE": "WEIGHT_UPDATE",
    "PROCEED": "PROCEED",
    "ROUND_COMPLETE": "ROUND_COMPLETE",
    "ERROR": "ERROR",
    "GOODBYE": "GOODBYE",
}


def msg_id():
    # generate a short unique id for each message
    unique = uuid.uuid4().hex[:8]
    return "m-" + unique


def send_msg(sock, sender_id, msg_type, **fields):
    # build the base message with required fields
    msg = {}
    msg["type"] = msg_type
    msg["msg_id"] = msg_id()
    msg["sender_id"] = sender_id

    # add any extra fields the caller passed in
    for key in fields:
        msg[key] = fields[key]

    # convert to JSON, add a newline, and send over the socket
    encoded = json.dumps(msg) + "\n"
    sock.sendall(encoded.encode("utf-8"))
    return msg


def recv_msgs(sock):
    # read incoming data and yield one message at a time
    buffer = ""

    while True:
        data = sock.recv(4096)

        # empty data means the connection closed
        if not data:
            break

        buffer = buffer + data.decode("utf-8")

        # process every complete line we have so far
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_addr(addr):
    # split "host:port" into ("host", port_number)
    host, port = addr.rsplit(":", 1)
    return host, int(port)
