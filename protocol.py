import json
import uuid

PROTOCOL_MSGS = {
    "CLIENT_READY": "CLIENT_READY",
    "READY_ACK": "READY_ACK",
    "ROUND_START": "ROUND_START",
    "START_ROUND": "START_ROUND",
    "GLOBAL_MODEL": "GLOBAL_MODEL",
    "WEIGHT_UPDATE": "WEIGHT_UPDATE",
    "PROCEED": "PROCEED",
    "ROUND_COMPLETE": "ROUND_COMPLETE",
    "ERROR": "ERROR",
    "GOODBYE": "GOODBYE",
}

def msg_id():
    # unique id generation
    unique = uuid.uuid4().hex[:8]
    return "m-" + unique

# base message w/ required fields
def send_msg(sock, sender_id, msg_type, **fields):
    msg = {}
    msg["type"] = msg_type
    msg["msg_id"] = msg_id()
    msg["sender_id"] = sender_id
    for key in fields:
        msg[key] = fields[key]

    # convert to JSON -> socket sending
    encoded = json.dumps(msg) + "\n"
    sock.sendall(encoded.encode("utf-8"))
    return msg

# message receiving
def recv_msgs(sock):
    buffer = ""
    while True:
        data = sock.recv(4096)
        if not data:
            break
        buffer = buffer + data.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield json.loads(line)

# address format parsing
def parse_addr(addr):
    host, port = addr.rsplit(":", 1)
    return host, int(port)
