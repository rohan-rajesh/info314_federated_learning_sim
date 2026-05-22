# Federated Learning Protocol Specification

## Overview

**Transport:** TCP, bidirectional async messaging (no request/response pairing)
**Encoding:** UTF-8 JSON objects

Every message includes these common fields: `type` (string, required), `msg_id` (string, required), `round_id` (int, optional — excused for messages not tied to a round), and `sender_id` (string, required).

---

## Message Types

### 1. `CLIENT_READY`
**Client → Coordinator**

Sent by a client at startup to register itself and announce availability for training rounds.

- `client_id` (string, required)
- `dataset_size` (int, required) — number of samples held; used for weighted FedAvg
- `protocol_version` (string, optional)

**Response:** `READY_ACK` from coordinator.

```json
{
  "type": "CLIENT_READY",
  "msg_id": "m-1",
  "sender_id": "client_a",
  "client_id": "client_a",
  "dataset_size": 4200,
  "protocol_version": "1.0"
}
```

---

### 2. `READY_ACK`
**Coordinator → Client**

Acknowledges a client's registration and provides the parameter server address to connect to.

- `client_id` (string, required) — echoes the registering client
- `status` (string, required) — `"accepted"` or `"rejected"`
- `parameter_server_addr` (string, required if accepted) — e.g. `"10.0.0.5:9100"`
- `reason` (string, optional) — only if rejected

**Response:** None. Client waits for `START_ROUND`.

```json
{
  "type": "READY_ACK",
  "msg_id": "m-002",
  "sender_id": "coordinator",
  "client_id": "client_a",
  "status": "accepted",
  "parameter_server_addr": "10.0.0.5:9100"
}
```

---

### 3. `START_ROUND`
**Coordinator → Client**

Notifies a selected client that a new training round has begun and that it should fetch the latest global model.

- `round_id` (int, required)
- `deadline_ms` (int, required) — milliseconds the client has to submit its update
- `local_epochs` (int, optional) — training hyperparameter for this round
- `learning_rate` (float, optional)

**Response:** None directly. Client will receive a `GLOBAL_MODEL` from the parameter server.

```json
{
  "type": "START_ROUND",
  "msg_id": "m-003",
  "sender_id": "coordinator",
  "round_id": 7,
  "deadline_ms": 30000,
  "local_epochs": 3,
  "learning_rate": 0.01
}
```

---

### 4. `GLOBAL_MODEL`
**Parameter Server → Client**

Sends the current global model weights to a client at the start of a round so it can initialize local training.

- `round_id` (int, required)
- `model_version` (int, required) — increments per aggregation; lets clients verify they have the right base
- `weights` (object, required) — maps layer names to arrays of floats

**Response:** Eventually a `WEIGHT_UPDATE` from the client. Client overwrites its local model with these weights and begins training.

```json
{
  "type": "GLOBAL_MODEL",
  "msg_id": "m-004",
  "sender_id": "ps",
  "round_id": 7,
  "model_version": 6,
  "weights": {
    "layer1.w": [0.12, -0.04, "..."],
    "layer2.b": [0.0, "..."]
  }
}
```

---

### 5. `WEIGHT_UPDATE`
**Client → Parameter Server**

Sends the locally-trained weight delta back to the parameter server after finishing local epochs.

- `round_id` (int, required)
- `client_id` (string, required)
- `weight_delta` (object, required) — same shape as `weights` in `GLOBAL_MODEL`
- `dataset_size` (int, required) — used as the FedAvg weighting factor
- `local_loss` (float, optional) — for monitoring/logging

**Response:** None directly. Aggregation is triggered later by `PROCEED`.

```json
{
  "type": "WEIGHT_UPDATE",
  "msg_id": "m-005",
  "sender_id": "client_a",
  "round_id": 7,
  "client_id": "client_a",
  "weight_delta": {
    "layer1.w": [0.001, -0.003, "..."]
  },
  "dataset_size": 4200,
  "local_loss": 0.342
}
```

---

### 6. `PROCEED`
**Coordinator → Parameter Server**

Signals that the round deadline has passed and aggregation should run with whatever updates have arrived.

- `round_id` (int, required)
- `participating_clients` (string[], required) — clients whose updates should be included
- `skipped_clients` (string[], optional) — clients that missed the deadline (informational)

**Response:** `ROUND_COMPLETE` from the parameter server once aggregation finishes.

```json
{
  "type": "PROCEED",
  "msg_id": "m-006",
  "sender_id": "coordinator",
  "round_id": 7,
  "participating_clients": ["client_a", "client_b"],
  "skipped_clients": ["client_c"]
}
```

---

### 7. `ROUND_COMPLETE`
**Parameter Server → Coordinator**

Reports that aggregation has finished, with summary stats for the round.

- `round_id` (int, required)
- `new_model_version` (int, required)
- `clients_used` (int, required)
- `global_loss` (float, optional)

**Response:** None. Coordinator decides whether to start the next round.

```json
{
  "type": "ROUND_COMPLETE",
  "msg_id": "m-007",
  "sender_id": "ps",
  "round_id": 7,
  "new_model_version": 7,
  "clients_used": 2,
  "global_loss": 0.298
}
```

---

### 8. `ERROR`
**Any → Any**

Generic error reply for messages that were malformed, unauthorized, out of order, or otherwise unprocessable.

- `error_code` (string, required) — e.g. `"BAD_ROUND_ID"`, `"UNKNOWN_CLIENT"`, `"MALFORMED"`, `"STALE_UPDATE"`
- `error_message` (string, required) — human-readable explanation
- `in_reply_to` (string, optional) — `msg_id` of the offending message
- `round_id` (int, optional)

**Response:** None. Receiver logs the error; sender may retry or abort depending on `error_code`.

```json
{
  "type": "ERROR",
  "msg_id": "m-009",
  "sender_id": "ps",
  "error_code": "STALE_UPDATE",
  "error_message": "Update for round 6 received during round 7",
  "in_reply_to": "m-005",
  "round_id": 6
}
```

---

### 9. `GOODBYE`
**Any → Any**

Clean disconnect signal. Receiver should not flag this as a crash.

- `reason` (string, optional) — e.g. `"shutdown"`, `"training_complete"`

**Response:** None. Receiver closes its side of the connection.

```json
{
  "type": "GOODBYE",
  "msg_id": "m-010",
  "sender_id": "client_a",
  "reason": "shutdown"
}
```
