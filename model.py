# following resources were used to develop this model logic and code:
# https://medium.com/mlcodex/how-to-train-your-model-faster-and-better-with-batch-gradient-descent-d9cd7f766e92
# http://medium.com/data-science/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399
# https://docs.python.org/3/library/random.html
# https://towardsdatascience.com/implementing-linear-regression-with-gradient-descent-from-scratch-f6d088ec1219/

import random

# our model is y = w*x + b (linear regression with 4 input features)
NUM_FEATURES = 4

# the "true" weights each client's data is secretly generated from
TRUE_W = [1.5, -2.0, 0.5, 3.0]
TRUE_B = 0.7


def make_dataset(client_id, n_samples):
    # give each client its own random data based on its id
    rng = random.Random(hash(client_id) & 0xFFFF)

    X = []  # list of input rows
    y = []  # list of target values

    for _ in range(n_samples):
        # build one input row with random values between -1 and 1
        row = []
        for _ in range(NUM_FEATURES):
            row.append(rng.uniform(-1, 1))

        # compute the target using the true weights
        target = 0.0
        for w, f in zip(TRUE_W, row):
            target = target + w * f
        target = target + TRUE_B

        # add a tiny bit of noise so the data isn't perfect
        target = target + rng.gauss(0, 0.05)

        X.append(row)
        y.append(target)

    return X, y


def init_weights():
    # start with all zeros (w is a list, b is a one-element list)
    w = [0.0] * NUM_FEATURES
    b = [0.0]
    return {"w": w, "b": b}


def predict(weights, row):
    # dot product of weights and features, then add bias
    total = 0.0
    for w, f in zip(weights["w"], row):
        total = total + w * f
    total = total + weights["b"][0]
    return total


def mse_loss(weights, X, y):
    # average squared error across all samples
    total = 0.0
    for row, target in zip(X, y):
        error = predict(weights, row) - target
        total = total + error ** 2
    return total / len(X)


def local_train(weights, X, y, epochs, lr):
    # batch gradient descent on mean squared error
    w = list(weights["w"])
    b = weights["b"][0]
    n = len(X)

    for _ in range(epochs):
        # start gradients at zero each epoch
        grad_w = [0.0] * NUM_FEATURES
        grad_b = 0.0

        for row, target in zip(X, y):
            # compute prediction error for this sample
            prediction = 0.0
            for wi, f in zip(w, row):
                prediction = prediction + wi * f
            prediction = prediction + b
            err = prediction - target

            # accumulate weight gradients
            for j in range(NUM_FEATURES):
                grad_w[j] = grad_w[j] + 2 * err * row[j] / n

            # accumulate bias gradient
            grad_b = grad_b + 2 * err / n

        # update weights using the gradients
        new_w = []
        for wi, g in zip(w, grad_w):
            new_w.append(wi - lr * g)
        w = new_w
        b = b - lr * grad_b

    return {"w": w, "b": [b]}


def subtract(new, base):
    # compute how much the weights changed after local training
    result = {}
    for layer in new:
        diff = []
        for a, b in zip(new[layer], base[layer]):
            diff.append(a - b)
        result[layer] = diff
    return result


def fedavg(base_weights, updates):
    # updates is a list of (weight_delta, dataset_size) pairs
    # we average the deltas, weighted by how much data each client had

    # count total samples across all clients
    total = 0
    for delta, size in updates:
        total = total + size

    # start the result from the current global weights
    result = {}
    for layer in base_weights:
        copied = []
        for val in base_weights[layer]:
            copied.append(val)
        result[layer] = copied

    # add each client's weighted delta to the result
    for delta, size in updates:
        frac = size / total
        for layer in result:
            new_vals = []
            for cur, d in zip(result[layer], delta[layer]):
                new_val = cur + frac * d
                new_vals.append(new_val)
            result[layer] = new_vals

    return result


if __name__ == "__main__":
    # three clients with different dataset sizes
    clients = [("client_a", 300), ("client_b", 500), ("client_c", 200)]

    # build a dataset for each client
    shards = {}
    for cid, n in clients:
        shards[cid] = make_dataset(cid, n)

    global_weights = init_weights()

    for round_id in range(1, 11):
        updates = []

        # each client trains locally and sends back a delta
        for cid, _ in clients:
            X, y = shards[cid]
            trained = local_train(global_weights, X, y, epochs=5, lr=0.1)
            delta = subtract(trained, global_weights)
            updates.append((delta, len(X)))

        # combine all deltas into a new global model
        global_weights = fedavg(global_weights, updates)

        # pool all data to measure global loss
        all_X = []
        all_y = []
        for X, y in shards.values():
            for row in X:
                all_X.append(row)
            for t in y:
                all_y.append(t)

        loss = mse_loss(global_weights, all_X, all_y)
        print("round", round_id, "global_loss", loss)

    print("learned w:", [round(v, 3) for v in global_weights["w"]], "b:", round(global_weights["b"][0], 3))
    print("true   w:", TRUE_W, "b:", TRUE_B)
