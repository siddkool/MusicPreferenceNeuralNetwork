import numpy as np

np.random.seed(42)

MUSIC = ["mellow", "unpretentious", "sophisticated", "intense", "contemporary"]


def load(path="data.tsv"):
    header = open(path).readline().strip().split("\t")
    idx = {name: i for i, name in enumerate(header)}
    X, Y = [], []
    for line in open(path).readlines()[1:]:
        parts = line.strip().split("\t")
        if len(parts) < len(header):
            continue
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            continue
        age = vals[idx["age"]]
        eq = np.mean([vals[idx["eq1"]], vals[idx["eq2"]], vals[idx["eq3"]]])
        sq = np.mean([vals[idx["sq1"]], vals[idx["sq2"]], vals[idx["sq3"]]])
        X.append([(age - 10) / 70.0, (eq - 1) / 4.0, (sq - 1) / 4.0])
        Y.append([(vals[idx[m]] - 1) / 4.0 for m in MUSIC])
    return np.array(X), np.array(Y)


def relu(z):
    return np.maximum(0.0, z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class Net:
    def __init__(self, n_in, h1=32, h2=16, n_out=5):
        self.W1 = np.random.randn(n_in, h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, n_out) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(n_out)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def train(self, X, Y, epochs=400, lr=0.01):
        mW1 = vW1 = mb1 = vb1 = 0
        mW2 = vW2 = mb2 = vb2 = 0
        mW3 = vW3 = mb3 = vb3 = 0
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            self.forward(X)
            total = Y.shape[0] * Y.shape[1]
            d3 = 2 * (self.a3 - Y) / total * (self.a3 * (1 - self.a3))
            dW3 = self.a2.T @ d3
            db3 = d3.sum(0)
            d2 = (d3 @ self.W3.T) * (self.z2 > 0)
            dW2 = self.a1.T @ d2
            db2 = d2.sum(0)
            d1 = (d2 @ self.W2.T) * (self.z1 > 0)
            dW1 = X.T @ d1
            db1 = d1.sum(0)
            mW1 = b1_ * mW1 + (1 - b1_) * dW1
            vW1 = b2_ * vW1 + (1 - b2_) * dW1 ** 2
            self.W1 -= lr * (mW1 / (1 - b1_ ** t)) / (np.sqrt(vW1 / (1 - b2_ ** t)) + eps)
            mb1 = b1_ * mb1 + (1 - b1_) * db1
            vb1 = b2_ * vb1 + (1 - b2_) * db1 ** 2
            self.b1 -= lr * (mb1 / (1 - b1_ ** t)) / (np.sqrt(vb1 / (1 - b2_ ** t)) + eps)
            mW2 = b1_ * mW2 + (1 - b1_) * dW2
            vW2 = b2_ * vW2 + (1 - b2_) * dW2 ** 2
            self.W2 -= lr * (mW2 / (1 - b1_ ** t)) / (np.sqrt(vW2 / (1 - b2_ ** t)) + eps)
            mb2 = b1_ * mb2 + (1 - b1_) * db2
            vb2 = b2_ * vb2 + (1 - b2_) * db2 ** 2
            self.b2 -= lr * (mb2 / (1 - b1_ ** t)) / (np.sqrt(vb2 / (1 - b2_ ** t)) + eps)
            mW3 = b1_ * mW3 + (1 - b1_) * dW3
            vW3 = b2_ * vW3 + (1 - b2_) * dW3 ** 2
            self.W3 -= lr * (mW3 / (1 - b1_ ** t)) / (np.sqrt(vW3 / (1 - b2_ ** t)) + eps)
            mb3 = b1_ * mb3 + (1 - b1_) * db3
            vb3 = b2_ * vb3 + (1 - b2_) * db3 ** 2
            self.b3 -= lr * (mb3 / (1 - b1_ ** t)) / (np.sqrt(vb3 / (1 - b2_ ** t)) + eps)


def evaluate(X, Y, folds=5):
    n = len(X)
    idx = np.random.permutation(n)
    parts = np.array_split(idx, folds)
    maes, withins, baselines = [], [], []
    for i in range(folds):
        test = parts[i]
        train = np.concatenate([parts[j] for j in range(folds) if j != i])
        net = Net(X.shape[1])
        net.train(X[train], Y[train])
        pred = net.forward(X[test])
        maes.append(np.mean(np.abs(pred - Y[test])) * 4)
        withins.append(np.mean(np.abs(pred - Y[test]) * 4 <= 1.0) * 100)
        base = Y[train].mean(0)
        baselines.append(np.mean(np.abs(base - Y[test])) * 4)
    return np.mean(maes), np.mean(withins), np.mean(baselines)


if __name__ == "__main__":
    X, Y = load()
    print("loaded", len(X), "responses")
    mae, within, base = evaluate(X, Y)
    print("mean absolute error (points):", round(mae, 2))
    print("within 1 point:", round(within), "%")
    print("baseline mae (predict mean):", round(base, 2))
    if mae < base:
        print("model beats baseline by", round(base - mae, 3), "points")
    else:
        print("model does not beat baseline (expected at this sample size)")
