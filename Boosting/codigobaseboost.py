# Tildes omitidas intencionalmente.
import sys
import math

EPS = 1e-15

def stump_predict(x, feature_idx, threshold, polarity):
    return polarity if x[feature_idx] >= threshold else -polarity

def best_stump(X, y, w):
    """Encontrar el mejor stump (feature, threshold, polaridad) con el menor error ponderado."""
    n = len(X)
    d = len(X[0])
    best_feature = 0
    best_threshold = 0.0
    best_polarity = 1
    best_err = float('inf')

    for j in range(d):
        vals = sorted(set(row[j] for row in X))
        # thresholds candidatos
        if len(vals) == 1:
            thresholds = [vals[0]]
        else:
            thresholds = [vals[0] - 1.0]
            for k in range(len(vals) - 1):
                thresholds.append((vals[k] + vals[k + 1]) / 2.0)
            thresholds.append(vals[-1] + 1.0)

        for t in thresholds:
            for pol in (1, -1):
                err = 0.0
                # Tasa de error ponderado
                for i in range(n):
                    pred = stump_predict(X[i], j, t, pol)
                    if pred != y[i]:
                        err += w[i]
                if err + 1e-12 < best_err:
                    best_err = err
                    best_feature = j
                    best_threshold = t
                    best_polarity = pol

    return best_feature, best_threshold, best_polarity, best_err

def train_adaboost(X, y, M):
    n = len(X)
    w = [1.0 / n] * n
    stumps = []
    alphas = []

    for _ in range(M):
        j, t, pol, err = best_stump(X, y, w)

        # Early stopping
        if err >= 0.5 - EPS:
            break

        #========================================
        # Aqui va tu codigo
        #========================================
        # Calcular alpha_m
        err = max(err, EPS)
        alpha = math.log((1.0 - err) / err)

        # Guardar stump y peso
        stumps.append((j, t, pol))
        alphas.append(alpha)

        # Actualizar pesos
        for i in range(n):
            pred = stump_predict(X[i], j, t, pol)
            if pred != y[i]:
                w[i] *= math.exp(alpha)

        # Normalizar pesos
        sum_w = sum(w)
        if sum_w > 0:
            w = [wi / sum_w for wi in w]

    return stumps, alphas

def predict_one(x, stumps, alphas):
    s = 0.0
    for (j, t, pol), a in zip(stumps, alphas):
        s += a * stump_predict(x, j, t, pol)
    return 1 if s >= 0.0 else -1  # tie -> +1

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)

    N = int(next(it))
    D = int(next(it))
    M = int(next(it))
    Q = int(next(it))

    X = []
    y = []
    for _ in range(N):
        row = [float(next(it)) for _ in range(D)]
        lab = int(next(it))
        if lab != 1 and lab != -1:
            lab = 1 if lab > 0 else -1
        X.append(row)
        y.append(lab)

    queries = []
    for _ in range(Q):
        qrow = [float(next(it)) for _ in range(D)]
        queries.append(qrow)

    stumps, alphas = train_adaboost(X, y, M)

    out_lines = []
    for q in queries:
        out_lines.append(str(predict_one(q, stumps, alphas)))

    sys.stdout.write("\n".join(out_lines))

if __name__ == "__main__":
    main()
