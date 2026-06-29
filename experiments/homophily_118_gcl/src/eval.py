import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


def linear_eval(embeddings, labels, train_mask, val_mask, test_mask, max_iter=3000):
    x = embeddings.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    train = train_mask.detach().cpu().numpy().astype(bool)
    val = val_mask.detach().cpu().numpy().astype(bool)
    test = test_mask.detach().cpu().numpy().astype(bool)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_val = scaler.transform(x[val])
    x_test = scaler.transform(x[test])
    best = None
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        clf = LogisticRegression(
            C=c,
            max_iter=int(max_iter),
            solver="lbfgs",
            random_state=0,
        )
        clf.fit(x_train, y[train])
        val_pred = clf.predict(x_val)
        val_acc = accuracy_score(y[val], val_pred)
        if best is None or val_acc > best["val_acc"]:
            best = {"c": c, "val_acc": val_acc, "clf": clf}
    pred = best["clf"].predict(x_test)
    return {
        "accuracy": float(accuracy_score(y[test], pred)),
        "F1Mi": float(f1_score(y[test], pred, average="micro")),
        "F1Ma": float(f1_score(y[test], pred, average="macro")),
        "val_accuracy": float(best["val_acc"]),
        "best_c": float(best["c"]),
        "num_train": int(np.sum(train)),
        "num_val": int(np.sum(val)),
        "num_test": int(np.sum(test)),
    }
