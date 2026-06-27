import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            base_random_state = kwargs.pop('random_state', None)
            results = []
            for repeat_idx in range(n_times):
                call_kwargs = dict(kwargs)
                if base_random_state is not None:
                    call_kwargs['random_state'] = base_random_state + repeat_idx
                results.append(f(*args, **call_kwargs))
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, dtype=bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


def format_statistics(values, function_name):
    statistics = {
        key: {
            'mean': value,
            'std': 0.0,
        }
        for key, value in values.items()
    }
    print_statistics(statistics, function_name)
    return statistics


def _to_numpy_masks(mask):
    if mask is None:
        return None
    return mask.detach().cpu().numpy().astype(bool)


def _fit_logreg(X_train, y_train, c_value):
    clf = LogisticRegression(solver='lbfgs', C=c_value, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


def label_classification_with_masks(embeddings, y, train_mask, val_mask, test_mask):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    train_mask = _to_numpy_masks(train_mask)
    val_mask = _to_numpy_masks(val_mask)
    test_mask = _to_numpy_masks(test_mask)

    c_values = 2.0 ** np.arange(-10, 10)
    best_c = c_values[0]
    best_score = -1.0

    if val_mask is not None and val_mask.any():
        for c_value in c_values:
            clf = _fit_logreg(X[train_mask], Y[train_mask], c_value)
            y_val_pred = clf.predict(X[val_mask])
            score = f1_score(Y[val_mask], y_val_pred, average='micro')
            if score > best_score:
                best_score = score
                best_c = c_value
    else:
        logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf = GridSearchCV(
            estimator=logreg,
            param_grid=dict(C=c_values),
            n_jobs=8,
            cv=5,
            verbose=0,
        )
        clf.fit(X[train_mask], Y[train_mask])
        best_c = clf.best_params_['C']

    clf = _fit_logreg(X[train_mask], Y[train_mask], best_c)
    y_pred = clf.predict(X[test_mask])
    values = {
        'F1Mi': f1_score(Y[test_mask], y_pred, average='micro'),
        'F1Ma': f1_score(Y[test_mask], y_pred, average='macro'),
    }
    return format_statistics(values, 'label_classification_with_masks')


@repeat(3)
def label_classification(embeddings, y, ratio, random_state=None):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio,
                                                        random_state=random_state)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")

    return {
        'F1Mi': micro,
        'F1Ma': macro
    }
