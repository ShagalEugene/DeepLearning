import numpy as np


def softmax_loss_naive(W, X, y, reg):
    """
    Функция потерь softmax, наивная реализация (с циклами)

    Входы имеют размерность D, число классов — C, вычисления ведутся над
    мини-батчами из N примеров.

    Входные данные:
    - W: массив numpy формы (D, C) с весами.
    - X: массив numpy формы (N, D) — мини-батч данных.
    - y: массив numpy формы (N,) с обучающими метками; y[i] = c означает,
      что X[i] имеет метку c, где 0 <= c < C.
    - reg: (float) коэффициент регуляризации

    Возвращает кортеж из:
    - значения функции потерь (одно число типа float)
    - градиента по весам W — массива той же формы, что и W
    """
    # Инициализируем значение потерь и градиент нулями.
    loss = 0.0
    dW = np.zeros_like(W)

    # вычисляем значение функции потерь и градиент
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # вычисляем вероятности численно устойчивым способом
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # нормировка
        logp = np.log(p)

        loss -= logp[y[i]]  # потери — это отрицательный логарифм вероятности

        p[y[i]] -= 1
        dW += np.outer(X[i], p) 


    # усредняем потери по примерам и добавляем регуляризацию
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Функция потерь softmax, векторизованная версия.

    Входные и выходные данные — такие же, как у softmax_loss_naive.
    """
    # Инициализируем значение потерь и градиент нулями.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    scores = X @ W
    scores -= np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)
    logp = -np.log(p[np.arange(num_train), y])
    
    loss = np.sum(logp) / num_train + reg * np.sum(W * W)

    p[np.arange(num_train), y] -= 1
    dW = X.T @ p / num_train + 2 * reg * W

    return loss, dW
