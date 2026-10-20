import numpy as np


def affine_forward(x, w, b):
    """
    Вычисляет прямой проход для аффинного (полносвязного) слоя.

    Вход x имеет форму (N, d_1, ..., d_k) и содержит мини-батч из N
    примеров, где каждый пример x[i] имеет форму (d_1, ..., d_k). Каждый
    вход нужно развернуть в вектор размерности D = d_1 * ... * d_k, а затем
    преобразовать его в выходной вектор размерности M.

    Входные данные:
    - x: массив numpy с входными данными формы (N, d_1, ..., d_k)
    - w: массив numpy весов формы (D, M)
    - b: массив numpy смещений формы (M,)

    Возвращает кортеж:
    - out: выход формы (N, M)
    - cache: (x, w, b)
    """
    out = None
    x_flatten = x.reshape(x.shape[0], -1)
    out = x_flatten.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Вычисляет обратный проход для аффинного слоя.

    Входные данные:
    - dout: градиент, пришедший сверху (upstream), формы (N, M)
    - cache: кортеж из:
      - x: входные данные формы (N, d_1, ..., d_k)
      - w: веса формы (D, M)
      - b: смещения формы (M,)

    Возвращает кортеж:
    - dx: градиент по x формы (N, d_1, ..., d_k)
    - dw: градиент по w формы (D, M)
    - db: градиент по b формы (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    
    x_flatten = x.reshape(x.shape[0], -1)
    dw = x_flatten.T.dot(dout)
    db = np.sum(dout, axis=0)
    dx_flatten = dout.dot(w.T)
    dx = dx_flatten.reshape(x.shape)

    return dx, dw, db


def relu_forward(x):
    """
    Вычисляет прямой проход для слоя ReLU (rectified linear units).

    Входные данные:
    - x: входные данные произвольной формы

    Возвращает кортеж:
    - out: выход той же формы, что и x
    - cache: x
    """
    out = None
    
    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Вычисляет обратный проход для слоя ReLU (rectified linear units).

    Входные данные:
    - dout: градиенты, пришедшие сверху (upstream), произвольной формы
    - cache: вход x той же формы, что и dout

    Возвращает:
    - dx: градиент по x
    """
    dx, x = None, cache
    
    dx = dout * (x > 0)

    return dx


def softmax_loss(x, y):
    """
    Вычисляет значение функции потерь и градиент для softmax-классификатора.

    Входные данные:
    - x: входные данные формы (N, C), где x[i, j] — оценка (score) j-го
      класса для i-го входа.
    - y: вектор меток формы (N,), где y[i] — метка для x[i] и
      0 <= y[i] < C

    Возвращает кортеж:
    - loss: скаляр — значение функции потерь
    - dx: градиент функции потерь по x
    """
    loss, dx = None, None

    loss = 0.0
    num_train = x.shape[0]

    scores = x - np.max(x, axis=1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)
    logp = -np.log(p[np.arange(num_train), y])
    
    loss = np.sum(logp) / num_train

    dx = p
    dx[np.arange(num_train), y] -= 1
    dx /= num_train

    return loss, dx
