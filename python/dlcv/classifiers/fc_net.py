import os
import numpy as np
from ..layers import *
from ..layer_utils import *

class TwoLayerNet(object):
    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg
        ############################################################################
        # Инициализация весов
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

    def loss(self, X, y=None):
        scores = None
        ############################################################################
        # Прямой проход
        ############################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]

        h1 = np.maximum(0, X.dot(W1) + b1)  # ReLU
        scores = h1.dot(W2) + b2
        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Обратный проход + loss
        ############################################################################
        # Softmax loss
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        # Градиенты
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W2'] = h1.T.dot(dscores) + self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        dh1 = dscores.dot(W2.T)
        dh1[h1 <= 0] = 0  # backprop через ReLU

        grads['W1'] = X.T.dot(dh1) + self.reg * W1
        grads['b1'] = np.sum(dh1, axis=0)
        ############################################################################
        #                            КОНЕЦ ВАШЕГО КОДА                             #
        ############################################################################
        return loss, grads

    def save(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "сохранён.")
    
    def load(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "не найден.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "загружен.")
        return True


class FullyConnectedNet(object):
    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
    ):
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, len(dims)):
            self.params[f'W{i}'] = weight_scale * np.random.randn(dims[i-1], dims[i])
            self.params[f'b{i}'] = np.zeros(dims[i])
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
        scores = None
        cache = {}
        out = X
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']
            out, cache[i] = affine_relu_forward(out, W, b)
        W_last = self.params[f'W{self.num_layers}']
        b_last = self.params[f'b{self.num_layers}']
        scores, cache_last = affine_forward(out, W_last, b_last)

        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)

        dx, grads[f'W{self.num_layers}'], grads[f'b{self.num_layers}'] = affine_backward(dscores, cache_last)
        grads[f'W{self.num_layers}'] += self.reg * self.params[f'W{self.num_layers}']

        for i in range(self.num_layers - 1, 0, -1):
            dx, grads[f'W{i}'], grads[f'b{i}'] = affine_relu_backward(dx, cache[i])
            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        # Добавляем регуляризацию к loss
        for i in range(1, self.num_layers + 1):
          W = self.params[f'W{i}']
          loss += 0.5 * self.reg * np.sum(W * W)
        return loss, grads

    def save(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "сохранён.")
    
    def load(self, fname):
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "не найден.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "загружен.")
        return True