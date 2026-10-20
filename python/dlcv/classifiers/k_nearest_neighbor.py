import numpy as np


class KNearestNeighbor(object):
    """ Классификатор kNN (k ближайших соседей) с L2-расстоянием """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Обучает классификатор. Для метода k ближайших соседей обучение сводится
        к простому запоминанию обучающих данных.

        Входные данные:
        - X: массив numpy формы (num_train, D) с обучающими данными:
          num_train примеров, каждый размерности D.
        - y: массив numpy формы (N,) с метками обучающей выборки, где
             y[i] — метка для X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Предсказывает метки для тестовых данных с помощью этого классификатора.

        Входные данные:
        - X: массив numpy формы (num_test, D) с тестовыми данными:
             num_test примеров, каждый размерности D.
        - k: число ближайших соседей, голосующих за предсказанную метку.
        - num_loops: определяет, какую реализацию использовать для вычисления
          расстояний между обучающими и тестовыми точками.

        Возвращает:
        - y: массив numpy формы (num_test,) с предсказанными метками для
          тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Недопустимое значение num_loops: %d" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой
        обучающей точкой из self.X_train с помощью вложенного цикла по
        обучающим и тестовым данным.

        Входные данные:
        - X: массив numpy формы (num_test, D) с тестовыми данными.

        Возвращает:
        - dists: массив numpy формы (num_test, num_train), где dists[i, j] —
          евклидово расстояние между i-й тестовой и j-й обучающей точками.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum((X[i] - self.X_train[j])**2))

        return dists

    def compute_distances_one_loop(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой
        обучающей точкой из self.X_train с помощью одного цикла по тестовым
        данным.

        Входные / выходные данные: такие же, как у compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i, :] = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))

        return dists

    def compute_distances_no_loops(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой
        обучающей точкой из self.X_train без явных циклов.

        Входные / выходные данные: такие же, как у compute_distances_two_loops
        """
        test_norm_squared = np.sum(X**2, axis=1)
        train_norm_squared = np.sum(self.X_train**2, axis=1)
        dists = np.sqrt(test_norm_squared[:, None] + train_norm_squared[None, :] - 2 * X @ self.X_train.T)

        return dists

    def predict_labels(self, dists, k=1):
        """
        По матрице расстояний между тестовыми и обучающими точками предсказывает
        метку для каждой тестовой точки.

        Входные данные:
        - dists: массив numpy формы (num_test, num_train), где dists[i, j] —
          расстояние между i-й тестовой и j-й обучающей точками.

        Возвращает:
        - y: массив numpy формы (num_test,) с предсказанными метками для
          тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i])[:k]]
            labels, counts = np.unique(closest_y, return_counts=True)
            max_count = max(counts)
            y_pred[i] = min(labels[counts == max_count])

        return y_pred