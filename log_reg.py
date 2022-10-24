import numpy as np
import torch
from scipy.special import softmax
from torch import nn


class Log_Reg:
    def __init__(self, K):
        self.K = K

    _disp = 0.01
    _gamma = 0.01
    _epsilon = 0.001

    def learn(self, X, T, X_valid, T_valid, rand_init=1):
        W = []
        b = []
        Accuracy = []
        U = np.ones(X.shape[0])
        if rand_init == 0:
            W.append(np.random.uniform(0, 1, X.shape[1] * self.K).reshape((self.K, X.shape[1])))
            b.append(np.random.uniform(0, 1, self.K))
        elif rand_init == 1:
            W.append(np.random.normal(0, self._disp, X.shape[1] * self.K).reshape((self.K, X.shape[1])))
            b.append(np.random.normal(0, self._disp, self.K))
        elif rand_init == 2:
            W.append(np.array(nn.init.xavier_normal_(torch.empty((self.K, X.shape[1])))))
            b.append(np.array(nn.init.xavier_normal_(torch.empty((1, self.K)))).reshape(self.K))
        else:
            print("rand_init may be 0, 1, 2, 3")
        Accuracy.append(self.accuracy(W[-1], b[-1], X_valid, T_valid))
        exit = False
        counter = 0
        while not exit:
            counter += 1
            Y = self.prediction(W[-1], b[-1], X)
            W.append(W[-1] - self._gamma * (Y - T).T.dot(X))
            b.append(b[-1] - self._gamma * (Y - T).T.dot(U))
            Accuracy.append(self.accuracy(W[-1], b[-1], X_valid, T_valid))
            # if Accuracy[-2] == Accuracy[-1] and Accuracy[-3] == Accuracy[-1]:
            #     print(counter)
            #     exit = True
            if counter == 200:
                exit = True
        return W, b, Accuracy

    def prediction(self, W, b, X):
        Z = np.array([W.dot(x) + b for x in X])
        Z = np.array([z - np.amax(z) for z in Z])
        return softmax(Z, axis=1)

    def accuracy(self, W, b, X, T):
        P = 0
        Y = self.prediction(W, b, X)
        A = np.argmax(Y, axis=1)
        B = np.argmax(T, axis=1)
        for i in range(len(A)):
            if A[i] == B[i]:
                P += 1
        return P / len(X)

    # def E(self, W, b, X, T):
