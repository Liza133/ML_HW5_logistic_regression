from sklearn.datasets import load_digits
from log_reg import Log_Reg
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import plotly.graph_objs as go

digits = load_digits()
X = digits.data
target = digits.target

# reshaping for OneHotEncoder
integer_encoded_reshape = target.reshape(len(target), 1)

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded_reshape)
# Normalisation
for i in range(X.shape[0]):
    X[i] = 2 * (X[i] - np.amin(X[i])) / (np.amax(X[i]) - np.amin(X[i])) - 1
# Shuffle data
tr = 0.8
val = 0.2
N = X.shape[0]
ind_prm = np.random.permutation(np.arange(N))
train_ind = ind_prm[:int(tr * N)]
valid_ind = ind_prm[int(tr * N):]
X_train, target_train, T_train, X_valid, target_valid, T_valid = X[train_ind], target[train_ind], onehot_encoded[
    train_ind], X[valid_ind], target[train_ind], onehot_encoded[valid_ind]

l_g = Log_Reg(10)
W, b, A = l_g.learn(X_train, T_train, X_valid, T_valid, rand_init=2)

x = [i for i in range(len(A))]
y = []
for i in range(len(W)):
    y.append(l_g.accuracy(W[i], b[i], X_train, T_train))
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=A, mode='lines+markers', name='valid'))
fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='train'))
fig.update_layout(title="Accuracy",
                  xaxis_title="Номер итерации",
                  yaxis_title="Значение accuracy")
fig.show()

print(A[-1])
