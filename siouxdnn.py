import numpy as np
import tensorflow as tf
import pandas

# Data and environment

def load_data():
    # Read data from csv file and separate input from output
    data = pandas.read_csv('data/Breast_cancer_data.csv')
    data = data.sample(frac=1, random_state=1)
    X = np.array(data.drop('diagnosis', axis=1), dtype=np.float32)
    Y = np.array(data['diagnosis'], dtype=np.float32).reshape((-1,1))

    # Normalize input
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    # Split into training and validation sets
    val_size = 64
    X_train = X[:-val_size]
    Y_train = Y[:-val_size]
    X_val = X[-val_size:]
    Y_val = Y[-val_size:]

    return X_train, Y_train, X_val, Y_val

def reset_seed(seed = 1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# Core functions

def dense(a_prev, w, b):
    return np.matmul(a_prev, w) + b

def relu(z):
    return np.maximum(z, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7) # Prevent log of zero
    losses = -(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))
    loss = np.mean(losses)
    return loss

def binary_cross_entropy_loss_backward(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7) # Prevent division by zero
    N = len(y_pred)
    dl_dlosses = 1 / N
    dlosses_dy_pred = (y_pred - y_true) / (y_pred * (1 - y_pred))
    dl_dy_pred = dl_dlosses * dlosses_dy_pred
    return dl_dy_pred

def sigmoid_backward(z, dl_da):
    sig = sigmoid(z)
    return dl_da * sig * (1 - sig)

def relu_backward(z, dl_da):
    dl_dz = np.array(dl_da, copy=True)
    dl_dz[z <= 0] = 0
    return dl_dz

def dense_backward(a_prev, w, dl_dz):
    dl_dw = np.matmul(a_prev.T, dl_dz)
    dl_db = np.sum(dl_dz, axis=0, keepdims=True)
    dl_da_prev = np.matmul(dl_dz, w.T)
    return dl_dw, dl_db, dl_da_prev


# Complete model


class Model(object):
    def __init__(self):
        N0, N1, N2,N3 = 5, 64, 64, 1
        self.w1 = np.random.uniform(-0.5, 0.5, size=(N0, N1))
        self.b1 = np.zeros((1, N1))
        self.w2 = np.random.uniform(-0.5, 0.5, size=(N1, N2))
        self.b2 = np.zeros((1, N2))
        self.w3 = np.random.uniform(-0.5, 0.5, size=(N2, N3))
        self.b3 = np.zeros((1, N3))
        
    def predict(self, x):
        a0 = x
        z1 = dense(a0, self.w1, self.b1)
        a1 = relu(z1)
        z2 = dense(a1, self.w2, self.b2)
        a2 = relu(z2)
        z3 = dense(a2, self.w3, self.b3)
        a3 = sigmoid(z3)
        y_pred = a3
        return y_pred
    
    def evaluate(self, x, y_true):
        y_pred = self.predict(x)
        loss = binary_cross_entropy_loss(y_true, y_pred)
        return loss

    def get_gradients(self, x, y_true):
        a0 = x
        z1 = dense(a0, self.w1, self.b1)
        a1 = relu(z1)
        z2 = dense(a1, self.w2, self.b2)
        a2 = relu(z2)
        z3 = dense(a2, self.w3, self.b3)
        a3 = sigmoid(z3)
        y_pred = a3

        loss = binary_cross_entropy_loss(y_true, y_pred)

        dl_dy_pred = binary_cross_entropy_loss_backward(y_true, y_pred)

        dl_da3 = dl_dy_pred
        dl_dz3 = sigmoid_backward(z3, dl_da3)
        dl_dw3, dl_db3, dl_da2 = dense_backward(a2, self.w3, dl_dz3)
        dl_dz2 = relu_backward(z2, dl_da2)
        dl_dw2, dl_db2, dl_da1 = dense_backward(a1, self.w2, dl_dz2)
        dl_dz1 = relu_backward(z1, dl_da1)
        dl_dw1, dl_db1, dl_da0 = dense_backward(a0, self.w1, dl_dz1)
        dl_dx = dl_da0
        return loss, dl_dx, dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3
    

def train(model, x, y_true, learning_rate):
    loss, dl_dx, dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3 = model.get_gradients(x, y_true)
    model.w1 -= learning_rate * dl_dw1
    model.b1 -= learning_rate * dl_db1
    model.w2 -= learning_rate * dl_dw2
    model.b2 -= learning_rate * dl_db2
    model.w3 -= learning_rate * dl_dw3
    model.b3 -= learning_rate * dl_db3
    return loss

# Metrics

def get_metrics(y_true, y_pred):
    y_true = (y_true >= 0.5)
    y_pred = (y_pred >= 0.5)

    tp = np.sum(y_pred & y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        
    return accuracy, precision, recall
