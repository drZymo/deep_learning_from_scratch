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
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred))

def binary_cross_entropy_loss_backward(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    return (1 / y_pred.shape[0]) * (y_pred - y_true) / (y_pred * (1 - y_pred))

def sigmoid_backward(z, da):
    sig = sigmoid(z)
    return da * sig * (1 - sig)

def relu_backward(z, da):
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz

def dense_backward(a_prev, dz, w):
    dw = np.matmul(a_prev.T, dz)
    db = np.sum(dz, axis=0, keepdims=True)
    da_prev = np.matmul(dz, w.T)
    return dw, db, da_prev


# Complete model


class Model(object):
    def __init__(self):
        N0, N1, N2,N3 = 5, 64, 64, 1
        self.w1 = np.random.uniform(-0.3, 0.3, size=(N0, N1))
        self.b1 = np.zeros((1, N1))
        self.w2 = np.random.uniform(-0.2, 0.2, size=(N1, N2))
        self.b2 = np.zeros((1, N2))
        self.w3 = np.random.uniform(-0.3, 0.3, size=(N2, N3))
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
    
    def compute_loss(self, x, y_true):
        y_pred = self.predict(x)
        return binary_cross_entropy_loss(y_true, y_pred)

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
        dy_pred = binary_cross_entropy_loss_backward(y_true, y_pred)

        da3 = dy_pred
        dz3 = sigmoid_backward(z3, da3)
        dw3, db3, da2 = dense_backward(a2, dz3, self.w3)
        dz2 = relu_backward(z2, da2)
        dw2, db2, da1 = dense_backward(a1, dz2, self.w2)
        dz1 = relu_backward(z1, da1)
        dw1, db1, da0 = dense_backward(a0, dz1, self.w1)
        dx = da0
        return loss, dx, dw1, db1, dw2, db2, dw3, db3
    

def train(model, x, y_true, learning_rate):
    loss, dx, dw1, db1, dw2, db2, dw3, db3 = model.get_gradients(x, y_true)
    model.w1 -= learning_rate * dw1
    model.b1 -= learning_rate * db1
    model.w2 -= learning_rate * dw2
    model.b2 -= learning_rate * db2
    model.w3 -= learning_rate * dw3
    model.b3 -= learning_rate * db3
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
