from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
import math
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import time


def score(data, model):
    x_train, y_train = data

    y_pred = model.predict(x_train)

    correct = 0
    for i in range(len(y_pred)):
        ind = np.argmax(y_pred[i])
        ind_corr = np.argmax(y_train[i])
        if ind == ind_corr:
            correct += 1

    return (len(y_pred) - correct) / len(y_pred)


def get_model():
    model = Sequential()
    model.add(Dense(30, input_shape=(4,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model


def update_weights(model):
    old_weights = model.get_weights()

    sigma = 1
    mu = 0
    for i, layer in enumerate(model.layers):
        w = layer.get_weights()[0].flatten()
        b = layer.get_weights()[1]
        new_w = np.random.normal(mu, sigma, len(w))
        new_b = np.random.normal(mu, sigma, len(b))

        set_weights(layer, new_w + w, new_b + b)

    return model, old_weights


def set_weights(layer, vector, b):
    shape = layer.get_weights()[0].shape
    weights = vector.reshape(shape)

    layer.set_weights([weights, b])


def acceptance_ratio(cost, new_cost, temperature):
    p = np.exp(- cost / temperature)
    p_new = np.exp(- new_cost / temperature)

    return p_new / p


def simpleSA(T, rate, data, model):
    cost = score(data, model)
    iter = 0
    while T >= 0.001:

        model, old_state = update_weights(model)
        new_cost = score(data, model)

        alpha = acceptance_ratio(cost, new_cost, T)
        u = np.random.uniform(0, 1, 1)
        if u <= alpha:
            cost = new_cost
        else:
            model.set_weights(old_state)

        T *= rate
        iter += 1


iris = datasets.load_iris()
x = iris.data
y = iris.target

onehot_encoder = OneHotEncoder(sparse=False)
reshaped = y.reshape(len(y), 1)
y_onehot = onehot_encoder.fit_transform(reshaped)

x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.20)

model = get_model()

start_time = time.time()
simpleSA(1, 0.99, (x_train, y_train), model)
print("loss: ", score((x_test, y_test), model))
print("--- %s seconds ---" % (time.time() - start_time))

model_back = get_model()

start_time = time.time()
opt = Adam(lr=0.04)
model_back.compile(opt, 'categorical_crossentropy', ['accuracy'])
model_back.fit(x_train, y_train, epochs=100)
print("loss: ", score((x_test, y_test), model_back))
print("--- %s seconds ---" % (time.time() - start_time))