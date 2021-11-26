from numpy import genfromtxt
from random import *
from math import *
import numpy as np


def distance(x, y):
    total = 0
    for i in range(len(x) - 1):
        total += (x[i] - y[i]) ** 2
    return sqrt(total)


def shuffle_data(my_data):
    train_set = []
    val_set = []
    test_set = []
    for s in my_data:
        r = random()
        if (r >= 0) and (r <= 0.7):
            train_set.append(s)
        elif (r >= 0.7) and (r <= 0.85):
            val_set.append(s)
        else:
            test_set.append(s)
    return train_set, val_set, test_set


data_path = './iris.csv'
my_data = genfromtxt(data_path, delimiter=',')
shuffle(my_data)

for i in range(1, 20,2):
    training_set, validation_set, test_set = shuffle_data(my_data)
    k=i
    L = []
    total_correct = 0
    for v in validation_set:
        for t in training_set:
            L.append([t, distance(v, t)])
        L.sort(key=lambda val: val[1])
        # print(L)
        s = [val[0][len(val[0]) - 1] for val in L[:k]]
        if v[len(v) - 1] == np.bincount(s).argmax():
            total_correct += 1
        L = []
    print(f'K: {k}\t Validate sample: {len(validation_set)}'
          f'\t\tCorrect: {total_correct}\t\taccuracy: {total_correct / len(validation_set) * 100}')
