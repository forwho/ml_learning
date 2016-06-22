from numpy import *
import sys
import random
import matplotlib.pyplot as plt
def generateData(w,b):
    data1 = []
    data2 = []
    for i in range(10):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        while (w * x + b - y < 0):
            x = random.randint(0, 100)
            y = random.randint(0, 100)
        data1.append([x, y])
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        while (w * x + b - y > 0):
            x = random.randint(0, 100)
            y = random.randint(0, 100)
        data2.append([x, y])
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([val[0] for val in data1], [val[1] for val in data1], c = 'r')
    ax.scatter([val[0] for val in data2], [val[1] for val in data2], c = 'b')
    plt.show()
    """
    return data1, data2

def endLearning(w, b, data, labels):
    x = -1; y = -1
    for val in data:
        if ((w * val[0] + b - val[1]) * labels[data.index(val)] < 0):
            x = val[0]; y = labels[data.index(val)]
            break;
    return x, y

def recLearning(w, b, step, data, labels, i):
    x, y = endLearning(w, b, data, labels)
    if (x != -1 ):
        w += step * x * y
        b += step * y
        i += 1
        x = range(100)
        y = w * array(x) - b
        ax.plot(array(x), array(y), label="line")
        return recLearning(w, b, step, data, labels, i)
    else:
        return w, b


if __name__ == "__main__":
    sys.setrecursionlimit(100000000)
    data1, data2 = generateData(3, 1)
    data = []
    labels = []
    for x in data1:
        data.append(x)
        labels.append(1)
    for x in data2:
        data.append(x)
        labels.append(-1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([val[0] for val in data1], [val[1] for val in data1], c='r')
    ax.scatter([val[0] for val in data2], [val[1] for val in data2], c='b')
    w, b =  recLearning(0, 0, 0.1, data, labels, i = 0)
    plt.show()