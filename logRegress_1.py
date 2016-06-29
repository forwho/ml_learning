from numpy import *
import random

def loadData(filename):
    data = [];  label = []
    for line in open("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch05\%s" % filename):
        line = [float(val) for val in line.split('\t')]
        dataLine = line[:-2]
        dataLine.append(1)
        data.append(dataLine)
        label.append(line[-1])
    return data, label

def sigmod(W, X):
    return 1 / (1 + exp(- W * X))

def logLearning(W, data, label, alpha, k):
    W = array(W)
    data = matrix(data)
    label = array(label)
    for i in range(k):
        proData = sigmod(W, data.T)
        diff = label - proData
        W = W + alpha * diff * data
    return W

def randomLearning(W, data, label, alpha, k):
    W = array(W)
    for i in range(k):
        dataIndex = list(range(len(data)))
        for j in range(len(data)):
            alpha = alpha / (1.0 + j + i) + 0.1
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod(W, array(data[randIndex]))
            error = label[randIndex] - h
            W = W + alpha * error * array(data[randIndex])
            del(dataIndex[randIndex])
    return W

def logForcast(W, inData):
    if sigmod(W, inData) > 0.5:
        return 1
    else:
        return 0

if __name__ == "__main__":
    data, label = loadData("horseColicTraining.txt")
    W = ones(len(data[0]))
    W = randomLearning(W, data, label, 0.010, 1000)
    forData, forLabel = loadData("horseColicTest.txt")
    erroNum = 0
    for val in forData:
        print("The real num is %f, the forcast num is %f." % (forLabel[forData.index(val)], logForcast(W, matrix(val).T)))
        if forLabel[forData.index(val)] != logForcast(W, matrix(val).T):
            erroNum += 1
    print("The error ratio is %f" % (erroNum / len(forData)))