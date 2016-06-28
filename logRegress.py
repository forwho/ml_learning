from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch05\\testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmod1(W, X):
    return 1 / (1 + exp(- W * X))

def logLearning(W, data, label, alpha, k):
    W = array(W)
    data = matrix(data)
    label = array(label)
    for i in range(k):
        proData = sigmod1(W, data.T)
        diff = label - proData
        W = W + alpha * diff * data
    return W

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(wei):
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    print(weights)
    y = (- weights[0] - weights[0] * x) / weights[0]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def randomLearning(W, data, label, alpha, k):
    W = array(W)
    for j in range(k):
        dataIndex = list(range(len(data)))
        for i in range(len(data)):
            alpha = alpha / (1.0 + j + i) + 0.1
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmod1(W, array(data[randIndex]))
            error = label[randIndex] - h
            W = W + alpha * error * array(data[randIndex])
            del(dataIndex[randIndex])
    return W

def gradTest():
    dataArr, labelMat = loadDataSet()
    wei = stoGradAscent1(dataArr, labelMat)
    #W = ones(len(dataArr[0]))
    #wei = randomLearning(W, dataArr, labelMat, 4, 150)
    plotBestFit(wei)

def stoGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * array(dataMatrix[i])
    return weights

def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.1
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights

if __name__ == "__main__":
    gradTest()