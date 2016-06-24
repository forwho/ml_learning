from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    fr = open("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch08\%s" % filename)
    linesArr = fr.readlines()
    xArr = [];  yArr = []
    for line in linesArr:
        xArr.append([float(val) for val in line.split('\t')[0 : 2]])
        yArr.append(float(line.strip().split('\t')[-1]))
    return xArr, yArr

def standRegress(xArr, yArr):
    xTx = mat(xArr).T * mat(xArr)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    w = xTx.I * mat(xArr).T * mat(yArr).T
    return w

def pltTest(xArr, yArr, ws):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x[1] for x in xArr], yArr)
    yHat = xArr * ws
    ax.plot([x[1] for x in xArr], yHat)
    plt.show()

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr);   yMat = mat(yArr).T
    m = shape(xMat)[0]
    weight = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, : ]
        weight[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weight * xMat)
    if linalg.det(xTx) == 0.0:
        print("This is matrix is singular, cannot do inverse")
        return
    ws = (xTx).I * (xMat.T * (weight * yMat))
    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

if __name__ == "__main__":
    xArr, yArr = loadDataSet("ex0.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xMat = mat(xArr)
    srtInd = xMat[ : , 1].argsort(0)
    xSort = xMat[srtInd][ : , 0, : ]
    print(yHat)
    ax.scatter([x[1] for x in xArr], yArr)
    ax.plot(xSort[ : , 1], yHat[srtInd])
    plt.show()