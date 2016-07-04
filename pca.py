from numpy import *
import matplotlib.pyplot as plt

def loadData(filename):
    dataSet = []
    for line in open(filename):
        dataSet.append([float(val) for val in line.strip().split('\t')])
    return mat(dataSet)

def pca(dataSet, topN):
    meanVal = mean(dataSet, axis = 0)
    meanRemoved = dataSet - meanVal
    covMat = cov(meanRemoved, rowvar = 0)
    eigVals, eigVectors = linalg.eig(mat(covMat))
    eigValsInd = argsort(eigVals)
    eigValsInd = eigValsInd[: -(topN + 1) : -1]
    eigVectors = eigVectors[:, eigValsInd]
    lowDDataMat = meanRemoved * eigVectors
    reconMat = lowDDataMat * eigVectors.T + meanVal
    return lowDDataMat, reconMat

if __name__ == '__main__':
    dataSet = loadData("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch13\\testSet3.txt")
    lowDDataMat, reconMat = pca(dataSet, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataSet[:, 0], dataSet[:, 1])
    plt.show()