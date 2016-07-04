from numpy import *
import random
import matplotlib.pyplot as plt

def loadData(filename):
    dataList = []
    for line in open(filename):
        line = line.strip().split()
        dataList.append([float(val) for val in line])
    return dataList

def k_means(dataList, k):
    brycenter = initial(dataList, k)
    flag = True
    while(flag):
        dataSet = allocate(dataList, k, brycenter)
        brycenter = calBrycenter(dataSet)
        dataSet1 = allocate(dataList, k, brycenter)
        if dataSet == dataSet1:
            flag = False
        else:
            dataSet = dataSet1
            brycenter = calBrycenter(dataSet1)
    return dataSet

def initial(dataList, k):
    dataSetIndex = []
    for i in range(k):
        ran = random.randint(1, len(dataList)) - 1
        if ran not in dataSetIndex:
            dataSetIndex.append(ran)
    dataSet  = []
    for index in dataSetIndex:
        dataSet.append([dataList[index]])
    return dataSet

def allocate(dataList, k, calBrycenter):
    dataSet = []
    for i in range(k):
        dataSet.append([])
    for data in dataList:
        minDistance = 100000
        for i in range(k):
            diff = [calBrycenter[i][0][j] - data[j] for j in range(len(data))]
            distance = sum([val ** 2 for val in diff])
            if distance < minDistance:
                minDistance = distance
                minK = i
        if minDistance > 0:
            dataSet[minK].append(data)
    return dataSet

def calBrycenter(dataSet):
    brycenterSet = []
    for data in dataSet:
        brycenterSet.append((matrix(data).sum(0) / len(data)).tolist())
    return brycenterSet


if __name__ == '__main__':
    dataList = loadData('E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch10\\testSet.txt')
    dataSet = k_means(dataList, 4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = ['r', 'b', 'y', 'g']
    for i in range(4):
        ax.scatter([val[0] for val in dataSet[i]], [val[1] for val in dataSet[i]], c = color[i])
    plt.show()



