import math

def loadData(filename):
    dataList = []
    labelList = []
    for line in open(filename):
        line = line.strip().split('\t')
        dataList.append(line[0:-1])
    return dataList

def calEntropy(dataList, val, feature = -1,):
    if feature == -1:
        labelNum = {}
        for label in dataList[-1]:
            if label not in labelNum:
                labelNum[label] = 1
            else:
                labelNum[label] += 1
        entropy = 0
        for num in labelNum:
            entropy -= num / len(dataList) * math.log(num / len(dataList))
    else:
        pass

def createTree(dataList, feature, val):
    right = []; left = []
    for data in dataList:
        if data[feature] == val:
            left.append(data)
        else:
            right.append(data)
    tree = {}
    tree['feature'] = feature
    tree['value'] = val
    tree["right"] = right
    tree["left"] = left
    return tree