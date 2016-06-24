import math

def loadData(filename):
    dataList = []
    labelList = []
    for line in open(filename):
        line = line.strip().split('\t')
        dataList.append(line[0:-1])
    return dataList

def calEntropy(dataList):
    labelNum = {}
    for label in [val[-1] for val in dataList]:
        if label not in labelNum:
            labelNum[label] = 1
        else:
            labelNum[label] += 1
    entropy = 0
    for num in labelNum.values():
        entropy -= num / len(dataList) * math.log(num / len(dataList))
    return entropy

def calEntropyAdd(dataList, val, feature):
    rawEntrophy = calEntropy(dataList)
    right = []; left = []
    for data in dataList:
        if data[feature] == val:
            left.append(data)
        else:
            right.append(data)
    newEntrophy = len(right) / len(dataList) * calEntropy(right) + len(left) / len(dataList) * calEntropy(left)
    return rawEntrophy - newEntrophy, right, left

def createSubTree(dataList, features):
    if len(set(dataList[-1])) == 1:
        return dataList, featrures
    if len(features) == 1:
        return dataList, features
    maxEntrophyAdd = 0
    subTree = {}
    for feature in features:
        for val in set(dataList[feature]):
            EntrophyAdd, right, left = calEntropyAdd(dataList, val, feature)
            if EntrophyAdd > maxEntrophyAdd:
                maxEntrophyAdd = EntrophyAdd
                subTree['feature'] = feature;   subTree['val'] = val;   subTree['left'] = left;     subTree['right'] = right
            """
    if maxEntrophyAdd < 0.1:
        return dataList, features
        """
    else:
        features.remove(subTree['feature'])
        return subTree, features

def createTree(dataList, features):
    tree, features = createSubTree(dataList, features)
    if len(set([val[-1] for val in tree['left']])) > 1 and len(features) > 1:
        tree['left'] = createTree(tree['left'], features)
    if len(set([val[-1] for val in tree['right']])) > 1 and len(features) > 1:
        tree['right'] = createTree(tree['right'], features)
    return tree

if __name__ == "__main__":
    dataList = loadData("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch03\lenses.txt")
    features = list(range(len(dataList[0]) - 1))
    tree = createTree(dataList, features)
    print(tree)