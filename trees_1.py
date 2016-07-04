#todo
import math

def loadData(filename):
    dataList = []
    for line in open(filename):
        line = line.strip().split('\t')
        data = line[:3]
        label = ' '.join(line[3:])
        data.append(label)
        dataList.append(data)
    print(dataList)
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
        return dataList, features
    if len(features) == 0:
        return dataList, features
    maxEntrophyAdd = 0
    subTree = {}
    for feature in features:
        for val in set([value[feature] for value in dataList]):
            EntrophyAdd, right, left = calEntropyAdd(dataList, val, feature)
            if EntrophyAdd >= maxEntrophyAdd:
                maxEntrophyAdd = EntrophyAdd
                subTree['feature'] = feature;   subTree['val'] = val;   subTree['left'] = left;     subTree['right'] = right
            """
    if maxEntrophyAdd < 0.1:
        return dataList, features
        """
    features.remove(subTree['feature'])
    return subTree, features, subTree['feature']

def createTree(dataList, features, child):
    tree, features, delFeature = createSubTree(dataList, features)
    children = child
    if len(set([val[-1] for val in tree['left']])) > 1 and len(features) > 0:
        tree['left'], children = createTree(tree['left'], features, 'left')
    if (children == 'left'):
        features.append(delFeature)
    if len(set([val[-1] for val in tree['right']])) > 1 and len(features) > 1:
        tree['right'], children = createTree(tree['right'], features, 'right')
    print(tree)
    return tree, child

if __name__ == "__main__":
    dataList = loadData("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch03\lenses.txt")
    features = list(range(len(dataList[0]) - 1))
    tree = createTree(dataList, features, 'parent')