from numpy import  *

def loadSimpleData():
    dataMat = [[1, 2.1],
                      [2, 1.1],
                      [1.3, 1],
                      [1, 1],
                      [2, 1]
                      ]
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def loadData(filename):
    dataList = [];  labelList = []
    for line in open("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch07\%s" % filename):
        line = [float(val) for val in line.strip().split()]
        dataList.append(line[:-1])
        labelList.append(line[-1])
    return dataList, labelList

def genTree(dataList, labelList, weight):
    ineq = ['lt', 'gt']
    classResult = ones(len(dataList))
    dataMatrix = matrix(dataList)
    labelMatrix = matrix(labelList)
    errMatrix = ones(len(dataList))
    errNum = len(dataList)
    tree = {}
    for i in range(len(dataList[0])):
        for j in set([val[i] for val in dataList]):
            for eq in ineq:
                errMatrix = ones(len(dataList))
                classResult = ones(len(dataList))
                if eq == 'lt':
                    for k in range(len(classResult)):
                        if dataMatrix[k, i] <= j:
                            classResult[k] = -1
                else:
                    for k in range(len(classResult)):
                        if dataMatrix[k, i] > j:
                            classResult[k] = -1
                for k in range(len(errMatrix)):
                    if labelMatrix[0, k] == classResult[k]:
                        errMatrix[k] = 0
                if errNum > errMatrix * matrix(weight).T:
                    errNum = errMatrix * matrix(weight).T
                    tree['errNum'] = errNum
                    tree['character'] = i
                    tree['threshold'] = j
                    tree['eq'] = eq
                    tree['alpha'] = 1/2 * log((1 - errNum)/errNum)
                    tree['bestPre'] = classResult.copy()
                    bestPredict = classResult
    return tree, errNum, bestPredict

def adaboost(dataList, labelList, D, iter):
    treeList = []
    classEst = zeros(len(dataList))
    for i in range(iter):
        tree, errNum, bestPredict = genTree(dataList, labelList, D)
        for j in range(len(D)):
            dSum = D.sum()
            if bestPredict[j] == labelList[j]:
                D[j] = D[j] * exp(-tree['alpha'])/dSum.sum()
            else:
                D[j] = D[j] * exp(tree['alpha']) / dSum.sum()
        treeList.append(tree)
        classEst = classEst + tree['alpha'] * bestPredict
        classEstNum = (sign(classEst) != matrix(labelList)).sum()
        #print("total ratio: %f" % (classEstNum / len(labelList)))
        if classEstNum == 0:
            break
    return treeList

if __name__ == "__main__":
    dataList, labelList = loadData("horseColicTraining2.txt")
    dataMatrix = matrix(dataList)
    D = ones(len(dataList)) / len(dataList)
    treeList = adaboost(dataList, labelList, D, 100)
    #print(treeList)
    testData, testLabel = loadData("horseColicTest2.txt")
    errNum = 0
    for i in range(len(testData)):
        pre = 0
        for tree in treeList:
            treePre = 1
            if tree['eq'] == 'lt':
                if testData[i][tree['character']] <= tree['threshold']:
                    treePre = -1
            else:
                if testData[i][tree['character']] > tree['threshold']:
                    treePre = -1
            pre = pre + tree['alpha'] * treePre
        print("the real is %f, and the predict is %d" % (testLabel[i], sign(pre)))
        if sign(pre) != testLabel[i]:
            errNum += 1
    print("erro ratio is %f" % (errNum/len(testData)))
