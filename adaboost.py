from numpy import *

def loadSimpData():
    datMat = [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]]
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def buildStump(dataArr, classLabels):
    m, n = shape(dataArr)
    minError = inf
    for i in list(range(n)):
        featureValue = sorted(set([data[i] for data in dataArr]))
        print(featureValue)
        for val in list(featureValue):
            errorLt = 0; errorGt = 0
            listA = []; listB = []
            for data in dataArr:
                if (data[i] < val):
                    listA.append(dataArr.index(data))
                else:
                    listB.append(dataArr.index(data))
            for labels in listA:
                if classLabels[labels] != -1.0:
                    errorLt += 1.0
                if classLabels[labels] != 1.0:
                    errorGt += 1.0
            for labels in listB:
                if classLabels[labels] != 1.0:
                    errorLt += 1.0
                if classLabels[labels] != -1.0:
                    errorGt += 1.0
            print("errorLt: %f, errorGt: %f" % (errorLt, errorGt))
            if errorLt < minError:
                minError = errorLt
                bestStump = {"dim" : i, "thresh" : val, "inequal" : "lt", "error" : minError/5 }
            if errorGt < minError:
                minError = errorGt
                bestStump = {"dim": i, "thresh": val, "inequal": "lt", "error": minError / 5}
    return bestStump

def adaBoostTrainDS(dataArr, classLabels, nuIt = 40):
    pass

def buildStumpTest():
    datMat, classLabels = loadSimpData()
    bestStump = buildStump(datMat, classLabels)
    print(bestStump)

if __name__ == "__main__":
    buildStumpTest()