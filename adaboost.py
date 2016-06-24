from numpy import *

def loadSimpData():
    datMat = [[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]]
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def buildStump(dataArr, classLabels, D):
    m, n = shape(dataArr)
    minError = inf
    for i in list(range(n)):
        featureValue = sorted(set([data[i] for data in dataArr]))
        for val in list(featureValue):
            labelEstListLt = ones((m, 1)); labelEstListGt = ones((m, 1))
            for data in dataArr:
                if (data[i] <= val):
                    labelEstListLt[dataArr.index(data)] = -1.0
            for data in dataArr:
                if (data[i] > val):
                    labelEstListGt[dataArr.index(data)] = -1.0
            errorEstLt = dot(array(D).T, array(labelEstListLt != mat(classLabels).T))
            errorEstGt = dot(array(D).T, array(labelEstListGt != mat(classLabels).T))
            if errorEstLt < minError:
                minError = errorEstLt
                bestStump = {"dim" : i, "thresh" : val, "inequal" : "lt", "error" : minError}
                bestEst = labelEstListLt
            if errorEstGt < minError:
                minError = errorEstGt
                bestStump = {"dim": i, "thresh": val, "inequal": "lt", "error": minError}
                bestEst = labelEstListGt
    return bestStump, bestEst

def adaBoostTrainDS(dataArr, classLabels, D, nuIt = 40):
    pass
    bestStrumpList = []
    m = shape(dataArr)[0]
    aggClassEst = zeros((m, 1))
    for i in range(nuIt):
        bestStrump, bestEst = buildStump(dataArr, classLabels, D)
        alpha = float(0.5 * log((1 - bestStrump["error"]) / max(bestStrump["error"], 1e-16)))
        bestStrump["alpha"] = alpha
        bestStrumpList.append(bestStrump)
        expon = multiply(-1 * alpha * mat(classLabels), bestEst.T)
        D = multiply(D, exp(expon).T)
        D = D/D.sum()
        aggClassEst += alpha * bestEst
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("D:",D.T)
        print("classEst:", bestEst.T)
        print("aggClassEst:", aggClassEst.T)
        print("total error:", errorRate.T)
        if errorRate == 0.0:  break
    return bestStrumpList

def buildStumpTest():
    datMat, classLabels = loadSimpData()
    D = ones((shape(datMat)[0], 1)) / 5
    bestStumpList = adaBoostTrainDS(datMat, classLabels, D, 9)
    #bestStumpList, bestEst = buildStump(datMat, classLabels, D)
    print(bestStumpList)
    #print(bestEst)
    #print(type(list(bestEst)))


if __name__ == "__main__":
    buildStumpTest()