from numpy import *

def loadDataSet(filenName):
    dataMat = []; labelMat = []
    fr = open(filenName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while (j == i):
       j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    testI = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas , labelMat).T * (dataMatrix * dataMatrix[i, : ].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                testI += 1
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, : ].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:  #print("L == H");
                    continue
                eta = 2.0 * dataMatrix[i, : ] * dataMatrix[j, : ].T - dataMatrix[i, : ] * dataMatrix[i, : ].T - dataMatrix[j, : ] * dataMatrix[j, : ].T
                if eta >= 0:    #print("eta >= 0");
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):  #print("j not moving enough");
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[i, : ].T - labelMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[i, : ] * dataMatrix[j, : ].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[j, : ].T - labelMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[j, : ] * dataMatrix[j, : ].T
                if (0 < alphas[i]) and (C > alphas[i]):     b = b1
                elif (0 < alphas[j]) and (C > alphas[i]):   b = b2
                else:   b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                #print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))

        if (alphaPairsChanged == 0):    iter += 1
        else:   iter = 0
        #print("iteration number: %d" % iter)
    return b, alphas

def smoLearning(data, labels, C, k, tol):
    alpha = zeros(len(data))
    b = 0
    dataMatrix = matrix(data)
    u = multiply(labels, alpha) * (dataMatrix * dataMatrix.T) + b
    iterNum = 0
    while iterNum < k:
        ChangedNum = 0
        for i in range(len(data)):
            if (labels[i] * u[0, i] < 1 - tol and alpha[i] < C) or (labels[i] * u[0, i] > 1 + tol and alpha[i] > 0):
                a1Num = i
                alphaIold = alpha[a1Num]
                e1 = u[0, a1Num] - labels[a1Num]
                maxE = 0
                for j in range(len(data)):
                    ej = u[0, j] - labels[j]
                    if abs(e1 - ej) > maxE:
                        maxE = abs(e1 - ej)
                        a2Num = j
                        alphaJold = alpha[a2Num]
                        e2 = u[0, a2Num] - labels[a2Num]
                if labels[a1Num] != labels[a2Num]:
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaJold + alphaIold - C)
                    H = min(C, alphaJold + alphaIold)
                if L == H:
                    continue
                eta = 2 * dataMatrix[a1Num] * dataMatrix[a2Num].T - dataMatrix[a1Num] * dataMatrix[a1Num].T - dataMatrix[a2Num] * dataMatrix[a2Num].T
                if eta == 0:
                    continue
                a2 = alphaJold - labels[a2Num] * (e1 - e2) / eta
                if a2 > H:
                    a2 = H
                elif e2 < L:
                    e2 = L
                else:
                    pass
                a1 = alphaIold + labels[a1Num] * labels[a2Num] * (alphaJold - a2)
                b1 = b - e1 - labels[a1Num] * (a1 - alphaIold) * (dataMatrix[a1Num] * dataMatrix[a1Num].T) - labels[a2Num] * (a2 - alphaJold) * (
                    dataMatrix[a1Num] * dataMatrix[a2Num].T)
                b2 = b - e2 - labels[a1Num] * (a1 - alphaIold) * (dataMatrix[a1Num] * dataMatrix[a2Num].T) - labels[a2Num] * (a2 - alphaJold) * (
                    dataMatrix[a2Num] * dataMatrix[a2Num].T
                )
                if a1 > 0 and a1 < C:
                    b = b1
                elif a2 > 0 and b2 < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alpha[a1Num] = a1;  alpha[a2Num] = a2
                u = multiply(labels, alpha) * (dataMatrix * dataMatrix.T) + b
                ChangedNum += 1
        iterNum += 1
        if ChangedNum == 0:
            break
    return b, alpha

def test():
    dataArr, labelArr = loadDataSet("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch06\\testSet.txt")
    #b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b, alphas = smoLearning(dataArr, labelArr, 0.6, 30000, 0.001)
    print(alphas)

if __name__ == "__main__":
    test()
    """
    0.20360655  0.          0.09201711  0.04869433  0.05411643 -0.04192737
  0.03698655  0.23673997  0.32260068  0.          0.15746858  0.12894878
  0.          0.          0.          0.          0.          0.6
  0.04539632  0.          0.          0.14141348  0.04557592  0.25684892
  0.          0.0937702   0.20148801  0.         -0.934729    0.4600151   0.
  0.09804829  0.          0.          0.10484748  0.          0.          0.
  0.          0.49887861  0.          0.         -0.81590348  0.          0.
  0.         -0.02055148  0.03358945  0.          0.          0.         -0.0581367
  0.52021494  0.          0.24948877  0.6         0.134687    0.06685472
  0.          0.         -0.01751268 -0.20004993  0.18002879  0.13105508
          """