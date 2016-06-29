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
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas , labelMat).T * (dataMatrix * dataMatrix[i, : ].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
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
                if L == H:  print("L == H");    continue
                eta = 2.0 * dataMatrix[i, : ] * dataMatrix[j, : ].T - dataMatrix[i, : ] * dataMatrix[i, : ].T - dataMatrix[j, : ] * dataMatrix[j, : ].T
                if eta >= 0:    print("eta >= 0");  continue
                alphas[j] -= labelMat[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):  print("j not moving enough"); continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[i, : ].T - labelMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[i, : ] * dataMatrix[j, : ].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, : ] * dataMatrix[j, : ].T - labelMat[j] * (alphas[j] - alphaJold) * \
                dataMatrix[j, : ] * dataMatrix[j, : ].T
                if (0 < alphas[i]) and (C > alphas[i]):     b = b1
                elif (0 < alphas[j]) and (C > alphas[i]):   b = b2
                else:   b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):    iter += 1
        else:   iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

def smoLearning(dataMatrix, labels, C, k):
    alpha = zeros(len(dataMatrix))
    dataMatrix = matrix(dataMatrix)
    b = 0
    u = multiply(labels, alpha) * (dataMatrix * dataMatrix.T) + b
    for i in range(k):
        for j in range(len(labels)):
            if ((labels[j] * u[0, j] < 1 and alpha[j] < C) or (
                labels[j] * u[0, j] > 1 and alpha[j] > 0
            )):
                a2 = alpha[j]
                a2Old = a2
                x2 = dataMatrix[j]
                y2 = labels[j]
                e2 = u[0, j] - labels[j]
                a1Max = 0
                for k in range(len(labels)):
                    e1Pre = u[0, k] - labels[k]
                    if abs(e1Pre - e2) > a1Max:
                        a1Max = abs(e1Pre -e2)
                        a1 = alpha[k]
                        e1 = e1Pre
                        a1Old = a1
                        a1K = k
                        x1 = dataMatrix[k]
                        y1 = labels[k]
                if (y1 != y2):
                    L = max(0, a2 - a1);    H = min(C, C + a2 -a1)
                else:
                    l = max(0, a2 + a1 -C);     H = min(C, a2 + a1)
                erta = 2 * x2 * matrix(x1).T - x1 * matrix(x1).T - x2 * matrix(x2).T
                a2 = a2 + y2 * (e1 - e2) / erta
                alpha[j] = a2
                if a2 > H:
                    a2 = H
                elif a2 < L:
                    a2 = L
                else:
                    pass
                a1 = a1 + y1 * y2 * (a2Old - a2)
                alpha[a1K] = a1
                b1 = b - e1 - y1 * (a1 - a1Old) * (x1 * x1.T) - y2 * (a2 - a2Old) * (x1 * x2.T)
                b2 = b - e2 - y1 * (a1 - a1Old) * (x1 * x2.T) - y2 * (a2 - a2Old) * (x2 * x2.T)
                if 0 < a1 and a1 < C:
                    b = b1
                elif 0 < a2 and a2 < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                u = multiply(labels, alpha) * (dataMatrix * dataMatrix.T) + b
    return b, alpha

def test():
    dataArr, labelArr = loadDataSet("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch06\\testSet.txt")
    #b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    b, alphas = smoLearning(dataArr, labelArr, 0.6, 40)
    print(alphas)

if __name__ == "__main__":
    test()