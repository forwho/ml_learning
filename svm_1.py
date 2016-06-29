from numpy import *
import os

def proPrecess(filename):
    print(filename)
    files = os.listdir(filename)
    labels = []
    dataMatrix = []
    for file in files:
        matrix = []
        if int(file.split("_")[0]) == 1:
            labels.append(1)
        else:
            labels.append(-1)
        for line in open(filename + '//' + file):
            matrix.extend([int(val) for val in line.strip()])
        dataMatrix.append(matrix)
    return dataMatrix, labels

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

def svmTest(alpha, b, data):
    if (W * matrix(data).T + b) >= 0:
        return 1
    else:
        return  -1

if __name__ == "__main__":
    dataMatrix, labels = proPrecess("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch06\digits\\testDigits")
    testData, testLabels = proPrecess("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch06\digits\\trainingDigits")
    b, alpha = smoLearning(dataMatrix, labels, 0.6, 40)
    print(alpha[alpha > 0])
    print(b)
    erroNum = 0
    for data in testData:
        print("The real number is %d, and we forcast is %d" % (testLabels[testData.index(data)], svmTest(W, b, data)))
        if svmTest(alpha, b, data) != testLabels[testData.index(data)]:
            erroNum += 1
    print(erroNum / len(testLabels))