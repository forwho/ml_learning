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

if __name__ == "__main__":
    dataMatrix, labels = proPrecess("E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch06\digits\\testDigits")
    print(labels)