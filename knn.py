from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistIndecies = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndecies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(range, (m, 1))
    return normDataSet, range, minVals

def img2vector(filename):
    returnVector = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * i + j] = int(lineStr[j])
        return returnVector

def file2matrixDigit(filename):
    labels = []
    FileList = listdir("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\%s" % filename)
    m = len(FileList)
    mat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = FileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        labels.append(classNumStr)
        mat[i, : ] = img2vector("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\%s\%s" % (filename, fileNameStr))
    return mat, labels


def handwritingClassTester():
    hwLabels = []
    trainingMat, hwLabels = file2matrixDigit("trainingDigits")
    testingMat, testingLabels = file2matrixDigit("testDigits")
    mTest = len(testingLabels)
    errorCount = 0.0
    for i in range(mTest):
        classifierResult = classify0(testingMat[i, : ], trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testingLabels[i]))
        if (classifierResult != testingLabels[i]): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is : %f" % (errorCount/float(mTest)))

def BPneuroNet():
    pass

if __name__ == "__main__":
    hwLabels = []
    trainingFileList = listdir('E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\trainingDigits/%s' % fileNameStr)
    testFileList = listdir('E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

