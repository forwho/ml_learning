import re
import random
from numpy import *
import math

def textParsing(bigString):
    wordList = re.split(r'\W*', bigString)
    return [word.lower() for word in wordList if len(word) > 2]

def string2vec(wordList, filename):
    wordVec = zeros(len(wordList))
    wordData = textParsing(open(filename).read())
    for val in wordData:
        wordVec[wordList.index(val)] += 1

    return wordVec

def crossVal(k):
    testSet = []
    wordList = []
    for i in range(k):
        num = random.randint(1,50)
        while num in testSet:
            num = random.randint(1,50)
        testSet.append(num)
    for i in range(1,51):
        if i < 26:
            wordList.extend(textParsing(open(
                "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spa\%d.txt" % i).read()))
        else:
            wordList.extend(textParsing(open(
                "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spam\%d.txt" % (i-25)).read()))
    return set(wordList), testSet

def priorP(wordList, testSet):
    p1 = 0; p2 = 0; p1Num = len(wordList); p2Num = len(wordList)
    wordP1 = ones(len(wordList)); wordP2 = ones(len(wordList))
    for i in range(1, 51):
        if i not in testSet:
            if i < 26:
                p1 += 1; p1Num += len(textParsing(open(
                    "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spa\%d.txt" % i).read()))
                wordP1 += string2vec(wordList,
                                     "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spa\%d.txt" % i)
            else:
                p2 += 1; p2Num += len(textParsing(open(
                    "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spam\%d.txt" % (i-25)).read()))
                wordP2 += string2vec(wordList,
                                     "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spam\%d.txt" % (i-25))
    p1 = p1 / (50 - len(testSet)); p2 = p2 / (50 - len(testSet))
    wordP1 = wordP1 / p1Num; wordP2 = wordP2 / p2Num
    return p1, p2, wordP1, wordP2

def bayesClassfy(wordList, p1, p2, wordP1, wordP2, testSet):
    erroNum = 0
    for i in testSet:
        if i < 26:
            print("The mail is not garbage")
            wordData = textParsing(open(
                "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spa\%d.txt" % i).read())
            class1 = p1; class2 = p2
            for val in wordData:
                class1 = class1 * wordP1[wordList.index(val)] * 100
                class2 = class2 * wordP2[wordList.index(val)] * 100
            if math.log(class1) > math.log(class2):
                print("Forecast:The mail is not garbage")
            else:
                print("Forecast:The mail is garbage")
                erroNum += 1
        else:
            print("The mail is garbage")
            wordData = textParsing(open(
                "E:\machine learning\codeReg\code\MLiA_SourceCode\machinelearninginaction\Ch04\email\spam\%d.txt" % (i-25)).read())
            class1 = p1
            class2 = p2
            for val in wordData:
                class1 = class1 * wordP1[wordList.index(val)] * 100
                class2 = class2 * wordP2[wordList.index(val)] * 100
            if math.log(class1) > math.log(class2):
                print("Forecast:The mail is not garbage")
                erroNum += 1
            else:
                print("Forecast:The mail is garbage")
    print("The error number is %d" % erroNum)

if __name__ == "__main__":
    wordList, testSet = crossVal(20)
    p1, p2, wordP1, wordP2 = priorP(list(wordList), testSet)
    bayesClassfy(list(wordList), p1, p2, wordP1, wordP2, testSet)
