from numpy import *
import os

def preProcess(filename):
	print(filename)
	files = os.listdir(filename)
	labels = []
	dataMatrix = []
	for file in files:
		matrix = []
		labels.append(int(file.split("_")[0]))
		for line in open(filename + '//' + file):
			matrix.extend([int(val) for val in line.strip()])
		dataMatrix.append(matrix)
	return dataMatrix, labels

def knn(dataMatrix, label, k, dataImg):
	dataMatrix =  matrix(dataMatrix)
	dataMatrix = dataMatrix - dataImg
	distance = []
	for i in range(len(dataMatrix)):
		distance.append(int(dataMatrix[i] * dataMatrix[i].T))
	distance = list(argsort(array(distance)))
	distanceK = {}
	for i in range(k):
		if label[distance[i]] not in distanceK:
			distanceK[label[distance[i]]] = 1
		else :
			distanceK[label[distance[i]]] += 1
	return sorted(distanceK)[-1]


if __name__ == "__main__":
	dataMatrix, label = preProcess("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\trainingDigits")
	inMatrix, inLabel = preProcess("E:\学习资料\ml\MLiA_SourceCode\machinelearninginaction\Ch02\digits\\testDigits")
	errNum = 0
	for i in range(len(inMatrix)):
		preLabel = knn(dataMatrix, label, 5, inMatrix[i])
		print("The preLabel is %d, the true label is %d" % (preLabel, inLabel[i]))
		if inLabel[i] != preLabel:
			errNum += 1

	print(errNum / len(inMatrix))
#	preLabel = knn(dataMatrix, label, 20, inMatrix[800])
#	print(preLabel)