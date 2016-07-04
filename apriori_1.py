def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def getUnit(dataSet):
    unitSet = []
    for data in dataSet:
        for val in data:
            if val not in unitSet:
                unitSet.append(val)
    return unitSet

def filterData(dataSet, minSupport):
    uniSet = {}
    count = 0
    filterSet = []
    for data in dataSet:
        for val in data:
            if val not in uniSet:
                uniSet[val] = 1
            else:
                uniSet[val] += 1
    for val in uniSet:
        if uniSet[val] / len(dataSet) >= minSupport:
            filterSet.append([val])
    return filterSet

def contruList(dataSet, minSupport, filterSet):
    conSet = []
    for i in range(len(filterSet)):
        for j in range(i+1, len(filterSet)):
            if [val for val in set(filterSet[i]).union(set(filterSet[j]))] not in conSet:
                conSet.append([val for val in set(filterSet[i]).union(set(filterSet[j]))])
    conDict = {}
    for i in range(len(conSet)):
        for data in dataSet:
            if set(conSet[i]).intersection(set(data)) == set(conSet[i]):
                if i not in conDict:
                    conDict[i] = 1
                else:
                    conDict[i] += 1
    nextSet = []
    for val in conDict:
        if conDict[val] / len(dataSet) >= minSupport:
            nextSet.append(conSet[val])
    return nextSet

def apriori(dataSet, minsupport):
    uniSet = filterData(dataSet, 0.5)
    i = 0
    apriSet = []
    for val in uniSet:
        apriSet.append(val)
    while i < len(uniSet):
        con = contruList(dataSet, minsupport, uniSet)
        for val in con:
            apriSet.append(val)
        uniSet = con
        i += 1
    return apriSet

def genSingleRule(dataSet, data, minsupport):
    dataSetrule = []
    i = 0
    for val in data[0]:
        dataCopy = [val[:] for val in data]
        dataCopy[0].remove(val)
        dataCopy[1].append(val)
        data01 = dataCopy[0] + dataCopy[1]
        data0Num = 0
        data01Num = 0
        for dataVal in dataSet:
            if set(data01).intersection(set(dataVal)) == set(data01):
                data01Num += 1
            if set(dataCopy[0]).intersection(set(dataVal)) == set(dataCopy[0]):
               data0Num += 1
        if data01Num / data0Num >= minsupport:
            dataSetrule.append([dataCopy[0], dataCopy[1]])
    return dataSetrule

def genRule(dataSet, minConf, aprSet):
    aprFilterSet = [[val[:]] for val in aprSet if len(val) > 1]
    for val in aprFilterSet:
        val.append([])
    dataSetrule = []
    flag = False
    while len(aprFilterSet) > 0 and not flag:
        print(aprFilterSet, flag)
        flag = True
        for val in aprFilterSet:
            if len(val[0]) > 1:
                dataSingleSetRule = genSingleRule(dataSet, val, minConf)
            for val in dataSingleSetRule:
                flag = False
                dataSetrule.append([val[0][:], val[1][:]])
        aprFilterSet = [[aprVal[0][:], aprVal[1][:]] for aprVal in dataSingleSetRule if len(aprVal[0]) > 1]
    dataSetruleCopy = [[val[0][:], val[1][:]] for val in dataSetrule]
    print(dataSetruleCopy)
    for i in range(len(dataSetruleCopy)):
        for j in range(i+1, len(dataSetruleCopy)):
            if set(dataSetruleCopy[i][0]) == set(dataSetruleCopy[j][0]) and set(dataSetruleCopy[i][1]) == set(dataSetruleCopy[j][1]):
                dataSetrule.remove(dataSetruleCopy[j])
    return dataSetrule




if __name__ == '__main__':
    dataSet = loadDataSet()
    aprSet = apriori(dataSet, 0.5)
    print(genSingleRule(dataSet, [[2,3,5], []], 0.7))
    print(genRule(dataSet, 0.7, aprSet))
