from math import log

def calcShannonEnt(dataSet):
    """
    计算熵
    :param dataSet: 数据集
    :return: float
    """
    numEntries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0
        labelsCounts[currentLabel] += 1
        shannonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no'],]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[: axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy =+ prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i
    return bestFeature