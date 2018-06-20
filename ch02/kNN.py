from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    k 临近算法
    计算距离 排序 取第一个
    :param inX: 需要计算的数组
    :param dataSet:  数据
    :param labels: array 所有类别
    :param k: 取距离前k
    :return:
    """
    dataSetSize = dataSet.shape[0]
    # 生成和dataSet一样的矩阵
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() # argSort() 返回输入数组排序后的下标
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLine = fr.readlines()
    numberOfLine = len(arrayOLine)
    returnMat = zeros((numberOfLine, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLine:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
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
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs: m], 5)
        print("the classifier came back with : %d, the real answer is : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]) :
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./data/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./data/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('./data/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector('./data/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with : %d, the real answer is : %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr) : errorCount += 1.0
    print("the total error count is: %f" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))



if __name__ == '__main__':
    datingClassTest()