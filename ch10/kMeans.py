from numpy import *

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssement = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 计算平均值作为第一个质心
    centList = [centroid0] # 储存质心
    for j in range(m):
        # 计算到质心的距离
        clusterAssement[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            # 获取质点的数据
            ptsInCurrCluster = dataSet[nonzero(clusterAssement[:, 0].A == i)[0], :]
            # 二分聚类
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算二分之后的sse
            sseSplit = sum(splitClustAss[:, 1])
            # 计算没有二分的sse
            sseNotSplit = sum(clusterAssement[nonzero(clusterAssement[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewsCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
                bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
                print("the bestCentToSplit is: ", bestCentToSplit)
                print("the len of bestClustAss is: ", len(bestClustAss))
                centList[bestCentToSplit] = bestNewsCents[0, :]
                centList.append(bestNewsCents[1, :])
                clusterAssement[nonzero(clusterAssement[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return centList, clusterAssement

if __name__ == "__main__":
    dataSet = mat(loadDataSet("testSet2.txt"))
    centList, clusterAssment = biKmeans(dataSet, 3)




