from math import *
import numpy as np
def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr = open('Ch05/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmod(inX):
    return 1/(1+np.exp(-inX))

def gradAscent(dataMat,classLabels):
    dataMatrix = np.mat(dataMat)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmod(dataMatrix*weights)
        error = (labelMat-h)
        weights = weights+alpha*dataMatrix.transpose()*error
    return weights


def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat = loadDataSet()
    dataArr= np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1=[]
    xcord2=[]
    ycord1=[]
    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=20,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent(dataMatric,classLabels):
    m,n = np.shape(dataMatric)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h=sigmod(sum(dataMatric[i]*weights))
        error = classLabels[i]-h
        weights = weights+alpha*error*dataMatric[i]
    return weights


def stocGradAscent1(dataMatric,classLabels,numIter=1500):
    import random
    m,n = np.shape(dataMatric)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex=list(np.arange(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h=sigmod(sum(dataMatric[randIndex]*weights))
            error = classLabels[randIndex]-h
            weights = weights+alpha*error*dataMatric[randIndex]
            #dataIndex = np.delete(dataIndex,randIndex)
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX,weights):
    prob = sigmod(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('Ch05/horseColicTraining.txt')
    frTest = open('Ch05/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),trainingLabels,2000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate = errorCount/numTestVec
    print('The error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests  = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the average error rate is: %f' %(numTests,errorSum/numTests))


def main():
    #dataMat,labelMat = loadDataSet()
    #weights = gradAscent(dataMat,labelMat)
    #weights = stocGradAscent1(np.array(dataMat),labelMat)
    # print(weights)
    #plotBestFit(np.array(weights))
    multiTest()


if __name__=='__main__':
    main()