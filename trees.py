from math import log
import operator
import treePlotter

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel,0)+1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob*log(prob,2)
    return shannonEnt


def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels


def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueValues = set(featList)
        newEntropy = 0.0
        for value in uniqueValues:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/len(dataSet)
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntropy
        if infoGain>bestInfoGain :
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCnt = {}
    for vote in classList:
        classCnt[vote] = classCnt.get(vote,0)+1
    sortedClassCount = sorted(classCnt.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    newLabels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(newLabels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = newLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)




def main():
    # dataSet,labels = createDataSet()
    # print(labels)
    # newLabels = labels[:]
    # myTree = createTree(dataSet,newLabels)
    # print(classify(myTree,labels,[1,0]))
    # print(classify(myTree,labels,[1,1]))
    # storeTree(myTree,'classfifierStorage.pickle')
    # tree = grabTree('classfifierStorage.pickle')
    # print(tree)
    #print(chooseBestFeatureToSplit(dataSet))
    #dataSet[0][-1] = 'maybe'
    # print(calcShannonEnt(dataSet))
    # print(splitDataSet(dataSet,0,1))
    # print(splitDataSet(dataSet,0,0))
    fr = open('Ch03/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']

    lensesTree = createTree(lenses,lensesLabels)
    storeTree(lensesTree,'lensesTree.pickle')
    lensesTree = grabTree('lensesTree.pickle')
    testVec=['presbyopic','myope','no','reduced']
    result = classify(lensesTree,lensesLabels,testVec)
    print(result)
    treePlotter.createPlot(lensesTree)


if __name__=='__main__':
    main()