from math import log

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


def main():
    dataSet,labels = createDataSet()
    dataSet[0][-1] = 'maybe'

    print(calcShannonEnt(dataSet))


if __name__=='__main__':
    main()