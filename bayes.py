import numpy as np
from math import log
import random
import feedparser
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocablist(dataList):
    vocabSet = set([])
    for document in dataList:
        #vocabSet = vocabSet.union(set(document))
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('The word: %s is not in my Vocabulary!' % word)
    return returnVec


def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('The word: %s is not in my Vocabulary!' % word)
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/numTrainDocs
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


def testNB():
    listPosts, listClasses = loadDataSet()
    myVocablist = createVocablist(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love','my','dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocablist,testEntry))
    print(testEntry,'Classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocablist,testEntry))
    print(testEntry,'Classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok)>2]


def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList = textParse(open('Ch04/email/spam/%d.txt' % i,encoding='ISO-8859-15').read())
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
        wordList = textParse(open('Ch04/email/ham/%d.txt' % i,encoding='ISO-8859-15').read())
        docList.append(wordList)
        classList.append(0)
        fullText.extend(wordList)
    vocabList = createVocablist(docList)
    trainingSet = list(range(50))
    testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('The error rate is: ', errorCount / len(testSet))
    return errorCount / len(testSet)

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocablist(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(trainMat,trainClasses)
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('The error rate is: ', errorCount / len(testSet))
    return vocabList,p0V,p1V








def main():
    #testNB()
    # sum = 0.0
    # for i in range(10):
    #     sum+=spamTest()
    # print('The average rate is: ', sum/10)
    ny = feedparser.parse('https://www.nytimes.com/section/world?emc=rss&partner=rss')
    sf = feedparser.parse('https://www.nytimes.com/2018/09/12/world/europe/pope-bishops-conference.html?partner=rss&emc=rss')
    vocabList,pSF,pNY = localWords(ny,sf)

if __name__ == '__main__':
    main()


