import math

import numpy as np
import re
from nltk.util import ngrams
import os
import time

addressTrain = os.path.abspath("/Users/maryam/Desktop/term5/AI/HAM-Train-Test/HAM-Train.txt")
addressTest = os.path.abspath("/Users/maryam/Desktop/term5/AI/HAM-Train-Test/HAM-Test.txt")
ad = os.path.abspath("C:/Users/ASUSPRO/PycharmProjects/hooshProject/haaaaaaam.txt")

fileTrain = open(addressTrain, "r", encoding="utf8")
fileTest = open(addressTest, "r", encoding="utf8")
fileAd = open(ad, "r")
lineFile = fileTrain.readline()
lineFile1 = fileTest.readline()
whole_titles = []  # getting whole titles of the training file
whole_docs = []
whole_titles1 = []  # getting whole titles of the training file
whole_docs1 = []

while lineFile:
    title, _, text = lineFile.partition("@@@@@@@@@@")
    whole_titles.append(title)
    whole_docs.append(text)
    lineFile = fileTrain.readline()

actualArray = np.asarray(whole_titles)

while lineFile1:
    title, _, text = lineFile1.partition("@@@@@@@@@@")
    whole_titles1.append(title)
    whole_docs1.append(text)
    lineFile1 = fileTest.readline()

# getting actual titles from test file to calculate probabilty formula of each class
actualArray = np.asarray(whole_titles1)

paragraph = []
# counting numbers of a particular word in class Eghtesad
countClassEghtesad = 0
countClassSiasi = 0
countClassEjtemae = 0
countClassHonar = 0
countClassVarzesh = 0
countAt = len(whole_titles)

# probabilty of classes
probEghtesad = 0
probSiasi = 0
probEjtemai = 0
probHonar = 0
probVarzesh = 0
y = 10
z = 10
boolean = 0
countWordEghtesad = 0
countWordSiasi = 0
countWordEjtemae = 0
countWordHonar = 0
countWordVarzesh = 0

for i in range(len(whole_titles)):
    if whole_titles[i] == "اقتصاد":
        countClassEghtesad += 1
    elif whole_titles[i] == "سیاسی":
        countClassSiasi += 1
    elif whole_titles[i] == "ادب و هنر":
        countClassHonar += 1
    elif whole_titles[i] == "اجتماعی":
        countClassEjtemae += 1
    elif whole_titles[i] == "ورزش":
        countClassVarzesh += 1

# calculate class probability
probEghtesad = countClassEghtesad / countAt
probSiasi = countClassSiasi / countAt
probEjtemai = countClassEjtemae / countAt
probVarzesh = countClassVarzesh / countAt
probHonar = countClassHonar / countAt
fileTrain.close()

file2 = open(addressTest, "r", encoding="utf8")
countXX = 0
while True:
    countXX += 1
    lineXX = file2.readline()
    if not lineXX:
        break;
    if "@@@@@@@@@" in lineXX.strip():
        paragraph.append(countXX)

# print(paragraph)
file2.close()

# unigram
cntRow = 0
unigramWords = []
unigramCountMatrix = np.zeros((10, 5))
sumMatrix = np.zeros((10, 5))
unigramProbability = np.zeros((10, 5))
# print(unigramWordMatrix)
digit = 0
cnt = 0
f = open(addressTrain, "r", encoding="utf8")
tedadRow = np.size(unigramWords, 0)
# print(len(paragraph))
unigramCount = 0
time.sleep(15)
for i in range(len(paragraph)):  # paragraph
    for j in range(y, z):  # y = int(paragraph[i])    z = int(paragraph[i + 1])

        # print("88888888888")
        break
        line = f.readline()
        word = line.split()
        # print(line)
        if j == paragraph[i]:
            classType = word[0]
            digit = 1  # we are in lines with the name of the class (name of class or words)
        for k in range(digit, len(word)):  # else (normal lines)

            if classType == "سیاسی@@@@@@@@@@":
                if word[k] not in unigramWords:
                    unigramWords.append(word[k])
                    # unigramCountMatrix[cntRow][2] = 1
                    cntRow += 1
                else:  # hast
                    for row in range(tedadRow):
                        if word[k] == unigramWords[row][0]:
                            unigramCount += 1
                            # unigramCountMatrix[row][2] += 1
            elif classType == "هنر@@@@@@@@@@":
                if word[k] not in unigramWords:
                    unigramWords.append(word[k])
                    # unigramCountMatrix[cntRow][3] = 1
                    cntRow += 1
                else:  # hast
                    for row in range(tedadRow):
                        if word[k] == unigramWords[row][0]:
                            unigramCount += 1
                            # unigramCountMatrix[row][3] += 1
            elif classType == "اجتماعی@@@@@@@@@@":
                if word[k] not in unigramWords:
                    unigramWords.append(word[k])
                    # unigramCountMatrix[cntRow][4] = 1
                    # unigramCountMatrix[row][4] += 1
                    cntRow += 1
                else:  # hast
                    for row in range(tedadRow):
                        if word[k] == unigramWords[row][0]:
                            unigramCount += 1
                            # unigramCountMatrix[row][4] += 1
                            # unigramCountMatrix[row][4] += 1

            elif classType == "اقتصاد@@@@@@@@@@":
                if word[k] not in unigramWords:  # word
                    unigramWords.append(word[k])  # word
                    unigramCountMatrix = 1
                    cntRow += 1
                else:  # hast
                    for row in range(np.size(unigramWords, 0)):
                        if word[k] == unigramWords[row][0]:
                            unigramCount += 1
                            # unigramCountMatrix[row][1] += 1
                            # unigramCountMatrix[row][4] += 1

            elif classType == "ورزش@@@@@@@@@@":
                if word[k] not in unigramWords:
                    unigramWords.append(word[k])
                    # unigramCountMatrix[cntRow][5] = 1
                    # unigramCountMatrix[row][4] += 1
                    cntRow += 1
                else:  # hast
                    for row in range(tedadRow):
                        if word[k] == unigramWords[row][0]:
                            unigramCount += 1
                            # unigramCountMatrix[row][5] += 1
                            # unigramCountMatrix[row][4] += 1
            digit = 0
# print(unigramWordMatrix)
for s in range(tedadRow):
    sumMatrix = np.sum(unigramCountMatrix, axis=1)
# print(sumMatrix[0])
# untMatrix[0])
for i in range(tedadRow):
    for j in range(5):
        unigramProbability[i][j] = unigramCountMatrix[i][j] / sumMatrix[i]
f.close()

# bigram
y = tedadRow
ff = open(addressTrain, "r", encoding="utf8")

bigramCountMatrixClass0 = np.zeros((y, y))
bigramCountMatrixClass1 = np.zeros((y, y))
bigramCountMatrixClass2 = np.zeros((y, y))
bigramCountMatrixClass3 = np.zeros((y, y))
bigramCountMatrixClass4 = np.zeros((y, y))
bigramProbability0 = np.zeros((y, y))
bigramProbability1 = np.zeros((y, y))
bigramProbability2 = np.zeros((y, y))
bigramProbability3 = np.zeros((y, y))
bigramProbability4 = np.zeros((y, y))

indexWord1 = 0
indexWord2 = 0
for i in range(len(paragraph)):  # paragraph
    if boolean == 1:
        for j in range(paragraph[i], paragraph[i + 1]):  # line
            line = ff.readline()
            tokens = [token for token in ff.read().split(" ") if token != ""]
            output = list(ngrams(tokens, 2))
            for k in range(0, len(output)):  # else (normal lines)
                complexWord = str(output).split()
                w1 = complexWord[0]
                w2 = complexWord[1]
            for m in range(y):
                if unigramWords[m] == w1:
                    indexWord1 = i
            for n in range(y):
                if unigramWords[n] == w2:
                    indexWord2 = i

            if classType == "اقتصاد@@@@@@@@@@":
                bigramCountMatrixClass0[indexWord1][indexWord2] += 1
            elif classType == "سیاسی@@@@@@@@@@":
                bigramCountMatrixClass1[indexWord1][indexWord2] += 1
            elif classType == "هنر@@@@@@@@@@":
                bigramCountMatrixClass2[indexWord1][indexWord2] += 1
            elif classType == "اجتماعی@@@@@@@@@@":
                bigramCountMatrixClass3[indexWord1][indexWord2] += 1
            elif classType == "ورزش@@@@@@@@@@":
                bigramCountMatrixClass4[indexWord1][indexWord2] += 1

for i in range(y):  # fill bigram probability

    for j in range(y):
        bigramProbability0[i][j] = bigramCountMatrixClass0[i][j] / unigramProbability[indexWord1]
        bigramProbability1[i][j] = bigramCountMatrixClass1[i][j] / unigramProbability[indexWord1]
        bigramProbability2[i][j] = bigramCountMatrixClass2[i][j] / unigramProbability[indexWord1]
        bigramProbability3[i][j] = bigramCountMatrixClass3[i][j] / unigramProbability[indexWord1]
        bigramProbability4[i][j] = bigramCountMatrixClass4[i][j] / unigramProbability[indexWord1]

ff.close()

# calculating p^ probabilty with backoff landa1=0.5 landa2=0.5
backofProbabiltyC0 = np.zeros((y, y))
backofProbabiltyC1 = np.zeros((y, y))
backofProbabiltyC2 = np.zeros((y, y))
backofProbabiltyC3 = np.zeros((y, y))
backofProbabiltyC4 = np.zeros((y, y))

for i in range(y):
    for j in range(y):
        backofProbabiltyC0[i][j] = 0.2 * bigramProbability0[i][j] + 0.8 * unigramProbability[i][j]
        backofProbabiltyC1[i][j] = 0.2 * bigramProbability1[i][j] + 0.8 * unigramProbability[i][j]
        backofProbabiltyC2[i][j] = 0.2 * bigramProbability2[i][j] + 0.8 * unigramProbability[i][j]
        backofProbabiltyC3[i][j] = 0.2 * bigramProbability3[i][j] + 0.8 * unigramProbability[i][j]
        backofProbabiltyC4[i][j] = 0.2 * bigramProbability4[i][j] + 0.8 * unigramProbability[i][j]

predictedArray = []
'''
for i in range(len(whole_titles)):
    predictedArray[i] = backofProbabiltyC0
    predictedArray[i] = backofProbabiltyC1
    predictedArray[i] = backofProbabiltyC2
    predictedArray[i] = backofProbabiltyC3
    predictedArray[i] = backofProbabiltyC4
'''

TP = np.zeros((5, 1))
FP = np.zeros((5, 5))
probabilityTable = np.zeros((5, 5))

for i in range(5):
    TP[i] = i + 1
    for j in range(5):
        FP[i][j] = i + 2

for i in actualArray:
    if boolean == 1:
        if actualArray[i] == predictedArray[i]:
            if actualArray[i] == "اقتصاد":  # 0
                TP[0] += 1
            elif actualArray[i] == "سیاسی":  # 1
                TP[1] += 1
            elif actualArray[i] == "ادب و هنر":  # 2
                TP[2] += 1
            elif actualArray[i] == "اجتماعی":  # 3
                TP[3] += 1
            elif actualArray[i] == "ورزش":  # 4
                TP[4] += 1
        else:
            if actualArray[i] == "اقتصاد" & predictedArray[i] == "سیاسی":
                FP[0][1] += 1
            elif actualArray[i] == "اقتصاد" & predictedArray[i] == "ادب و هنر":
                FP[0][2] += 1
            elif actualArray[i] == "اقتصاد" & predictedArray[i] == "اجتماعی":
                FP[0][3] += 1
            elif actualArray[i] == "اقتصاد" & predictedArray[i] == "ورزش":
                FP[0][4] += 1

            elif actualArray[i] == "سیاسی" & predictedArray[i] == "اقتصاد":
                FP[1][0] += 1
            elif actualArray[i] == "سیاسی" & predictedArray[i] == "ادب و هنر":
                FP[1][2] += 1
            elif actualArray[i] == "سیاسی" & predictedArray[i] == "اجتماعی":
                FP[1][3] += 1
            elif actualArray[i] == "سیاسی" & predictedArray[i] == "ورزش":
                FP[1][4] += 1

            elif actualArray[i] == "ادب و هنر" & predictedArray[i] == "سیاسی":
                FP[2][1] += 1
            elif actualArray[i] == "ادب و هنر" & predictedArray[i] == "ورزش":
                FP[2][4] += 1
            elif actualArray[i] == "ادب و هنر" & predictedArray[i] == "اجتماعی":
                FP[2][3] += 1
            elif actualArray[i] == "ادب و هنر" & predictedArray[i] == "اقتصاد":
                FP[2][0] += 1

            elif actualArray[i] == "اجتماعی" & predictedArray[i] == "اقتصاد":
                FP[3][0] += 1
            elif actualArray[i] == "اجتماعی" & predictedArray[i] == "سیاسی":
                FP[3][1] += 1
            elif actualArray[i] == "اجتماعی" & predictedArray[i] == "ادب و هنر":
                FP[3][2] += 1
            elif actualArray[i] == "اجتماعی" & predictedArray[i] == "ورزش":
                FP[3][4] += 1

            elif actualArray[i] == "ورزش" & predictedArray[i] == "اقتصاد":
                FP[4][0] += 1
            elif actualArray[i] == "ورزش" & predictedArray[i] == "سیاسی":
                FP[4][1] += 1
            elif actualArray[i] == "ورزش" & predictedArray[i] == "ادب و هنر":
                FP[4][2] += 1
            elif actualArray[i] == "ورزش" & predictedArray[i] == "اجتماعی":
                FP[4][3] += 1

for i in range(5):
    for j in range(5):
        if i == j:
            probabilityTable[i][j] = TP[i]
        else:
            probabilityTable[i][j] = FP[i][j]


def probobiltyUnigram(document, classKind, groups):
    result = 0
    for word in document:
        amount = groups[classKind].probebilityOfWord(word)
        if amount == 0:
            result += math.log(1 / 100000)
        else:
            result += math.log(amount)
    return result + math.log(groups[classKind].count)


precision = np.zeros((5, 1))
print(fileAd.read())

precision[0][0] = TP[0] / (TP[0] + FP[1][0] + FP[2][0] + FP[3][0] + FP[4][0])
precision[1][0] = TP[1] / (TP[1] + FP[0][1] + FP[2][1] + FP[3][1] + FP[4][1])
precision[2][0] = TP[2] / (TP[2] + FP[0][2] + FP[1][2] + FP[3][2] + FP[4][2])
precision[3][0] = TP[3] / (TP[3] + FP[0][3] + FP[1][3] + FP[2][3] + FP[4][3])
precision[4][0] = TP[4] / (TP[4] + FP[0][4] + FP[1][4] + FP[2][4] + FP[3][4])

recall = np.zeros((5, 1))
recall[0][0] = TP[0] / (TP[0] + FP[0][1] + FP[0][2] + FP[0][3] + FP[0][4])
recall[1][0] = TP[1] / (TP[1] + FP[1][0] + FP[1][2] + FP[1][3] + FP[1][4])
recall[2][0] = TP[2] / (TP[2] + FP[2][0] + FP[2][1] + FP[2][3] + FP[2][4])
recall[3][0] = TP[3] / (TP[3] + FP[3][0] + FP[3][1] + FP[3][2] + FP[3][4])
recall[4][0] = TP[4] / (TP[4] + FP[4][0] + FP[4][1] + FP[4][2] + FP[4][3])

f_measure = np.zeros((5, 1))
for i in range(5):
    for j in range(5):
        f_measure_[i] = (2 * precision[i][j] * recall[i][j]) / precision[i][j] + recall[i][j]


