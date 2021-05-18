import pandas as pd
import random
import numpy as np


def padArray(array, newLength, value=-1):
    array.extend([value for _ in range(newLength - len(array))])


def onehot(label, totalLabels):
    return [1.0 if i == label else 0.0 for i in range(totalLabels)]


def randomList(length, maxVal, exlcludedValues=None):
    if exlcludedValues is None:
        exlcludedValues = []
    init = [i for i in range(maxVal)]
    random.shuffle(init)
    index = 0
    end = length
    out = []
    while index < end:
        if init[index] not in exlcludedValues:
            out.append(init[index])
        else:
            end += 1
        index += 1
    return out


def listToOnehotMatrix(list, totalLabels):
    m = []
    for i in list:
        m.append(onehot(i, totalLabels))
    return np.array(m)


def categorize(table):
    colToCat = ["Game", 'ID']
    for col in colToCat:
        table[col] = pd.Categorical(table[col])
        table[col] = table[col].cat.remove_unused_categories()
        table[col] = table[col].cat.codes
    return table


def readCSV():
    csv = pd.read_csv("./Video_Games.csv", names=["Game", "ID", "Rating", "Timestamp"])
    csv = csv.sort_values('Timestamp')
    return csv


def filterColumn(table, col, key):
    return table[table[col] == key]


def getUserInteractions(table, user):
    return filterColumn(table, 'ID', user)


def getUniqueAsList(table, col):
    return table[col].unique()


def getTopNofCol(table, col, n):
    indices = table[col].value_counts()[:n].index.tolist()
    table = table[table[col].isin(indices)].copy()
    return table


def filterLowInteractions(table, col, minInteractions):
    filtered = table[table[col].isin(table[col].value_counts()[table[col].value_counts() >= minInteractions].index)].copy()
    return filtered


def maxDuplicate(table, column):
    return table[column].value_counts().max()


def perRowAverages(data):
    avg = []
    for d in data:
        avg.append(sum(d)/len(d))
    return avg


def removeDuplicateInteractions(table):
    ret = table.drop_duplicates(subset=['Game', 'ID'], keep='first')
    return ret


def test():
    c = readCSV()
    print(str(len(c)) + " | " + str(len(getUniqueAsList(c, 'ID'))) + " | " + str(len(getUniqueAsList(c, 'Game'))))
    c = getTopNofCol(c, 'Game', 1500)
    print(str(len(c)) + " | " + str(len(getUniqueAsList(c, 'ID'))) + " | " + str(len(getUniqueAsList(c, 'Game'))))
    c = filterLowInteractions(c, 'ID', 5)
    print(str(len(c)) + " | " + str(len(getUniqueAsList(c, 'ID'))) + " | " + str(len(getUniqueAsList(c, 'Game'))))
    print(len(getUniqueAsList(c, 'ID'))) 