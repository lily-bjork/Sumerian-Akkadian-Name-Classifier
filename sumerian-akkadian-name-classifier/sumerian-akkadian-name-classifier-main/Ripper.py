import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import Counter
import json
import numpy as np
import os
import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import wittgenstein as lw

SUMERIAN = '1'
AKKADIAN = '2'

# return true if the sign/pair does not contain illegal characters
# otherwise return false
def validSign(sign):
    return re.search('(!|\?|<|>|[|]|x|X)', str(sign)) == None

# returns a list of the x most frequent signs/pairs
# if max = 0, then it returns a list of every unique signs/pairs
def getMostFreqSigns(signsList, max):
    if max == 0:
        return list(dict.fromkeys(signsList))
    else:
        signTotal = Counter(signsList).most_common(max)
        signs = []
        for sign in signTotal:
            signs.append(sign[0])
        return signs

# get all signs in the files
# returns two lists of Akkadian or Sumerian signs containing duplicates
def getSigns(files):
    signs = [[],[]]
    i = 0
    for file in files:
        for name in file:
            name = re.sub('DINGIR', 'dinger', name)
            name = name.strip()
            name = name.split('-')
            
            # documentation for finding all possible pairs
            # for j in range(1, len(name)):
            #     if(validSign(name[j-1]) and validSign(name[j])):
            #         pair = [name[j-1], name[j]]
            #         signs[i].append(pair[0] + "-" + pair[1])

            for sign in name:
                if validSign(sign):
                    signs[i].append(sign)
        i+=1
    return set(signs[0]), set(signs[1])

# add all names to the dataset to be used by RIPPER
# names are represented as numbers indicating where 
# the sign shows up in its name
def getNames(signs, files):
    dataSet = []
    classification = SUMERIAN
    for file in files:
        file.seek(0)
        for name in file:
            #attributes = findPairs(name, signs)
            attributes = [classification]
            attributes.extend(getSignOrder(name, signs))

            dataSet.append(attributes)
        classification = AKKADIAN
    return dataSet

# sign-pairs performs worse than sign order, but function is still here
# for documentation
def findPairs(name, signs):
    name = name.strip()
    name = name.split('-')
    attributes = ["0" for i in range(len(signs))]
    pos = 1
    for i in range(1, len(name)):
        for k in range(len(signs)):
            pair = signs[k].strip()
            pair = signs[k].split('-')
            if (name[i-1] == pair[0] and name[i] == pair[1]):
                if attributes[k] == "0":
                    attributes[k] = str(pos)
                else:
                    attributes[k] += str(pos)
        pos+=1
    return attributes

# return a list containing 0's if the sign is not in
# the name, and positive numbers representing the
# position of the sign in the name
def getSignOrder(name, signs):
    name = name.strip()
    name = name.split('-')
    pos = 1
    attributes = ["0" for i in range(len(signs))]
    for sign in name:
        for i in range(len(signs)):
            if (signs[i] == sign):
                if attributes[i] == "0":
                    attributes[i] = str(pos)
                else:
                    attributes[i] += str(pos)
        pos+=1
    return attributes

def createSignsList(files, max):
    sumerianSigns, akkadianSigns = getSigns(files)
    
    commonSigns = sumerianSigns.intersection((akkadianSigns))
    uniqueSumerianSigns = sumerianSigns.difference(akkadianSigns)
    uniqueAkkadianSigns = akkadianSigns.difference(sumerianSigns)

    signs = []
    signs.extend(getMostFreqSigns(list(uniqueSumerianSigns), max))
    signs.extend(getMostFreqSigns(list(uniqueAkkadianSigns), max))
    signs.extend(getMostFreqSigns(list(commonSigns), 0))
    
    return signs

# creates and store the dataset in a csv file
# returns true if opening input files are successful
# otherwise return false
def createDataSet(suTrainFile, akTrainFile, max):
    # check if file is empty or missing
    try:
        if os.stat(suTrainFile).st_size == 0 or os.stat(akTrainFile).st_size == 0:
            return False
    except OSError:
        return False
    
    files = [
        open(suTrainFile, 'r', encoding='utf8'),
        open(akTrainFile, 'r', encoding='utf8')]

    signs = createSignsList(files, max)
    names = getNames(signs, files)
    
    for file in files:
        file.close()

    # store signs list for future use such as predicting names
    data = np.array(names)
    with open('signs.json', 'w', encoding='utf-8') as file:
        json.dump(signs, file, ensure_ascii=False)
        file.close()

    return data[:, 1:], data[:, 0], signs

# trains and returns two ripper models that classifies either Sumerian or Akkadian
def train(max):
    X, y, signs = createDataSet('input_files/Sumerian.txt', 'input_files/Akkadian.txt', max)

    df = pd.DataFrame(data=X)
    df.columns = signs
    df['class'] = y

    rand_seed = 43
    X_train, X_test, y_train, y_test = train_test_split(df, df['class'], test_size=.5, random_state=rand_seed, stratify=y)
    
    print("Training")
    ripper_su = lw.RIPPER(verbosity=0, k=2, prune_size=0.3, random_state=rand_seed)
    ripper_su.fit(X_train, y_train, class_feat='class', pos_class=SUMERIAN)
    ripper_su
    
    ripper_ak = lw.RIPPER(verbosity=0, k=2, prune_size=0.3, random_state=rand_seed)
    ripper_ak.fit(X_train, y_train, class_feat='class', pos_class=AKKADIAN)
    ripper_ak
    
    print('Sumerian')
    testModel(ripper_su, X_test, y_test)
    print('\nAkkaidian')
    testModel(ripper_ak, X_test, y_test)

    return ripper_su, ripper_ak

# tests the trained ripper model on the testing set
def testModel(ripper, X_test, y_test):    
    ripper.out_model()

    f1 = ripper.score(X_test, y_test, f1_score)
    precision = ripper.score(X_test, y_test, precision_score)
    recall = ripper.score(X_test, y_test, recall_score)
    conf_matrix = ripper.score(X_test, y_test, confusion_matrix)
    accuracy = ripper.score(X_test, y_test, accuracy_score)
    
    print(f'accuracy: {accuracy}')
    print(f'f1 score: {f1}')
    print(f'precision: {precision}')
    print(f'recall: {recall}')
    print(conf_matrix)

# classifies input name using an input trained ripper model
def predictName(ripper, name):
    signs = []
    with open('signs.json', 'r', encoding='utf-8') as file:
        signs = json.load(file)

    signOrder = getSignOrder(name, signs)

    df = pd.DataFrame(data=[signOrder])
    df.columns = signs

    return ripper.predict(df)[0]
     
if __name__ == '__main__':
    ripper_su, ripper_ak = train(0)