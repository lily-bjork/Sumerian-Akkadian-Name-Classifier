import re

def startComp():
    listOutput = open("SearchResults.txt", "r", encoding="utf8")
    rnnOutput = open("output_files/SumerianDBNames_output.txt", encoding="utf8")

    listClass, listTotalS, listTotalA, totalSameNames = getStats(listOutput)

    rnnClass, rnnTotalS, rnnTotalA = getStats(rnnOutput)

    agree, disagreeListSumerian, disagreeListAkkadian, totalNames = compareNames(listClass, rnnClass)

    printResults(listTotalS, listTotalA, rnnTotalS, rnnTotalA, totalSameNames, agree, disagreeListSumerian, disagreeListAkkadian, totalNames)

def getStats(file):
    classification = []
    totalSumerian = 0
    totalAkkadian = 0
    totalSameNames = 0
    i = 0
    for name in file:
        classification.append(re.split(r"\t|\: ", name.strip())[1])
        if classification[i] == "Akkadian/Sumerian":
            totalSameNames+=1
        elif classification[i] == "Akkadian":
            totalAkkadian+=1
        elif classification[i] == "Sumerian":
            totalSumerian+=1
        i+=1
    
    if totalSameNames == 0:
        return classification, totalSumerian, totalAkkadian
    else:
        return classification, totalSumerian, totalAkkadian, totalSameNames

def compareNames(listOutput, rnnOutput):
    agree = 0
    disagreeListSumerian = 0
    disagreeListAkkadian = 0
    totalNames = len(listOutput)
    for i in range(totalNames):
        if listOutput[i] == rnnOutput[i]:
            agree+=1
        elif listOutput[i] == "Sumerian" and rnnOutput[i] == "Akkadian":
            disagreeListSumerian+=1
        elif listOutput[i] == "Akkadian" and rnnOutput[i] == "Sumerian":
            disagreeListAkkadian+=1
    
    return agree, disagreeListSumerian, disagreeListAkkadian, totalNames

def printResults(listTotalS, listTotalA, rnnTotalS, rnnTotalA, totalSameNames, agree, disagreeListSumerian, disagreeListAkkadian, totalNames):
    print(f"Total Names:\t{totalNames}\n")
    
    print("List Output")
    print(f"Total Sumerian Names:\t{listTotalS}\n")
    print(f"Total Akkadian Names:\t{listTotalA}\n")

    print("RNN Output\n")
    print(f"Total Sumerian Names:\t{rnnTotalS}\n")
    print(f"Total Akkadian Names:\t{rnnTotalA}\n")

    print(f"Both Sumerian and Akkadian Names:\t{totalSameNames}\n")

    print("Compare Names")
    print(f"Agree:\t{agree}\n")
    print(f"Misclassify as Sumerian:\t{disagreeListAkkadian}\n")
    print(f"Misclassify as Akkadian:\t{disagreeListSumerian}\n")

startComp()