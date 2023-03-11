# Values correspond to lace in confusion matrix list
TP = 0 # True Positive
TN = 1 # True Negative
FP = 2 # False Positive
FN = 3 # False Negative

def testAlgorithm(predicted, actual):
    # Confusion matrices for Sumerian and Akkadian
    akkadianMatrix = [0, 0, 0, 0]
    sumerianMatrix = [0, 0, 0, 0]

    for name in predicted:
        
        # Populating Confusion matrix for Akkadian names
        if actual[name] == "Akkadian" and predicted[name] == "Akkadian":
            akkadianMatrix[TP] = akkadianMatrix[TP] + 1
        elif actual[name] == "Akkadian" and predicted[name] == "Sumerian":
            akkadianMatrix[FN] = akkadianMatrix[FN] + 1
        elif actual[name] == "Sumerian" and predicted[name] == "Akkadian":
            akkadianMatrix[FP] = akkadianMatrix[FP] + 1
        elif actual[name] != "Akkadian" and predicted[name] != "Akkadian":
            akkadianMatrix[TN] = akkadianMatrix[TN] + 1

        # Populating Confusion matrix for Sumerian names
        if actual[name] == "Sumerian" and predicted[name] == "Sumerian":
            sumerianMatrix[TP] = sumerianMatrix[TP] + 1
        elif actual[name] == "Sumerian" and predicted[name] == "Akkadian":
            sumerianMatrix[FN] = sumerianMatrix[FN] + 1
        elif actual[name] == "Akkadian" and predicted[name] == "Sumerian":
            sumerianMatrix[FP] = sumerianMatrix[FP] + 1
        elif actual[name] != "Sumerian" and predicted[name] != "Sumerian":
            sumerianMatrix[TN] = sumerianMatrix[TN] + 1

    # Accuracy = (TP+TN) / Total
    # Precision = TP / (TP + FP)
    # Print Accuracy and precision for Akkadian Names

    akAccuracy = (akkadianMatrix[TP] + akkadianMatrix[TN]) / len(predicted)
    akPrecision = akkadianMatrix[TP] / (akkadianMatrix[TP] + akkadianMatrix[FP])
    akFscore = akkadianMatrix[TP] / (akkadianMatrix[TP] + (.5 * (akkadianMatrix[FP] + akkadianMatrix[FN])))
    print(f"Akkadian Accuracy: {akAccuracy}")
    print(f"Akkadian Precision: {akPrecision}")
    print(f"Akkadian F1-Score: {akFscore}")

    # Print Accuracy and precision for Sumerian Names
    suAccuracy = (sumerianMatrix[TP] + sumerianMatrix[TN]) / len(predicted)
    suPrecision = sumerianMatrix[TP] / (sumerianMatrix[TP] + sumerianMatrix[FP])
    suFscore = sumerianMatrix[TP] / (sumerianMatrix[TP] + (.5 * (sumerianMatrix[FP] + sumerianMatrix[FN])))
    print(f"Sumerian Accuracy: {suAccuracy}")
    print(f"Sumerian Precision: {suPrecision}")
    print(f"Sumerian F1-Score: {suFscore}")
