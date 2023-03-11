from sklearn import model_selection, datasets
import joblib
import pickle
from RNN import classifyNameList, getModel, classify, getTestData
from AlgoTester import testAlgorithm

def createOutputFile(filename, classifications, names):
	filepath = filename.replace(".txt", "_output.txt")
	filename = filepath.split("/")[-1]

	outfile = open("output_files/" + filename, 'w')
	i = 0
	for name in names:
		outfile.write(f"{name}: {classifications[i]}\n")
		i+=1

#train and save a new model
def saveModel():
	filename = "Model.joblib"
	joblib.dump(getModel(),filename)

#fetch the model
def setupAlgorithm():
	filename = "Model.joblib"
	loaded_model = joblib.load(filename)

	return loaded_model

# Uncomment to train a new model
# saveModel()
RNN = setupAlgorithm()

def classifyNames(filename, RNN = RNN):
	# Classify Names using RNN
	mylst, names = classify(RNN, filename)
	
	createOutputFile(filename, mylst, names)

# # Needs to print to a file, not to terminal
# classifiedNames = classifyNames("test_files/akkadian_test.txt")



# Tests the Accuracy and precision before and after ripper algorithm

testData = getTestData()
classifiedNames = classifyNameList(testData, RNN)
testAlgorithm(classifiedNames, testData)

# classifyNames("SumerianDBNames.txt")