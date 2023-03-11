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

def setupAlgorithm():
	# Train the RNN model
	RNN = getModel()

	return RNN

RNN = setupAlgorithm()

def classifyNames(filename, RNN = RNN):
	# Classify Names using RNN
	mylst, names = classify(RNN, filename)
	
	createOutputFile(filename, mylst, names)

# Needs to print to a file, not to terminal
classifiedNames = classifyNames("test_files/akkadian_test.txt")


# Write a function that waits for user in

# Tests the Accuracy and precision before and after ripper algorithm
testData = getTestData()
classifiedNames = classifyNameList(testData, RNN)
testAlgorithm(classifiedNames, testData)


