from __future__ import division, print_function, unicode_literals

import glob
import math
import os
import random
import string
import time
import unicodedata
from io import open

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Do not edit
languages = []
test_languages = []
data = []
X = []
y = []


def findFiles(path): return glob.glob(path)
deleteChars = "[]<>!?-x/X"

# Read a file and split into lines (each line is a name)
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().replace(deleteChars,"").replace("Ú","U2").replace("Ù","U3").replace("ú","u2").replace("ù","u3").replace("Í","I2").replace("Ì","I3").replace("í","i2").replace("ì","i3").replace("Á","A2").replace("À","A3").replace("á","a2").replace("à","a3").replace("É","E2").replace("È","E3").replace("é","e2").replace("è","e3").replace("š","sz").replace("Š","SZ").replace("ṣ","s").replace("Ṣ","S").replace("ṭ","t").replace("Ṭ","T").replace("DINGIR","dingir").split('\n')
    return lines

for filename in findFiles('input_files/*.txt'):
    language = os.path.splitext(os.path.basename(filename))[0]
    languages.append(language)
    lines = readLines(filename)
    for line in lines:
      X.append(line)
      y.append(language)
      data.append((line, language))

n_languages = len(languages)

#split the data (70 30 is the default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 123, stratify = y)
# split test data (70 30)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.3, random_state = 123, stratify = y_test)
print("Training Data: ", len(X_train))
print("Testing Data: ", len(X_test))


#get all the letters
all_letters = string.ascii_letters + "ŠšṢṣÌÍìíĪīÁÀàáĀāâÚÙùúŪūḪḫṬṭÉÈéèʾ𝄒Ēē"
n_letters = len(all_letters)

# Turn the names into tensors
def lineToTensor(name):
    rep = torch.zeros(len(name), 1, n_letters)
    for index, letter in enumerate(name):
        pos = all_letters.find(letter)
        rep[index][0][pos] = 1
    return rep
 

#function to create lang representation
def nat_rep(lang):
    return torch.tensor([languages.index(lang)], dtype = torch.long)

#define a basic rnn network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTM(input_size, hidden_size) #LSTM cell
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 2)

    def forward(self, input, hidden):
        out, hidden = self.lstm_cell(input.view(1, 1, -1), hidden)
        output = self.h2o(hidden[0])
        output = self.softmax(output)
        return output.view(1, -1), hidden
    
    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        

# Do not edit
#returns the prediction of the language classification
def predict(net, name):
    net.eval()
    # turn name into a tensor
    name_ohe = lineToTensor(name)
    hidden = net.initHidden()
    for i in range(name_ohe.size()[0]):
        output, hidden = net(name_ohe[i], hidden)
    return output


#Creates a list of data pairs
def dataloader(npoints, X_, y_):
    to_ret = []
    for i in range(npoints):
        index_ = np.random.randint(len(X_))
        name, lang = X_[index_], y_[index_] #get the data at the random index
        to_ret.append((name, lang, lineToTensor(name), nat_rep(lang)))

    return to_ret

# dataloader(1, X_train, y_train)

# Do not edit
#Evaluates and returns the model accuracy
def eval(net, n_points, k, X_, y_):
     data_ = dataloader(n_points, X_, y_)
     correct = 0
     n = n_points

     for name, language, name_ohe, lang_rep in data_:
         output = predict(net, name) #prediction
         val, indices = output.topk(k) #get the top k predictions
        #  print(name,languages[indices[0]],"\n")
         if lang_rep in indices:
             correct += 1
         elif (val[0] >-.4):
             n -= 1
     accuracy = correct/n
     return accuracy 

# Prints the name and most likely name classification to the output file
def prediction(net, names):
  file = open("accuracy_test.txt","w")
  for name in names:
    output = predict(net,name)
    val, indices = output.topk(1)
    file.write((name+output))


#function to train the data
def train(net, opt, criterion, n_points):
    opt.zero_grad()
    total_loss = 0
    data_ = dataloader(n_points, X_train, y_train)
    
    for name, language, name_ohe, lang_rep in data_:
        hidden = net.initHidden()
        for i in range(name_ohe.size()[0]):
            output, hidden = net(name_ohe[i], hidden) 
        loss = criterion(output, lang_rep)
        loss.backward(retain_graph=True)
        total_loss += loss

    opt.step()       
    return total_loss/n_points

def train_setup(net, lr = 0.005, n_batches = 100, batch_size = 20, momentum = 0.9, display_freq = 10):
    #loss function
    net.train()
    criterion = nn.CrossEntropyLoss() 
    #optimizer
    opt = optim.Adam(net.parameters(), lr = lr) 
    loss_arr = np.zeros(n_batches*20 + 1)
    #train the model
    # 4 seems the ideal number of Epochs before the results become... interesting
    for j in range(5):
      print((j),"Epoch")
      # print("\n")
      for i in range(n_batches):
          loss_arr[(j*10)+i + 1] = (loss_arr[i]*i + train(net, opt, criterion, batch_size))/(i + 1)
          # print accuracy and loss every display_freq iterations (default 10)
          # if i%display_freq == display_freq - 1:
          #   net.eval()
          #   # print("Iteration number", i + 1, "Accuracy:", round(eval(net, len(X_test), 1, X_test, y_test),4), 'Loss:', round(loss_arr[i],4))
          #   net.train()
      print("Accuracy:", round(eval(net, len(X_test), 1, X_test, y_test),4), 'Loss:', round(loss_arr[i],4))
    # plt.figure()
    # plt.plot(loss_arr[1:i], "-*")
    # plt.xlabel("Iteration")
    # plt.ylabel("Loss")     
    # print("\n\n")
    # plt.show()

# Classifies an input file
def classify(net, filename):
    names = readLines(filename)
    classifiedNames = {}
    i = 0
    for name in names:
        classifiedNames[i] = classifyName(net, name)
        i+=1
    print(len(names), len(classifiedNames))
    return classifiedNames, names




# Classifies a single name, and returns the classification as a string
def classifyName(net, name):
    output = predict(net, name) #prediction
    val, indices = output.topk(1)
    if (val[0] >-.4):
        return languages[indices]
    else:
        return "Other"

# classifies a list of names
def classifyNameList(names, net):
    classifiedNames = {}
    for name in names:
        classifiedNames[name] = classifyName(net, name)
    return classifiedNames

# create output file
#   outputFile = open("output.txt", "w")
#   for i in names:
#     output = predict(net, i) #prediction
#     val, indices = output.topk(1)
#     # print(val)
#     if (val[0] >-.4):
#       outputFile.write(i)
#       outputFile.write(" ")
#       outputFile.write(languages[indices])
#       outputFile.write("\n")
#     else:
#       outputFile.write(i)
#       outputFile.write(" ")
#       outputFile.write("Other\n")



def getModel():
    n_hidden = 128
    net = LSTM(n_letters, n_hidden, n_languages)
    train_setup(net, lr=0.0005, n_batches = 100, batch_size = 150)
    print("\n")
    # Validation accuracy
    print("Test Accuracy:", round(eval(net, len(X_val), 1, X_val, y_val),4),"\n")
    return net


# Get Test Data
# Returns a dictionary with all key, value pairs of test data
def getTestData():
    pairs = {}
    for i in range(len(X_val)):
        name = X_val[i]
        c = y_val[i]
        pairs[name] = c
    return pairs