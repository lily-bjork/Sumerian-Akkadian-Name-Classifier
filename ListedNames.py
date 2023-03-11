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
from IPython.display import clear_output

# Do not edit
languages = []
test_languages = []
s = []
a = []


def findFiles(path): return glob.glob(path)
deleteChars = "[]<>!?-x/X"

# Read a file and split into lines (each line is a name)
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().lower().replace(deleteChars,"").replace("Ú","U2").replace("Ù","U3").replace("ú","u2").replace("ù","u3").replace("Í","I2").replace("Ì","I3").replace("í","i2").replace("ì","i3").replace("Á","A2").replace("À","A3").replace("á","a2").replace("à","a3").replace("É","E2").replace("È","E3").replace("é","e2").replace("è","e3").replace("š","sz").replace("Š","SZ").replace("ṣ","s").replace("Ṣ","S").replace("ṭ","t").replace("Ṭ","T").replace("DINGIR","dingir").split('\n')
    return lines

for filename in findFiles('input_files/*.txt'):
    language = os.path.splitext(os.path.basename(filename))[0]
    languages.append(language)
    knownNames = readLines(filename)
    for line in knownNames:
        if (language == "Akkadian"):
            a.append(line)
        else:
            s.append(line)



names = readLines("SumerianDBNames.txt")
f = open("SearchResults.txt","w")
u = open("UnknownNames.txt", "w")
found = 0
both = 0
Akkadian = 0
Sumerian = 0
Unknown = 0
either = 0

for name in names:
    found = 0
    for i in range(len(a)):
        both = 0
        if name == a[i]:
            f.write(name)
            f.write("\t")
            for i in range(len(s)):
                if name == s[i]:
                    f.write("Akkadian/Sumerian")
                    either +=1
                    both = 1
                    found = 1
                    break
            if (both == 0):
                f.write("Akkadian")
                Akkadian +=1
            f.write("\n")
            found = 1
            break
    if (found == 0):
        for i in range(len(s)):
            both = 0
            if name == s[i]:
                f.write(name)
                f.write("\t")
                f.write("Sumerian")
                Sumerian +=1
                f.write("\n")
                found = 1
                break
    if found == 0:
        f.write(name)
        f.write("\tUnknown\n")
        u.write(name)
        u.write("\n")
        Unknown +=1

f.write("\nSumerian: ")
f.write(str(Sumerian))
f.write("\nAkkadian: ")
f.write(str(Akkadian))
f.write("\nEither: ")
f.write(str(either))
f.write("\nUnknown: ")
f.write(str(Unknown))
f.close()
u.close()