from collections import Counter

def getSyllables(file):
  syllables = []

  for name in file:
    name = name.strip()
    name = name.split('-')
    for s in name:
      syllables.append(s)
  return syllables

def predictName(name, acount, scount):
  name = name.strip()
  name = name.split('-')
  numOfSyllables = len(name)

  total_ak_syllables = sum(acount.values())
  total_sm_syllables = sum(scount.values())
  sTotal = 0
  aTotal = 0

  for s in name:
    aTotal += (acount[s]/total_ak_syllables)*1000
  for s in name:
    sTotal += (scount[s]/total_sm_syllables)*1000
  aPrecision = aTotal/numOfSyllables
  sPrecision = sTotal/numOfSyllables

  if aPrecision < 3 and sPrecision < 3:
    return "Other"
  elif aPrecision > sPrecision:
    return "Akkadian"
  else:
    return "Sumerian"

def getPrediction(name):
    af = open("input_files/Akkadian.txt", 'r')
    sf = open("input_files/Sumerian.txt", 'r')

    ak_syllables = getSyllables(af)
    sm_syllables = getSyllables(sf)

    ak_counter = Counter(ak_syllables)
    sm_counter = Counter(sm_syllables)
    return predictName(name, ak_counter, sm_counter)
