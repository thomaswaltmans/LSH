# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:13:41 2021

@author: Thomas
"""

import json
import re
import os
import numpy as np
import pandas as pd
import math
import time
import random
import copy
from sklearn.model_selection import train_test_split

os.chdir('C:\\Users\\Thomas\\Documents')

# Opening JSON file
json_file = open('Econometrics\Master\Computer Science\paper\TVs-all-merged.json')
json_load = json.load(json_file)

#function for extracting some values from JSON
def json_extract(obj, key):
    arr = []
    
    def extract(obj, arr, key):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, arr, key)
                elif k == key:
                    arr.append(v)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr

    values = extract(obj, arr, key)
    return values

#extract titles from JSON
titles = json_extract(json_load, 'title')

#modify titles
def modify(mylist):
    mod = []
    for t in mylist:
        t = t.replace("Inch", "inch")
        t = t.replace(" inch", "inch")
        t = t.replace("-inch", "inch")
        t = t.replace("\"", "inch")
        t = t.replace("\”", "inch")
        t = t.replace(" Hz", "Hz")
        t = t.replace("P", "p")
        t = t.replace(" p", "p")
        mod.append(t)
    return mod
titles = modify(titles)

#extract ids from JSON
ids = json_extract(json_load, 'modelID')

#split the data into train/test
titles_train, titles_test, ids_train, ids_test = train_test_split(titles, ids, test_size=0.37)

#part of a function to find model words using regex
model_words = '([a-zA-Z]*(([0-9]+[ˆ0-9,])|([ˆ0-9,]+[0-9]))[a-zA-Z]*)'

#function for finding real duplicates from titles and ids
def realDup(titlelist, idlist):
    idlistcopy = idlist.copy()
    
    #combine (title, id) pairs
    titleids = {}
    for key in titlelist:
        for value in idlistcopy:
            titleids[key] = value
            idlistcopy.remove(value)
            break
    
    
    
    #empty duplicate matrix
    duplicates = pd.DataFrame(np.zeros((len(titleids),len(titleids))), columns = titleids, index = titleids)
    for product1 in duplicates:
        for product2 in duplicates:
            if product1 != product2:
                if titleids[product1] == titleids[product2]:
                    duplicates.loc[product1, product2] = 1 #mark duplicates as 1
    
    return duplicates

realduplicates = realDup(titles_train, ids_train)

### Binary vectors from regex ###
def binaryvector(product):
    modelwordset = set()
    for i in titles:
        MW = re.findall(model_words, i, flags=re.IGNORECASE)
        for mw in MW:
            modelwordset.add(mw[0])
        
    b = {}
    for mw in modelwordset:
        if mw in product:
            b[mw] = 1
        else:
            b[mw] = 0
    return b

def binaryvectors(products):
    modelwordset = set()
    for i in titles:
        MW = re.findall(model_words, i, flags=re.IGNORECASE)
        for mw in MW:
            modelwordset.add(mw[0])
    
    x = {}
    for title in products:
        x[title] = binaryvector(title)
    return pd.DataFrame(x)


### Min-hashing ###
def minhash(n, P):
    #make empty signature matrix
    S = pd.DataFrame(np.zeros((n,len(P.columns))), columns = P.columns)
    #find first 1 in a binary vector permutation (sample)
    for i in range(n):
        p = P.sample(frac=1)
        print("Iteration ", i)
        for v in p:
            vector = p.loc[:,v] #binary vector of product v from total set/matrix of vectors
            first = 0
            for f in vector:
                first += 1
                if f==1:
                   S.loc[i, v] = first
                   break
               
    return S


### Locality-Sensitive Hashing ###
def LSH(S, b):
    r = int(len(S)/b)
    
    #make empty buckethashmatrix
    BHM = pd.DataFrame(np.zeros((b,len(S.columns))), columns = S.columns)
    #fill the buckethash matrix [1,2,3,4,5] => 12345
    string = ""
    for v in S:
        rowcount = 0
        bucketrow = 0
        for i in range(len(S)):
            string += str(int(S.loc[i,v]))
            rowcount += 1
            if rowcount==r:
                BHM.loc[bucketrow,v] = string
                rowcount = 0
                string = ""
                bucketrow += 1
    
    #make empty candidates matrix
    candidates = pd.DataFrame(np.zeros((len(S.columns), len(S.columns))), columns = S.columns, index = S.columns)
    #Hash to bucket per band and label candidates 1 in the candidate matrix
    buckets = dict()
    for i in range(b):
        for v in BHM:
            thisbucket = BHM.loc[i,v]
            if thisbucket in buckets:
                buckets[thisbucket].append(v)
            else:
                buckets[thisbucket] = [v]
        for j in buckets:
            duplicates = buckets[j]
            for product1 in duplicates:
                for product2 in duplicates:
                    if product1 != product2: #do not label itself als candidate
                        if candidates.loc[product1, product2] == 0 and candidates.loc[product2, product1] == 0: #only label if not found yet (and no double labeling [A,B], [B,A])
                            candidates.loc[product1, product2] = 1
                            print('Duplicate!', product1, "and", product2)
        buckets = dict() #emtpy the buckets for the next band
        print('Next band')

    return candidates


### Jaccard similarity ###
def jacSim(a,b):
    aandb = 0
    aorb = 0
    for i in range(len(a)):
        if a[i] != 0 and b[i] != 0:
            aandb += 1
        if a[i] != 0 or b[i] != 0:
            aorb += 1
    sim = aandb/aorb
    return sim

### Cos-similarity ###
def cosSim(a, b):
    ab = 0
    lengthA = 0
    lengthB = 0
    for i in range(len(a)):
        ab += a[i]*b[i]
        lengthA += a[i]**2
        lengthB += b[i]**2
    sim = ab/(math.sqrt(lengthA)*math.sqrt(lengthB))
    return sim

def checkSim(vectormatrix, candidates, t):
    
    #create empty checked candidate matrix
    checked = pd.DataFrame(np.zeros((len(candidates.columns), len(candidates.columns))), columns = candidates.columns, index = candidates.columns)
    
    for cand1 in checked:
        for cand2 in checked:
            if candidates.loc[cand1,cand2] == 1: #find labeled candidates
                print('Found duplicate')
                v1 = vectormatrix[cand1].values
                v2 = vectormatrix[cand2].values
                if jacSim(v1, v2) > t or cosSim(v1, v2) > t: #if either jaccard or cosine is above threshold (sufficiently similar)
                    print("Real duplicates", cand1, "and", cand2)
                    checked.loc[cand1,cand2] = 1 #label checked candidate
    
    return checked


### Running the algorithms ###
n = 100 #number of minhashes
b = 10 #number of bands
r = int(n/b)
threshold = (1/b)**(1/r)

print("Running LSH using a threshold of", '%.2f' % threshold)

#begin timer
time0 = time.time()

#algorithms
print("Creating binary vectors...")
P = binaryvectors(titles_train)
print("Found them!")
time1 = time.time()

print("Hashing...")
S = minhash(n, P)
time2 = time.time()

print("LSH...")
pairs = LSH(S, b)
time3 = time.time()

print("Checking candidates...")
final = checkSim(P, pairs, threshold)
time4 = time.time()

#timers
Ptime = time1-time0
Stime = time2-time1
LSHtime = time3-time2
Checktime = time4-time3
totaltime = time4-time0

print('%.2f' % Ptime,"s for creating binary vectors")
print('%.2f' % Stime,"s for min-hashing")
print('%.2f' % LSHtime,"s for LSH")
print('%.2f' % Checktime,"s for checking candidate pairs")
print('%.2f' % totaltime,"s in total")

#performance measures
dn = 0
ncomparisons = 0
dfound = 0
ncompsim = 0
dfsim = 0

for t1 in final:
    for t2 in final:
        if pairs.loc[t1,t2] == 1:
            ncomparisons +=1
        if final.loc[t1,t2] == 1:
            ncompsim += 1
        if realduplicates.loc[t1,t2] == 1:
            dn += 0.5
            if pairs.loc[t1,t2] == 1:
                dfound += 1
            if final.loc[t1,t2] == 1:
                dfsim +=1

PQ = dfound/ncomparisons
PC = dfound/dn
F1 = (2*PQ*PC)/(PQ+PC)
FOC = 2*ncomparisons/(len(pairs)**2-1000)

PQsim = dfsim/ncompsim
PCsim = dfsim/dn
F1sim = (2*PQsim*PCsim)/(PQsim+PCsim)
FOCsim = 2*ncompsim/(len(final)**2-1000)

print(" ")
print("--- LSH performance ---")
print("Pair quality: ",PQ)
print("Pair completeness: ",PC)
print("F1-measure: ",F1)
print("Fraction of comparisons:", FOC)
print(" ")
print("---Final performance---")
print("Pair quality: ",PQsim)
print("Pair completeness: ",PCsim)
print("F1-measure: ",F1sim)
print("Fraction of comparisons:", FOCsim)
