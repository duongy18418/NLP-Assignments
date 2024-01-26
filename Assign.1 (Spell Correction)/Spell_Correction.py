from nltk.corpus import wordnet #dictionnary D
import PyDictionary
import pytrec_eval as pe
import BirkBeck #Spelling error C
import _io
import numpy as np

def MED(source, target):
    n = len(source)
    m = len(target)
    del_cost = 1
    ins_cost = 1
    sub_cost = 0
    distance = np.zeros([n+1, m+1])

    for i in range(1, n):
        distance[i,0] = distance[i-1, 0] + del_cost
    for j in range(1,m):
        distance[0,j] = distance[0, j-1] + ins_cost

    for i in range(1, n):
        for j in range(1, m):
            if source[n-1] == target[m-1]:
                sub_cost = 0
            else:
                sub_cost = 1
            distance[i,j] = min (   
                                distance[i-1, j] + del_cost,
                                distance[i-1, j-1] + sub_cost,
                                distance[i, j-1] + ins_cost
                                )

    return distance

def maxMatch (string, dictionnary):
    if not string:
        return []
    for i in range(len(string)-1, -1, -1):
        first_word = (string[i])
        remainder = string[i+1:len(string)]
        if first_word in dictionnary:
            return 


corpus = open('./Assign.1 (Spell Correction)/Data/UPWARDDAT.643', 'r')
line = 0
corpus_list = []

while line <= 5:
    read_string = corpus.readline()
    temp_string = read_string.replace('\n', '')
    string = temp_string.split("  ")
    corpus_list.append(string)
    line += 1
corpus.close()

correct_word = ''
incorrect_word = ''
words = [n for n in wordnet.all_lemma_names() if len(n) <= 10 and n.find("_") == -1]



"""for i in range (len(corpus_list)):
    print(corpus_list[i])
    correct_word = corpus_list[i][0]
    incorrect_word = corpus_list[i][1]
    print(MED(correct_word, incorrect_word))
    print('\n')"""