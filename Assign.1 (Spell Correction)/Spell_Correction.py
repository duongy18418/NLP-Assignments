from nltk.corpus import wordnet #dictionnary D
import PyDictionary
import pytrec_eval
import BirkBeck #Spelling error C
import _io
import numpy as np


def MED(source, target):
    s = len(source)
    t = len(target)
    del_cost = 1
    ins_cost = 1
    sub_cost = 0
    distance = np.zeros((s+1, t+1))

    for i in range(1, s+1):
        distance[i,0] = i
    for j in range(1,t+1):
        distance[0,j] = j

    for i in range(1, s+1):
        for j in range(1, t+1):
            func1 = distance[i, j-1]
            func2 = distance[i-1, j]
            func3 = distance[i-1, j-1]

            if source[i-1] == target[j-1]:
                sub_cost = 0
            else:
                sub_cost = 2
            
            if func1 <= func2 and func1 <= func3:
                distance[i, j] = func1 + ins_cost
            elif func2 <= func1 and func2 <= func3:
                distance[i, j] = func2 + del_cost
            else:
                distance[i, j] = func3 + sub_cost

    return distance[s, t]

corpus = open('./Assign.1 (Spell Correction)/Data/UPWARDDAT.643', 'r')
line = 0
corpus_list = []

while line <= 20:
    read_string = corpus.readline()
    temp_string = read_string.replace('\n', '')
    string = temp_string.split("  ")
    corpus_list.append(string)
    line += 1
corpus.close()

correct_word = ''
incorrect_word = ''
words = [n for n in wordnet.all_lemma_names() if len(n) <= 10 and n.find("_") == -1]

s1 = []
s5 = []
s10 = []
count = 0

for c in corpus_list:
    distance = []
    count += 1
    for w in words:
        new_word = (w, MED('#'+w, '#'+c[0]))
        #print(new_word)
        distance.append(new_word)
    sort_distance = sorted(distance, key=lambda x: x[1])

    lowest_distance = []
    for i in range(0,10):
        lowest_distance.append(sort_distance[i])
    #print(lowest_distance)
    if c[1] == lowest_distance[0][0]:
        k1 = 1
    else:
        k1 = 0
    s1.append(k1)
    #print(s1)

    k5 = 0
    for i in range(0, 4):
        if c[1] == lowest_distance[i][0]:
            k5 = 1
            break
    s5.append(k5)
    
    k10 = 0
    for i in range(0,9):
        if c[1] == lowest_distance[i][0]:
            k10 = 1
            break
    s10.append(k10)

print("Average s@1 ", pytrec_eval.compute_aggregated_measure("gm", s1))
print("Average s@5 ", pytrec_eval.compute_aggregated_measure("gm", s5))
print("Average s@10 ", pytrec_eval.compute_aggregated_measure("gm", s10))



"""for i in range (len(corpus_list)):
    print(corpus_list[i])
    correct_word = corpus_list[i][0]
    incorrect_word = corpus_list[i][1]
    print(MED(correct_word, incorrect_word))
    print('\n')"""