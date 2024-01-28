from nltk.corpus import wordnet #dictionnary D
import pytrec_eval
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

def find_nearest_token(token):
    special_char = '|!@#$%^&*()_-=+,.<>/\?;:~`1234567890'
    words = [w for w in wordnet.all_lemma_names()
        if len([c for c in list(token) if c in list(w)]) > 0 and not any (c in special_char for c in w)]
    nearest_token = sorted([(MED(token, w), w) for w in words], key=lambda n:n[0])
    return nearest_token

def find_top_k (nearest_token, k):
    length = len(nearest_token) - 1 
    while length > k:
        nearest_token.pop(length)
        length -= 1
    return nearest_token

corpus = open('Assign.1 (Spell Correction)/Data/APPLING2DAT.643', 'r')
line = 0
corpus_list = []
correct_word = []
incorrect_word = []

while line <= 5:
    read_string = corpus.readline()
    temp_string = read_string.replace('\n', '')
    string = temp_string.split(" ")
    corpus_list.append(string)
    line += 1

for i in range(len(corpus_list)):
    correct_word.append(corpus_list[i][1])
    incorrect_word.append(corpus_list[i][0])

for c in range(len(correct_word))  :
    correct_spell = correct_word[c]
    print(correct_spell)
    incorrect_spell = incorrect_word[c]
    checked_token = []
    s1_list = []
    s5_list = []
    s10_list = []
    count = 0

    for k in [1, 5, 10]:
        temp_list = find_nearest_token(incorrect_spell)
        checked_token = find_top_k(temp_list, k)       
        for i in range(len(checked_token)):
            count += 1
            if checked_token[i][1] == correct_spell:
                if  i == 0:
                    s1_list.append(checked_token[i][0])
                elif i == 4:
                    s5_list.append(checked_token[i][0])
                elif i == 9:
                    s10_list.append(checked_token[i][0])
            else:
                s1_list.append(0)
                s5_list.append(0)
                s10_list.append(0)

    print("s@k for k = 1: ", pytrec_eval.compute_aggregated_measure('gm',s1_list))
    print("s@k for k = 5: ", pytrec_eval.compute_aggregated_measure("gm", s5_list))
    print("s@k for k = 10: ", pytrec_eval.compute_aggregated_measure("gm", s10_list))  