#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models
@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller
DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random
import codecs

vocab = codecs.open("brown_vocab_100.txt")
print(vocab)
#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i
    #TODO: import part 1 code to build dictionary

with open('word_to_index_100.txt', 'w') as wf:
    wf.write(str(word_index_dict))

f = codecs.open("brown_100.txt")    #corpus

knowns = [('in','the'),('the', 'jury'),('jury', 'said'), ('agriculture', 'teacher')]

#counts = #TODO: initialize numpy 0s array
counts = np.zeros((len(knowns), len(word_index_dict))) #Need to match the vocabulary, so to store the bigrams counts for every pair of words. It must fit the bigram model
# counts += 0.1

#TODO: iterate through file and update counts
for line in f:

    previous_word = '<s>'
    previous_word_2 = '<s>'

    words = line.lower().split()       #Was first line.strip().split()

    print(words)

    for i in range(2, len(words)):
        word = words[i]

        prev_word = words[i - 1]

        prev_word_2 = words[i - 2]

        print(prev_word_2,prev_word,word)




