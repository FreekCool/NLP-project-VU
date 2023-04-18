#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word = line.rstrip()
    word_index_dict[word] = i

with open('word_to_index_100.txt', 'w') as wf:
    wf.write(str(word_index_dict))

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict)) #TODO: initialize counts to a zero vector

#TODO: iterate through file and update counts
for line in f:
    words = line.lower().split()
    for i in range(1, len(words)):
        word = words[i]
        counts[word_index_dict[word]] += 1


f.close()

#TODO: normalize and writeout counts. 
probs = counts / np.sum(counts)

with open('unigram_probs.txt', 'w') as wf:
    prob = probs[word_index_dict['all']]
    wf.write('the probability of "all" = {}\n'.format(prob))

    prob = probs[word_index_dict['resolution']]
    wf.write('probability of "resolution" = {}\n'.format(prob))



