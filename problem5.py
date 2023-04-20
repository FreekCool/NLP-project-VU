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


#counts = #TODO: initialize numpy 0s array
counts = np.zeros((813, 813, 813)) #Need to match the vocabulary, so to store the bigrams counts for every pair of words. It must fit the bigram model
#counts += 0.1

#TODO: iterate through file and update counts
previous_word = '<s>'
previous_word_2 = '<s>'
for line in f:
    words = line.lower().split()       #Was first line.strip().split()
    for i in range(1, len(words)):
        word = words[i]
        counts[word_index_dict[previous_word_2], word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word
        previous_word_2 = previous_word

#TODO: normalize counts
probs = counts / np.sum(counts, axis=(1, 2), keepdims=True)

#TODO: writeout bigram probabilities
with open('unsmooth_trigram_probs.txt', 'w') as wf:
    prob = probs[word_index_dict['past'], word_index_dict['in'], word_index_dict['the']]
    wf.write('p(past|in, the) = {}\n'.format(prob))

    prob = probs[word_index_dict['the'], word_index_dict['in'], word_index_dict['time']]
    wf.write('p(time|in, the) = {}\n'.format(prob))

    prob = probs[word_index_dict['jury'], word_index_dict['the'], word_index_dict['said']]
    wf.write('p(said|the, jury) = {}\n'.format(prob))

    #prob = probs[word_index_dict['anonymous'], word_index_dict['calls']]
    #wf.write('p(the|all) = {}\n'.format(prob))


f.close()