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

file = open('brown_vocab_100.txt', 'r')
for index, line in enumerate(file):
    word = line.rstrip()
    word_index_dict[word] = index

# TODO: write word_index_dict to word_to_index_100.txt
with open('word_to_index_100.txt', 'w') as wf:
    wf.write(str(word_index_dict))

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict)) #TODO: initialize counts to a zero vector

#TODO: iterate through file and update counts
for line in f:
    words = line.lower().split()
    for word in words:
        counts[word_index_dict[word]] += 1

f.close()


#TODO: normalize and writeout counts.
probs = counts / np.sum(counts)

with open('unigram_probs.txt', 'w') as wf:
    prob = probs[word_index_dict['all']]
    wf.write('the probability of "all" = {}\n'.format(prob))

    prob = probs[word_index_dict['resolution']]
    wf.write('probability of "resolution" = {}\n'.format(prob))

#TODO:iterate through each sentence in the toy corpus.
f2 = open("toy_corpus.txt")

for line in f2:
    sentprob = 1
    words = line.lower().split()
    for word in words:
        wordprob = probs[word_index_dict[word]]
        sentprob *= wordprob
    print(sentprob)
#with open('unigram_eval.txt') as wf:
#    wf.write('the probability of')
#TODO: calculate the joint probabilities of all words in the sentence under the unigram model.
#TODO: write the probability of each sentence to a file unigram_eval.txt, formatted to have one probability for each line of the output file.
