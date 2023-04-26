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
    #print(words)
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


'''PROBLEM 6'''
with open('toy_corpus.txt', 'r') as file:
    for line in file:
        sentprob = 1

        words = line.lower().split()

        sent_len = len(words)

        for word in words:
            word_prob = probs[word_index_dict[word]]

            sentprob *= word_prob

        perplexity = 1/(pow(sentprob, 1.0/sent_len))

        print(perplexity)


'''PROBLEM 7, GENERATION'''
from generate import GENERATE

with open('unigram_generation.txt', 'w') as wf:
    for i in range(0,15):
        sentence = GENERATE(word_index_dict, probs, 'unigram', 15, '<s>')
        wf.write('{}\n'.format(sentence))