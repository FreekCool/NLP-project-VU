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
counts = np.zeros((len(word_index_dict), len(word_index_dict))) #Need to match the vocabulary, so to store the bigrams counts for every pair of words. It must fit the bigram model


#TODO: iterate through file and update counts
previous_word = '<s>'
for line in f:
    words = line.lower().split()       
    for i in range(1, len(words)):
        word = words[i]
        counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
with open('bigram_probs.txt', 'w') as wf:
    prob = probs[word_index_dict['all'], word_index_dict['the']]
    wf.write('p(the | all) = {}\n'.format(prob))

    prob = probs[word_index_dict['the'], word_index_dict['jury']]
    wf.write('p(jury | the) = {}\n'.format(prob))

    prob = probs[word_index_dict['the'], word_index_dict['campaign']]
    wf.write('p(campaign | the) = {}\n'.format(prob))

    prob = probs[word_index_dict['anonymous'], word_index_dict['calls']]
    wf.write('p(calls | anonymous) = {}\n'.format(prob))


f.close()

'''PROBLEM 6'''
with open('toy_corpus.txt', 'r') as file:
    for line in file:
        sentprob = 1

        words = line.lower().split()

        sent_len = len(words)

        bigrams = 0

        for i in range(1, len(words)):

            prev_word = words[i - 1]
            word = words[i]

            word_prob = probs[word_index_dict[prev_word], word_index_dict[word]]

            sentprob *= word_prob

            bigrams += 1

        perplexity = 1 / (pow(sentprob, 1.0 / bigrams))

        print(perplexity)

'''PROBLEM 7, GENERATE SENTENCES'''
from generate import GENERATE

with open('bigram_generation.txt', 'w') as wf:
    for i in range(0,15):
        sentence = GENERATE(word_index_dict, probs, 'bigram', 15, '<s>')
        wf.write('{}\n'.format(sentence))
