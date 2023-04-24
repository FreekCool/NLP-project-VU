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

    words = line.lower().split()       #Was first line.strip().split()

    for i in range(2, len(words)):
        word = words[i]

        prev_word = words[i - 1]

        prev_word_2 = words[i - 2]

        if (prev_word_2, prev_word) in knowns:
            knowns_index = knowns.index((prev_word_2, prev_word))

            word_index = word_index_dict[word]

            counts[knowns_index,word_index] += 1

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

'''Write unsmoothened probabilities to txt file'''
with open('problem5_unsmooth.txt', 'w') as wf:
    # For p(past | in, the)
    prev_index = knowns.index(('in','the'))
    word_index = word_index_dict['past']

    prob = probs[prev_index,word_index]
    wf.write(f'p(past | in, the) = {prob}\n')

    # For p(time | in, the)
    prev_index = knowns.index(('in','the'))
    word_index = word_index_dict['time']

    prob = probs[prev_index, word_index]
    wf.write(f'p(time | in, the) = {prob}\n')

    # For p(said | the, jury)
    prev_index = knowns.index(('the', 'jury'))
    word_index = word_index_dict['said']

    prob = probs[prev_index, word_index]
    wf.write(f'p(said | the, jury) = {prob}\n')

    # For p(recommended | the, jury)
    prev_index = knowns.index(('the', 'jury'))
    word_index = word_index_dict['recommended']

    prob = probs[prev_index, word_index]
    wf.write(f'p(recommended | the, jury) = {prob}\n')

    # For p(that | jury, said)
    prev_index = knowns.index(('jury', 'said'))
    word_index = word_index_dict['that']

    prob = probs[prev_index, word_index]
    wf.write(f'p(that | jury, said) = {prob}\n')

    # For p(, | agriculture, teacher)
    prev_index = knowns.index(('agriculture', 'teacher'))
    word_index = word_index_dict[',']

    prob = probs[prev_index, word_index]
    wf.write(f'p(, | agriculture, teacher) = {prob}\n')

'''Create the smoothened probabilities'''
counts += 0.1

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

'''Write smoothened probabilities to txt file'''
with open('problem5_smooth.txt', 'w') as wf:
    # For p(past | in, the)
    prev_index = knowns.index(('in','the'))
    word_index = word_index_dict['past']

    prob = probs[prev_index,word_index]
    wf.write(f'p(past | in, the) = {prob}\n')

    # For p(time | in, the)
    prev_index = knowns.index(('in','the'))
    word_index = word_index_dict['time']

    prob = probs[prev_index, word_index]
    wf.write(f'p(time | in, the) = {prob}\n')

    # For p(said | the, jury)
    prev_index = knowns.index(('the', 'jury'))
    word_index = word_index_dict['said']

    prob = probs[prev_index, word_index]
    wf.write(f'p(said | the, jury) = {prob}\n')

    # For p(recommended | the, jury)
    prev_index = knowns.index(('the', 'jury'))
    word_index = word_index_dict['recommended']

    prob = probs[prev_index, word_index]
    wf.write(f'p(recommended | the, jury) = {prob}\n')

    # For p(that | jury, said)
    prev_index = knowns.index(('jury', 'said'))
    word_index = word_index_dict['that']

    prob = probs[prev_index, word_index]
    wf.write(f'p(that | jury, said) = {prob}\n')

    # For p(, | agriculture, teacher)
    prev_index = knowns.index(('agriculture', 'teacher'))
    word_index = word_index_dict[',']

    prob = probs[prev_index, word_index]
    wf.write(f'p(, | agriculture, teacher) = {prob}\n')







