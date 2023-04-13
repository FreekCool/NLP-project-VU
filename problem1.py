#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
vocab_file = "brown_vocab_100.txt"
word_to_index_file = "word_to_index_100.txt"

# Create an empty dictionary to store the word-to-index mappings
word_index_dict = {}

# TODO: read brown_vocab_100.txt into word_index_dict

# TODO: write word_index_dict to word_to_index_100.txt

# Open the vocabulary file and iterate over each line
with open(vocab_file, "r") as f:
    for i, word in enumerate(f):
        # Remove the newline character from the end of the word
        word = word.rstrip()
        # Add the word to the dictionary mapped to its index
        word_index_dict[word] = i

# Write the dictionary to a file
with open(word_to_index_file, "w") as wf:
    # Convert the dictionary to a string and write it to the file
    wf.write(str(word_index_dict))

# Verify that the dictionary was created correctly
print(word_index_dict['all']) # Should print 0
print(word_index_dict['resolution']) # Should print 812
print(len(word_index_dict))
