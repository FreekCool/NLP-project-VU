import nltk
from nltk.corpus import brown

# nltk.download('brown')
# nltk.download('punkt')

'''0. i, unique words of the whole corpus by frequency'''
tokens = brown.words()

freq_dist = nltk.FreqDist(tokens)

sorted_words = sorted(freq_dist, key = freq_dist.get, reverse = True)

'''0. ii, unique words of two chosen categories'''
# Categorie romance
romance_words = brown.words(categories = 'romance')

romance_freq_dist = nltk.FreqDist(romance_words)

romance_sorted_words = sorted(romance_freq_dist, key = romance_freq_dist.get, reverse = True)

# Categorie learned
learned_words = brown.words(categories = 'learned')

learned_freq_dist = nltk.FreqDist(learned_words)

learned_sorted_words = sorted(learned_freq_dist, key = learned_freq_dist.get, reverse = True)

# Print number of tokens
print(len(tokens))

# print number of types
print(len(set(tokens)))

# print number of words
brown_corpus = brown.raw()

brown_words = nltk.word_tokenize(brown_corpus)

# remove non-word tokens (e.g., punctuation, digits, etc.)
brown_words = [word.lower() for word in brown_words if word.isalpha()]


